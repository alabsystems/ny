// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Alpha parameter updates for all node types in DAG α-CROWN.
//!
//! Handles ReLU, S-shaped, Sqrt, BilinearCrown, and MulBinary alpha updates
//! using Adam or SGD optimizers. Includes a generic Adam update for ndarray-based
//! alpha maps that unifies the bilinear (Array4) and MulBinary (Array2) update logic.

use crate::bounds::alpha_reciprocal::ReciprocalGradients;
use crate::bounds::{
    AdamParams, AlphaCrownConfig, MonotoneSShapedGradients, Optimizer, SqrtGradients,
};

use ndarray::{Array1, Array2, Array4};
use ny_core::{NyError, Result};
use std::collections::{BTreeMap, HashMap};
use tracing::warn;

use super::super::runtime_state::DagAlphaRuntimeState;

/// Generic Adam update for ndarray alpha parameters.
///
/// Covers BilinearCrown (Array4<f32>) and MulBinary (Array2<f32>).
/// Gradient ascent: negates gradients internally (maximizing lower bound).
///
/// After calling this, the caller should reset gradient accumulators with `fill(0.0)`.
#[allow(clippy::too_many_arguments)]
fn adam_update_alpha_map<D: ndarray::Dimension>(
    alphas: &mut HashMap<String, ndarray::Array<f32, D>>,
    grads: &HashMap<String, ndarray::Array<f32, D>>,
    adam_m: &mut HashMap<String, ndarray::Array<f32, D>>,
    adam_v: &mut HashMap<String, ndarray::Array<f32, D>>,
    adam_params: &AdamParams,
    label: &str,
    iter: usize,
    total_gradient_skips: &mut usize,
) {
    let t_f = adam_params.t.max(1) as f32;
    let bias_correction1 = (1.0_f32 - adam_params.beta1.powf(t_f)).max(f32::EPSILON);
    let bias_correction2 = (1.0_f32 - adam_params.beta2.powf(t_f)).max(f32::EPSILON);

    for (name, grad) in grads {
        if grad.iter().any(|v| !v.is_finite()) {
            warn!(
                "DAG α-CROWN iter {}: skipping {} '{}' gradient — non-finite (#2835)",
                iter, label, name
            );
            *total_gradient_skips += 1;
            continue;
        }
        if let (Some(alpha), Some(m), Some(v)) = (
            alphas.get_mut(name),
            adam_m.get_mut(name),
            adam_v.get_mut(name),
        ) {
            ndarray::Zip::from(alpha.view_mut())
                .and(grad.view())
                .and(m.view_mut())
                .and(v.view_mut())
                .for_each(|a, &g, m_val, v_val| {
                    let neg_g = -g;
                    *m_val = adam_params.beta1 * *m_val + (1.0 - adam_params.beta1) * neg_g;
                    *v_val = adam_params.beta2 * *v_val + (1.0 - adam_params.beta2) * neg_g * neg_g;
                    let m_hat = *m_val / bias_correction1;
                    let v_hat = *v_val / bias_correction2;
                    *a -= adam_params.learning_rate * m_hat / (v_hat.sqrt() + adam_params.epsilon);
                    *a = a.clamp(0.0, 1.0);
                    if a.is_nan() {
                        *a = 0.5;
                        *m_val = 0.0;
                        *v_val = 0.0;
                    }
                });
        }
    }
}

/// Update all alpha parameters for one iteration of the optimization loop.
///
/// Dispatches to per-node-type update logic:
/// - ReLU: Adam or SGD with separate upper-path gradients
/// - S-shaped (Sigmoid/Tanh): Adam or SGD
/// - Sqrt: Adam or SGD
/// - BilinearCrown: Adam (generic)
/// - MulBinary: Adam (generic)
/// - INVPROP: ny clipping
#[allow(clippy::too_many_arguments)]
pub(super) fn update_all_alphas(
    runtime: &mut DagAlphaRuntimeState,
    config: &AlphaCrownConfig,
    numerical_gradients: &[Array1<f32>],
    numerical_gradients_upper: &[Array1<f32>],
    s_shaped_grads: &BTreeMap<String, MonotoneSShapedGradients>,
    sqrt_grads: &BTreeMap<String, SqrtGradients>,
    reciprocal_grads: &BTreeMap<String, ReciprocalGradients>,
    bilinear_alphas: &mut HashMap<String, Array4<f32>>,
    bilinear_grads: &mut HashMap<String, Array4<f32>>,
    bilinear_adam_m: &mut HashMap<String, Array4<f32>>,
    bilinear_adam_v: &mut HashMap<String, Array4<f32>>,
    mul_binary_alphas: &mut HashMap<String, Array2<f32>>,
    mul_binary_grads: &mut HashMap<String, Array2<f32>>,
    mul_binary_adam_m: &mut HashMap<String, Array2<f32>>,
    mul_binary_adam_v: &mut HashMap<String, Array2<f32>>,
    has_bilinear: bool,
    has_mul_binary: bool,
    invprop_enabled: bool,
    lr: f32,
    iter: usize,
    total_gradient_skips: &mut usize,
) -> Result<()> {
    let adam_params = config.adam_params(lr, iter + 1);

    // ReLU alpha update
    for (relu_idx, (grad, grad_upper)) in numerical_gradients
        .iter()
        .zip(numerical_gradients_upper.iter())
        .enumerate()
    {
        // Guard: reject non-finite gradients before optimizer update.
        // Without this gate, NaN/Inf gradients from numerical instability
        // in the backward pass silently enter the optimizer state (m/v for
        // Adam, velocity for SGD). Skipping preserves the current alpha
        // (a better heuristic than silent reset). (#2809, #2835)
        if grad.iter().any(|v| !v.is_finite()) {
            warn!(
                "DAG α-CROWN iter {}: skipping ReLU {} gradient update — non-finite values detected (#2835)",
                iter, relu_idx
            );
            *total_gradient_skips += 1;
            continue;
        }
        let neg_grad = grad.mapv(|v: f32| -v);
        // Upper gradient: separate from lower (#3393).
        let neg_grad_upper = if grad_upper.iter().any(|v| !v.is_finite()) {
            neg_grad.clone()
        } else {
            grad_upper.mapv(|v: f32| -v)
        };
        let node_name = runtime
            .relu_name(relu_idx)
            .ok_or_else(|| {
                NyError::InternalError(format!(
                    "missing DAG ReLU node for gradient index {relu_idx}"
                ))
            })?
            .to_string();
        // Channel-only alpha reduction (#4404): when full_conv_alpha is False,
        // gradients are per-neuron [C*H*W] but alpha is per-channel [C].
        // Reduce gradients to match alpha shape before optimizer update.
        let neg_grad = runtime.graph().reduce_gradient(&node_name, &neg_grad);
        let neg_grad_upper = runtime.graph().reduce_gradient(&node_name, &neg_grad_upper);
        match config.optimizer {
            Optimizer::Adam => {
                runtime
                    .graph_mut()
                    .update_adam(&node_name, &neg_grad, &adam_params);
                runtime
                    .graph_mut()
                    .update_adam_upper(&node_name, &neg_grad_upper, &adam_params);
            }
            Optimizer::Sgd => {
                let momentum = if config.use_momentum {
                    config.momentum
                } else {
                    0.0
                };
                runtime
                    .graph_mut()
                    .update(&node_name, &neg_grad, lr, momentum);
                runtime
                    .graph_mut()
                    .update_upper(&node_name, &neg_grad_upper, lr, momentum);
            }
        }
    }
    if std::env::var("NY_BETA_GPU_PROBE").ok().as_deref() == Some("1") {
        // #w4-root-alpha diagnosis: did THIS iteration's ReLU updates land in
        // the runtime alpha state? (root-state n_interior=0 with nonzero GPU
        // grads means either dead updates or a discarded state copy.)
        let (n, interior) = runtime.graph().relu_lower_alpha_interior_count();
        eprintln!("[warmup-alpha] iter={iter} lr={lr:e} n={n} interior={interior}");
    }

    // S-shaped (Sigmoid/Tanh) alpha update
    for (node_name, grad) in s_shaped_grads {
        if grad.any_non_finite() {
            warn!(
                "DAG α-CROWN iter {}: skipping monotone S-shaped '{}' gradient update — non-finite values detected",
                iter, node_name
            );
            *total_gradient_skips += 1;
            continue;
        }
        let neg_grad = grad.negate();
        if let Some(alpha) = runtime.graph_mut().monotone_s_shaped_alpha_mut(node_name) {
            match config.optimizer {
                Optimizer::Adam => alpha.update_adam(&neg_grad, &adam_params),
                Optimizer::Sgd => {
                    let momentum = if config.use_momentum {
                        config.momentum
                    } else {
                        0.0
                    };
                    alpha.update_sgd(&neg_grad, lr, momentum);
                }
            }
        }
    }

    // Sqrt alpha update
    for (node_name, grad) in sqrt_grads {
        if grad.any_non_finite() {
            warn!(
                "DAG α-CROWN iter {}: skipping sqrt '{}' gradient update — non-finite values detected",
                iter, node_name
            );
            *total_gradient_skips += 1;
            continue;
        }
        let neg_grad = grad.negate();
        if let Some(alpha) = runtime.graph_mut().sqrt_alpha_mut(node_name) {
            match config.optimizer {
                Optimizer::Adam => alpha.update_adam(&neg_grad, &adam_params),
                Optimizer::Sgd => {
                    let momentum = if config.use_momentum {
                        config.momentum
                    } else {
                        0.0
                    };
                    alpha.update_sgd(&neg_grad, lr, momentum);
                }
            }
        }
    }

    // Reciprocal alpha update (#4399 Slice 2)
    for (node_name, grad) in reciprocal_grads {
        if grad.any_non_finite() {
            warn!(
                "DAG α-CROWN iter {}: skipping reciprocal '{}' gradient update — non-finite values detected",
                iter, node_name
            );
            *total_gradient_skips += 1;
            continue;
        }
        let neg_grad = grad.negate();
        if let Some(alpha) = runtime.graph_mut().reciprocal_alpha_mut(node_name) {
            match config.optimizer {
                Optimizer::Adam => alpha.update_adam(&neg_grad, &adam_params),
                Optimizer::Sgd => {
                    let momentum = if config.use_momentum {
                        config.momentum
                    } else {
                        0.0
                    };
                    alpha.update_sgd(&neg_grad, lr, momentum);
                }
            }
        }
    }

    // Bilinear Adam update (#3287). Always uses Adam (matching batched path
    // in alpha_crown_batched/spsa.rs). Gradient ascent: negate gradients.
    if has_bilinear {
        adam_update_alpha_map(
            bilinear_alphas,
            bilinear_grads,
            bilinear_adam_m,
            bilinear_adam_v,
            &adam_params,
            "bilinear",
            iter,
            total_gradient_skips,
        );
        // Reset bilinear gradient accumulators for next iteration.
        for grad in bilinear_grads.values_mut() {
            grad.fill(0.0);
        }
    }

    // MulBinary Adam update (#3439 Phase 3). Same pattern as bilinear.
    if has_mul_binary {
        adam_update_alpha_map(
            mul_binary_alphas,
            mul_binary_grads,
            mul_binary_adam_m,
            mul_binary_adam_v,
            &adam_params,
            "mul_binary",
            iter,
            total_gradient_skips,
        );
        for grad in mul_binary_grads.values_mut() {
            grad.fill(0.0);
        }
    }

    // Clip gammas to enforce non-negativity (INVPROP constraint).
    // Matches alpha_crown_loop.rs:220-223. Negative gammas invert constraint
    // contributions, producing bounds tighter than sound.
    if invprop_enabled {
        runtime.clip_gammas();
    }

    Ok(())
}
