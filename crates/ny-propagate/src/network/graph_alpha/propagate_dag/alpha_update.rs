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

// Under `targo trust ... --contracts` tRustc supplies the first-class contract
// attributes and STATICALLY verifies them; otherwise the `trust` facade provides
// the no-op form. Same pattern as `crates/ny-cert/src/selfcheck.rs`.
#[cfg(trust_verify)]
use core::contracts::ensures;
#[cfg(not(trust_verify))]
use trust::ensures;

use super::super::runtime_state::DagAlphaRuntimeState;

/// One clamped α step — the ONLY way an α value is ever written.
///
/// # Why this is a contract and not just a `clamp`
///
/// α is the slope of the ReLU LOWER envelope `relu z ≥ α·z`. That envelope is valid
/// for **every** `α ∈ [0,1]` and for no `α` outside it. So this single postcondition
/// is what separates "the α optimizer is a bound-QUALITY component" from "the α
/// optimizer is part of the soundness argument".
///
/// `#[ensures]` states the locally-provable range property.
/// `#[trust::cite(crownproof::relu_lower)]` grounds the entailment in the
/// Clean-kernel-checked theorem
///
/// ```text
/// theorem relu_lower (alpha z : ℚ) (h0 : 0 ≤ alpha) (h1 : alpha ≤ 1) :
///     alpha * z ≤ relu z
/// ```
///
/// (`Crownproof.Basic`, in the pinned Clean dependency; axioms
/// `[propext, Classical.choice, Quot.sound]`, no `sorryAx`).
///
/// Together: **no gradient, however wrong, can make ny unsound through α.** That is
/// not a style claim — it is the load-bearing licence for changing the α gradient,
/// which is currently provably broken. `gradients.rs:91,115,119` computes
/// `g_i = Σ_{j : A[j,i] > 0} A[j,i] · l_i` with `l_i < 0` forced by the
/// unstable-neuron guard, so `g_i ≤ 0` for every neuron, objective and iteration —
/// a constant-sign field carrying no directional information, which drives every α
/// monotonically to the `0` clamp. Measured on cifar100 `idx_7704`: `best_impr =
/// 0.000e0` at every iteration, and sweeping `lr_alpha` over 0.25 / 0.05 / 0.01
/// leaves `best_lower_sum` bit-identical at −3564.689453. Machine-checked in
/// `crates/ny-cert/proofs/lean/NyProof/AlphaGradientDefect.lean`
/// (`local_rule_nonpos`, `clamped_step_nonincreasing`,
/// `local_rule_sign_can_be_wrong`, `alpha_sound_regardless`).
///
/// Because of the contract below, replacing that gradient is a bound-quality change
/// with **zero** false-`unsat` exposure — the one direction that costs −150.
///
/// VERIFICATION STATUS of the `#[ensures]`, measured 2026-08-03 — it is NOT yet
/// machine-checked, and the reason is upstream, not here. `targo trust check`
/// does reach this crate (350 functions, 470 obligations, 290 proved), but no
/// postcondition on a multi-branch function can be proved by the current
/// toolchain at any level:
///
/// * L0 refutes them. `#[ensures(|r: &i32| *r >= 0)] fn f(x) { if x<0 {1} else {2} }`
///   reports `postcond FAILED`, pinning the merged return SSA name to −1 — a
///   value no branch returns. L0 should not be discharging these at all;
///   `trust-ir-bridge/src/flip.rs` asserts `[L0] postcondition is an L1
///   obligation`, so L0 evaluates an L1 obligation without L1's per-predecessor
///   guards.
/// * L1 declines them: "trust-wp native pure verifier does not support
///   obligation ... TrustContractBundle lowering into TrustWpPureExprV1 ... is
///   required".
///
/// Reproducer committed upstream at `~/trust`
/// `examples/contracts/postcondition-branch-repro` (`41962e43e1`). This function
/// branches (NaN guard, then range), so it is squarely in the refuted class.
/// Until that is fixed the `#[ensures]` is a precise, executable statement of
/// the invariant whose PROOF lives in Lean; do not read it as solver-checked,
/// and do not "fix" a red verdict here by weakening it.
#[ensures(|r: &f32| *r >= 0.0 && *r <= 1.0)]
#[trust::cite(crownproof::relu_lower)]
#[inline]
pub(crate) fn clamp_alpha_to_envelope_domain(candidate: f32) -> f32 {
    // NaN maps to the interior point 0.5 rather than propagating: `f32::clamp`
    // panics on a NaN bound and returns NaN for a NaN input, and a NaN α would make
    // the envelope meaningless. 0.5 is in range, so the postcondition holds on every
    // path. This mirrors the NaN reset the Adam loop already performs.
    if candidate.is_nan() {
        return 0.5;
    }
    candidate.clamp(0.0, 1.0)
}

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
                    // Single contracted write site (#alpha-envelope-domain): the
                    // `#[ensures]` on this call is what guarantees every stored α
                    // lies in the envelope's domain [0,1], which by the cited
                    // `crownproof::relu_lower` makes the resulting relaxation sound
                    // for ANY gradient. Reset the Adam moments on NaN as before, so
                    // a poisoned history cannot persist past the repair.
                    let was_nan = a.is_nan();
                    *a = clamp_alpha_to_envelope_domain(*a);
                    if was_nan {
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
    let grad_probe = std::env::var("NY_ALPHA_GRAD_PROBE").ok().as_deref() == Some("1");

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
        let raw_len = neg_grad.len();
        let raw_nz = neg_grad.iter().filter(|v| **v != 0.0).count();
        let neg_grad = runtime.graph().reduce_gradient(&node_name, &neg_grad);
        let neg_grad_upper = runtime.graph().reduce_gradient(&node_name, &neg_grad_upper);
        if grad_probe {
            let n = neg_grad.len();
            let nz = neg_grad.iter().filter(|v| **v != 0.0).count();
            let absmax = neg_grad.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let absmean = if n > 0 {
                neg_grad.iter().map(|v| v.abs()).sum::<f32>() / n as f32
            } else {
                0.0
            };
            let (a_lo, a_hi, a_int, a_n) = runtime.graph().alpha(&node_name).map_or(
                (f32::NAN, f32::NAN, 0usize, 0usize),
                |a| {
                    (
                        a.iter().copied().fold(f32::INFINITY, f32::min),
                        a.iter().copied().fold(f32::NEG_INFINITY, f32::max),
                        a.iter().filter(|v| **v != 0.0 && **v != 1.0).count(),
                        a.len(),
                    )
                },
            );
            let unstable = runtime
                .graph()
                .relu_unstable_mask(&node_name)
                .map_or(0usize, |m| m.iter().filter(|b| **b).count());
            eprintln!(
                "[grad-probe] iter={iter} relu={relu_idx} name={node_name} raw_len={raw_len} \
                 raw_nz={raw_nz} n={n} nz={nz} absmean={absmean:.3e} absmax={absmax:.3e} \
                 alpha_n={a_n} alpha_interior={a_int} alpha_range=[{a_lo:.3},{a_hi:.3}] \
                 unstable_mask={unstable}"
            );
        }
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
    if crate::beta_gpu_probe_armed() {
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
