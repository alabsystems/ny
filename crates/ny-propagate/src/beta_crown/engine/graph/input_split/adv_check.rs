// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Lightweight PGD probe for early counterexample detection during BaB.
//!
//! Runs a few PGD restarts on the current domain to check if a concrete input
//! violates the property. This enables early SAT detection without waiting for
//! CROWN bounds to tighten.
//!
//! Reference: alpha-beta-CROWN attack_in_input_split.py:24-82
//! (`pgd_attack_on_domains`): selects worst domains, runs 5 restarts × 5 steps.
//! Ny adaptation: probes the current (worst-priority) domain since our
//! sequential BaB loop processes one domain at a time.

use std::time::Instant;

use ndarray::{ArrayD, IxDyn};
use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use tracing::debug;

use crate::beta_crown::result::ViolationWitness;
use crate::GraphNetwork;

/// Run the PGD probe on one domain.
///
/// Returns `Some(witness)` with the concrete point (and its concrete-forward
/// output) when a restart lands on a violating point, `None` otherwise.
///
/// SOUNDNESS (#advcheck-witness): the returned point is a CANDIDATE, not a
/// verdict. Before this change the caller learned only `true` and threw `x`
/// and `output` away, so the post-BaB confirmer had to re-search the ROOT box
/// for a point that was already in hand -- and a validated candidate routinely
/// downgraded to Unknown. Carrying it changes nothing about who decides: the
/// confirmer still re-evaluates the model at the point and checks the full
/// VNN-LIB constraints, and the trusted ONNX-Runtime gate still has the final
/// word on every scored `sat`.
pub(crate) fn try_adv_check_on_domain(
    graph: &GraphNetwork,
    input_bounds: &BoundedTensor,
    objective: &[f32],
    threshold: f32,
    verify_upper_bound: bool,
    deadline: Option<Instant>,
    seed_offset: u64,
    engine: Option<&dyn GemmEngine>,
) -> Result<Option<ViolationWitness>> {
    const PGD_RESTARTS: usize = 5;
    const PGD_STEPS: usize = 5;
    const SPSA_DELTA: f32 = 0.001;

    // Auto step size: max(domain_width) / 8 (reference: pgd_alpha="auto")
    let widths = input_bounds
        .upper()
        .iter()
        .zip(input_bounds.lower().iter())
        .map(|(u, l)| u - l);
    let max_width = widths.fold(0.0_f32, f32::max);
    let step_size = if max_width.is_finite() && max_width > 0.0 {
        max_width / 8.0
    } else {
        0.01
    };

    for restart in 0..PGD_RESTARTS {
        if deadline.is_some_and(|d| Instant::now() >= d) {
            break;
        }

        let mut rng = StdRng::seed_from_u64(
            42_u64
                .wrapping_add(restart as u64)
                .wrapping_add(seed_offset),
        );

        // Sample uniform random point in domain
        let shape = input_bounds.lower().shape().to_vec();
        let mut x = ArrayD::zeros(IxDyn(&shape));
        for (val, (lo, hi)) in x
            .iter_mut()
            .zip(input_bounds.lower().iter().zip(input_bounds.upper().iter()))
        {
            *val = lo + rng.random::<f32>() * (hi - lo);
        }

        for _step in 0..PGD_STEPS {
            // SPSA gradient estimation
            let n = x.len();
            let pert_vals: Vec<f32> = (0..n)
                .map(|_| {
                    if rng.random::<bool>() {
                        1.0_f32
                    } else {
                        -1.0_f32
                    }
                })
                .collect();
            let pert = ArrayD::from_shape_vec(IxDyn(&shape), pert_vals).map_err(|e| {
                NyError::InternalError(format!("adv_check perturbation reshape: {}", e))
            })?;

            let x_plus = &x + &pert * SPSA_DELTA;
            let x_minus = &x - &pert * SPSA_DELTA;

            // Evaluate graph at perturbed points via a TRUE concrete (point) forward.
            // A whole-box IBP forward on a point input returns a non-degenerate box
            // (per-node soundness widening amplified by the deep DAG), so its
            // `.lower()` is not the network value — it fabricates false violations
            // that ORT rejects (cgan_2023 unknown-downgrade). #cgan-eval.
            let out_plus = {
                let b = BoundedTensor::concrete(x_plus)?;
                graph.propagate_concrete_point(&b, engine, deadline)?
            };
            let out_minus = {
                let b = BoundedTensor::concrete(x_minus)?;
                graph.propagate_concrete_point(&b, engine, deadline)?
            };

            // Compute objective: objective · output
            let obj_plus: f32 = objective
                .iter()
                .zip(out_plus.lower().iter())
                .map(|(a, b)| a * b)
                .sum();
            let obj_minus: f32 = objective
                .iter()
                .zip(out_minus.lower().iter())
                .map(|(a, b)| a * b)
                .sum();

            let diff = obj_plus - obj_minus;
            if !diff.is_finite() {
                continue;
            }
            let grad = &pert * (diff / (2.0 * SPSA_DELTA));

            // Step direction: minimize objective for standard mode (find obj <= threshold),
            // maximize for upper_bound mode (find obj >= threshold).
            if verify_upper_bound {
                x = &x + &grad * step_size;
            } else {
                x = &x - &grad * step_size;
            }

            // Project back to input bounds, replacing NaN with lower bound
            for (val, (lo, hi)) in x
                .iter_mut()
                .zip(input_bounds.lower().iter().zip(input_bounds.upper().iter()))
            {
                if val.is_nan() {
                    *val = *lo;
                } else {
                    *val = val.clamp(*lo, *hi);
                }
            }
        }

        // Check if violation found at final point (TRUE concrete point forward).
        // `x` is cloned into the box rather than moved: on a hit it becomes the
        // carried witness instead of being dropped on the floor.
        let output = {
            let b = BoundedTensor::concrete(x.clone())?;
            graph.propagate_concrete_point(&b, engine, deadline)?
        };
        let obj_val: f32 = objective
            .iter()
            .zip(output.lower().iter())
            .map(|(a, b)| a * b)
            .sum();

        if !obj_val.is_finite() {
            continue;
        }

        let is_violation = if verify_upper_bound {
            obj_val >= threshold
        } else {
            obj_val <= threshold
        };

        if is_violation {
            debug!(
                "adv_check PGD: found counterexample at restart {} (obj={:.6}, threshold={:.6})",
                restart, obj_val, threshold
            );
            // Carry the exact point and the exact concrete-forward output that
            // produced `obj_val`. `output` is degenerate (concrete point in,
            // concrete point out), so `.lower()` IS the network value the
            // violation decision above was taken on.
            return Ok(Some(ViolationWitness {
                input_shape: x.shape().to_vec(),
                input: x.iter().copied().collect(),
                output: output.lower().iter().copied().collect(),
            }));
        }
    }

    Ok(None)
}

pub(crate) fn try_adv_check_on_input_bounds_batch<'a, I>(
    graph: &GraphNetwork,
    input_bounds_batch: I,
    objective: &[f32],
    threshold: f32,
    verify_upper_bound: bool,
    deadline: Option<Instant>,
    seed_offset: u64,
    engine: Option<&dyn GemmEngine>,
) -> Result<Option<ViolationWitness>>
where
    I: IntoIterator<Item = &'a BoundedTensor>,
{
    for (offset, input_bounds) in input_bounds_batch.into_iter().enumerate() {
        if let Some(witness) = try_adv_check_on_domain(
            graph,
            input_bounds,
            objective,
            threshold,
            verify_upper_bound,
            deadline,
            seed_offset.wrapping_add(offset as u64),
            engine,
        )? {
            return Ok(Some(witness));
        }
    }
    Ok(None)
}

/// Check interval for adv_check PGD probes during BaB.
///
/// In the reference, adv_check runs once per batch iteration (~batch_size domains).
/// Our sequential loop processes one domain at a time, so we check at a fixed
/// interval to avoid per-domain overhead. 1000 domains ≈ 1 probe per ~0.1s on
/// small networks.
pub(crate) const ADV_CHECK_INTERVAL: usize = 1000;
