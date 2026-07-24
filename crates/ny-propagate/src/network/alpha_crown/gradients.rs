// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Gradient computation methods for alpha-CROWN optimization.
//!
//! Extracts SPSA and finite-difference gradient estimators from the main
//! optimization loop. The analytic and chain-rule methods remain inline
//! because they are trivial (1 line and ~15 lines respectively).
//!
//! Reference: designs/2026-02-14-alpha-crown-backward-dedup.md (Phase 2)

use ndarray::Array1;
use ny_core::Result;
use ny_tensor::BoundedTensor;

use crate::bounds::AlphaState;
use crate::network::alpha_crown_loop::finite_lower_sum;

/// Compute SPSA gradient estimates for alpha parameters.
///
/// SPSA (Simultaneous Perturbation Stochastic Approximation) perturbs ALL
/// parameters at once with random ±1 directions. Requires only 2 forward
/// passes per sample (vs 2*n for finite differences).
///
/// `single_pass_fn` evaluates a single forward+backward pass with the
/// current alpha state, returning concrete bounds.
pub(super) fn compute_spsa_gradients(
    alpha_state: &mut AlphaState,
    eps: f32,
    spsa_samples: usize,
    mut single_pass_fn: impl FnMut(&AlphaState) -> Result<BoundedTensor>,
) -> Result<Vec<Array1<f32>>> {
    use rand::RngExt;

    let num_relus = alpha_state.alphas.len();
    let mut avg_grads: Vec<Array1<f32>> = (0..num_relus)
        .map(|relu_idx| Array1::zeros(alpha_state.alphas[relu_idx].len()))
        .collect();

    // Save original alpha values for restoration (both lower and upper paths, #3393)
    let original_alphas: Vec<Array1<f32>> = alpha_state.alphas.clone();
    let original_alphas_upper: Vec<Array1<f32>> = alpha_state.alphas_upper.clone();

    // RNG must be created once before the loop so that each sample gets
    // a different perturbation vector. In test mode crate::random::rng()
    // returns a deterministically-seeded RNG; recreating it per-sample
    // would produce identical perturbations, making multi-sampling useless.
    let mut rng = crate::random::rng();

    // Wrap loop body in closure so `?` returns to the closure, not the
    // outer function. This ensures alpha restoration runs unconditionally,
    // even when CROWN propagation returns an error (#2554).
    let result = (|| -> Result<()> {
        // Average over multiple samples to reduce variance
        for _sample in 0..spsa_samples {
            // Generate random Bernoulli perturbation (+1 or -1) for each alpha
            let perturbations: Vec<Array1<f32>> = (0..num_relus)
                .map(|relu_idx| {
                    let n = alpha_state.alphas[relu_idx].len();
                    Array1::from_iter((0..n).map(|i| {
                        if alpha_state.unstable_mask[relu_idx][i] {
                            if rng.random_bool(0.5) {
                                1.0
                            } else {
                                -1.0
                            }
                        } else {
                            0.0 // Don't perturb stable neurons
                        }
                    }))
                })
                .collect();

            // Apply +eps perturbation from original (both lower and upper paths, #3393)
            for relu_idx in 0..num_relus {
                for i in 0..alpha_state.alphas[relu_idx].len() {
                    let p = eps * perturbations[relu_idx][i];
                    alpha_state.alphas[relu_idx][i] =
                        (original_alphas[relu_idx][i] + p).clamp(0.0, 1.0);
                    alpha_state.alphas_upper[relu_idx][i] =
                        (original_alphas_upper[relu_idx][i] + p).clamp(0.0, 1.0);
                }
            }
            // Propagate CROWN errors instead of swallowing them (#1935, #1941).
            let bounds_plus = single_pass_fn(alpha_state)?;
            let lower_plus: f32 = finite_lower_sum(bounds_plus.lower());

            // Apply -eps perturbation from original (both lower and upper paths, #3393)
            for relu_idx in 0..num_relus {
                for i in 0..alpha_state.alphas[relu_idx].len() {
                    let p = eps * perturbations[relu_idx][i];
                    alpha_state.alphas[relu_idx][i] =
                        (original_alphas[relu_idx][i] - p).clamp(0.0, 1.0);
                    alpha_state.alphas_upper[relu_idx][i] =
                        (original_alphas_upper[relu_idx][i] - p).clamp(0.0, 1.0);
                }
            }
            // Propagate CROWN errors instead of swallowing them (#1935, #1941).
            let bounds_minus = single_pass_fn(alpha_state)?;
            let lower_minus: f32 = finite_lower_sum(bounds_minus.lower());

            // SPSA gradient estimate: g_i = (f+ - f-) / (2 * eps * delta_i)
            let diff = lower_plus - lower_minus;
            for relu_idx in 0..num_relus {
                for i in 0..alpha_state.alphas[relu_idx].len() {
                    if alpha_state.unstable_mask[relu_idx][i]
                        && perturbations[relu_idx][i].abs() > 0.5
                    {
                        avg_grads[relu_idx][i] += diff / (2.0 * eps * perturbations[relu_idx][i]);
                    }
                }
            }
        }
        Ok(())
    })();

    // Always restore alpha state, including early error returns from `?` (#2554).
    restore_alpha_snapshot(alpha_state, &original_alphas, &original_alphas_upper);
    result?;

    // Average the gradients (.max(1) prevents div-by-zero when spsa_samples=0, #2079)
    let num_samples = spsa_samples.max(1) as f32;
    for grad in &mut avg_grads {
        *grad /= num_samples;
    }

    Ok(avg_grads)
}

/// Compute finite-difference gradient estimates for alpha parameters.
///
/// Perturbs each alpha individually using central differences:
///   grad[i] = (f(alpha+eps) - f(alpha-eps)) / (2*eps)
///
/// Accurate but O(n) forward passes per iteration where n is the number
/// of unstable neurons.
///
/// `single_pass_fn` evaluates a single forward+backward pass with the
/// current alpha state, returning concrete bounds.
pub(super) fn compute_finite_difference_gradients(
    alpha_state: &mut AlphaState,
    eps: f32,
    mut single_pass_fn: impl FnMut(&AlphaState) -> Result<BoundedTensor>,
) -> Result<Vec<Array1<f32>>> {
    // Save original alpha values for unconditional restoration (#2554, #3393).
    let original_alphas: Vec<Array1<f32>> = alpha_state.alphas.clone();
    let original_alphas_upper: Vec<Array1<f32>> = alpha_state.alphas_upper.clone();

    let result = (|| -> Result<Vec<Array1<f32>>> {
        let num_relus = alpha_state.alphas.len();
        let mut grads = Vec::with_capacity(num_relus);

        for relu_idx in 0..num_relus {
            let num_neurons = alpha_state.alphas[relu_idx].len();
            let mut grad = Array1::<f32>::zeros(num_neurons);

            // Only compute gradient for unstable neurons
            for neuron_idx in 0..num_neurons {
                if !alpha_state.unstable_mask[relu_idx][neuron_idx] {
                    continue;
                }

                let orig_alpha = alpha_state.alphas[relu_idx][neuron_idx];
                let orig_alpha_upper = alpha_state.alphas_upper[relu_idx][neuron_idx];

                // Compute f(alpha + eps) — perturb both paths together (#3393)
                alpha_state.alphas[relu_idx][neuron_idx] = (orig_alpha + eps).clamp(0.0, 1.0);
                alpha_state.alphas_upper[relu_idx][neuron_idx] =
                    (orig_alpha_upper + eps).clamp(0.0, 1.0);
                // Propagate CROWN errors instead of swallowing them (#1935, #1941).
                let bounds_plus = single_pass_fn(alpha_state)?;
                let lower_plus: f32 = finite_lower_sum(bounds_plus.lower());

                // Compute f(alpha - eps) — perturb both paths together (#3393)
                alpha_state.alphas[relu_idx][neuron_idx] = (orig_alpha - eps).clamp(0.0, 1.0);
                alpha_state.alphas_upper[relu_idx][neuron_idx] =
                    (orig_alpha_upper - eps).clamp(0.0, 1.0);
                // Propagate CROWN errors instead of swallowing them (#1935, #1941).
                let bounds_minus = single_pass_fn(alpha_state)?;
                let lower_minus: f32 = finite_lower_sum(bounds_minus.lower());

                // Restore original alpha (per-neuron, inside closure)
                alpha_state.alphas[relu_idx][neuron_idx] = orig_alpha;
                alpha_state.alphas_upper[relu_idx][neuron_idx] = orig_alpha_upper;

                // Central difference gradient
                grad[neuron_idx] = (lower_plus - lower_minus) / (2.0 * eps);
            }
            grads.push(grad);
        }

        Ok(grads)
    })();

    // Always restore alpha state, including early error returns from `?` (#2554).
    restore_alpha_snapshot(alpha_state, &original_alphas, &original_alphas_upper);
    result
}

/// Unconditionally restore alpha values from a snapshot.
///
/// Used after closure-wrapped gradient computation to ensure alpha state
/// is restored even when CROWN propagation returns an error (#2554).
/// Restores both lower and upper alpha paths (#3393).
fn restore_alpha_snapshot(
    alpha_state: &mut AlphaState,
    original_alphas: &[Array1<f32>],
    original_alphas_upper: &[Array1<f32>],
) {
    for (alpha, original) in alpha_state.alphas.iter_mut().zip(original_alphas.iter()) {
        alpha.assign(original);
    }
    for (alpha, original) in alpha_state
        .alphas_upper
        .iter_mut()
        .zip(original_alphas_upper.iter())
    {
        alpha.assign(original);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;
    use ny_core::NyError;

    /// Build a minimal AlphaState with one ReLU layer containing one unstable neuron.
    fn build_alpha_state() -> AlphaState {
        let pre_activation =
            BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
                .expect("pre-activation bounds should construct");
        let mut state =
            AlphaState::from_preactivation_bounds(&[pre_activation], &[0]).expect("alpha init");
        // Set a non-default alpha value so we can detect if restoration fails.
        state.alphas[0][0] = 0.7;
        state.unstable_mask[0][0] = true;
        state
    }

    #[test]
    fn spsa_restores_alphas_when_single_pass_errors() {
        let mut alpha_state = build_alpha_state();
        let original = alpha_state.alphas.clone();

        // Closure that always errors, simulating a CROWN propagation failure.
        let result = compute_spsa_gradients(&mut alpha_state, 1e-3, 1, |_| {
            Err(NyError::InternalError("simulated CROWN failure".into()))
        });

        assert!(result.is_err(), "expected propagation error");
        assert_eq!(
            alpha_state.alphas, original,
            "SPSA must restore alpha state on early error return (#2554)"
        );
    }

    #[test]
    fn finite_difference_restores_alphas_when_single_pass_errors() {
        let mut alpha_state = build_alpha_state();
        let original = alpha_state.alphas.clone();

        // Closure that always errors, simulating a CROWN propagation failure.
        let result = compute_finite_difference_gradients(&mut alpha_state, 1e-3, |_| {
            Err(NyError::InternalError("simulated CROWN failure".into()))
        });

        assert!(result.is_err(), "expected propagation error");
        assert_eq!(
            alpha_state.alphas, original,
            "finite differences must restore alpha state on early error return (#2554)"
        );
    }
}
