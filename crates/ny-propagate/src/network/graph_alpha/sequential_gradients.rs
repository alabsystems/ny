// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Gradient computation helpers for sequential graph alpha-CROWN.

use crate::bounds::{AlphaCrownConfig, AlphaState, GradientMethod};
use crate::network::alpha_crown_loop::finite_lower_sum;
use crate::network::core::GraphNetwork;

use ndarray::Array1;
use ny_core::{GemmEngine, Result};
use ny_tensor::BoundedTensor;
use std::collections::HashMap;
use tracing::warn;

use super::alpha_projection::{graph_alpha_state_from_sequential, relu_names_in_alpha_order};
use super::propagate_sequential::SequentialSinglePassRequest;

const SEQUENTIAL_GRAD_EPS: f32 = 1e-3;

#[cfg(test)]
thread_local! {
    static SEQUENTIAL_GRADIENT_SANITIZED_VALUES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static SEQUENTIAL_CHAIN_FALLBACKS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn sanitize_non_finite_gradients(
    mut gradients: Vec<Array1<f32>>,
    method: &str,
    iter: usize,
) -> Vec<Array1<f32>> {
    let mut sanitized = 0usize;
    for gradient in &mut gradients {
        for value in gradient.iter_mut() {
            if !value.is_finite() {
                *value = 0.0;
                sanitized += 1;
            }
        }
    }

    if sanitized > 0 {
        warn!(
            "GraphNetwork: {method} produced {sanitized} non-finite gradient values at iteration {iter}; zeroing them before optimizer update (#2544)"
        );
        #[cfg(test)]
        SEQUENTIAL_GRADIENT_SANITIZED_VALUES.with(|slot| {
            slot.set(slot.get() + sanitized);
        });
    }

    gradients
}

fn record_chain_rule_fallback(iter: usize, reason: &str) {
    warn!(
        "GraphNetwork: AnalyticChain falling back to local gradients at iteration {iter}: {reason} (#2544)"
    );
    #[cfg(test)]
    SEQUENTIAL_CHAIN_FALLBACKS.with(|slot| {
        slot.set(slot.get() + 1);
    });
}

#[cfg(test)]
fn reset_sequential_gradient_diagnostics() {
    SEQUENTIAL_GRADIENT_SANITIZED_VALUES.with(|slot| slot.set(0));
    SEQUENTIAL_CHAIN_FALLBACKS.with(|slot| slot.set(0));
}

#[cfg(test)]
fn sequential_gradient_sanitized_values() -> usize {
    SEQUENTIAL_GRADIENT_SANITIZED_VALUES.with(std::cell::Cell::get)
}

#[cfg(test)]
fn sequential_chain_fallbacks() -> usize {
    SEQUENTIAL_CHAIN_FALLBACKS.with(std::cell::Cell::get)
}

/// Compute alpha-CROWN gradients for sequential graph optimization.
#[allow(clippy::too_many_arguments)]
pub(super) fn compute_sequential_gradients(
    graph: &GraphNetwork,
    config: &AlphaCrownConfig,
    alpha_state: &mut AlphaState,
    input: &BoundedTensor,
    node_bounds: &HashMap<String, BoundedTensor>,
    exec_order: &[String],
    output_dim: usize,
    relu_name_to_idx: &HashMap<String, usize>,
    engine: Option<&dyn GemmEngine>,
    analytic_gradients: &[Array1<f32>],
    iter: usize,
) -> Result<Vec<Array1<f32>>> {
    let (method, gradients) = match config.gradient_method {
        GradientMethod::Spsa => (
            "SPSA",
            spsa_gradients(
                graph,
                config,
                alpha_state,
                input,
                node_bounds,
                exec_order,
                output_dim,
                relu_name_to_idx,
                engine,
            )?,
        ),
        GradientMethod::FiniteDifferences => (
            "FiniteDifferences",
            finite_difference_gradients(
                graph,
                alpha_state,
                input,
                node_bounds,
                exec_order,
                output_dim,
                relu_name_to_idx,
                engine,
            )?,
        ),
        GradientMethod::Analytic => ("Analytic", analytic_gradients.to_vec()),
        GradientMethod::AnalyticChain => (
            "AnalyticChain",
            analytic_chain_gradients(
                graph,
                alpha_state,
                input,
                node_bounds,
                exec_order,
                output_dim,
                relu_name_to_idx,
                engine,
                analytic_gradients,
                iter,
            )?,
        ),
    };

    Ok(sanitize_non_finite_gradients(gradients, method, iter))
}

#[allow(clippy::too_many_arguments)]
fn spsa_gradients(
    graph: &GraphNetwork,
    config: &AlphaCrownConfig,
    alpha_state: &mut AlphaState,
    input: &BoundedTensor,
    node_bounds: &HashMap<String, BoundedTensor>,
    exec_order: &[String],
    output_dim: usize,
    relu_name_to_idx: &HashMap<String, usize>,
    engine: Option<&dyn GemmEngine>,
) -> Result<Vec<Array1<f32>>> {
    use rand::RngExt;

    let mut rng = crate::random::rng();
    let mut avg_grads: Vec<Array1<f32>> = alpha_state
        .alphas
        .iter()
        .map(|a| Array1::zeros(a.len()))
        .collect();
    let original_alphas: Vec<Array1<f32>> = alpha_state.alphas.clone();
    let original_alphas_upper: Vec<Array1<f32>> = alpha_state.alphas_upper.clone();

    let result = (|| -> Result<()> {
        // Average over multiple samples to reduce variance.
        for _sample in 0..config.spsa_samples {
            // Generate random Bernoulli perturbation (+1/-1) for each alpha.
            // Uses zip to avoid unchecked indexing (#2499).
            let perturbations: Vec<Array1<f32>> = alpha_state
                .alphas
                .iter()
                .zip(alpha_state.unstable_mask.iter())
                .map(|(alphas, mask)| {
                    Array1::from_iter(alphas.iter().zip(mask.iter()).map(|(_, &unstable)| {
                        if unstable {
                            if rng.random_bool(0.5) {
                                1.0
                            } else {
                                -1.0
                            }
                        } else {
                            0.0
                        }
                    }))
                })
                .collect();

            // Apply +epsilon perturbation from original (both paths, #3393).
            for ((alpha, orig), pert) in alpha_state
                .alphas
                .iter_mut()
                .zip(original_alphas.iter())
                .zip(perturbations.iter())
            {
                for ((a, &o), &p) in alpha.iter_mut().zip(orig.iter()).zip(pert.iter()) {
                    *a = (o + SEQUENTIAL_GRAD_EPS * p).clamp(0.0, 1.0);
                }
            }
            for ((alpha, orig), pert) in alpha_state
                .alphas_upper
                .iter_mut()
                .zip(original_alphas_upper.iter())
                .zip(perturbations.iter())
            {
                for ((a, &o), &p) in alpha.iter_mut().zip(orig.iter()).zip(pert.iter()) {
                    *a = (o + SEQUENTIAL_GRAD_EPS * p).clamp(0.0, 1.0);
                }
            }
            // Propagate CROWN errors instead of swallowing them (#2064, #1941).
            let bounds_plus = graph.propagate_alpha_crown_single_pass_sequential_graph(
                SequentialSinglePassRequest {
                    input,
                    node_bounds,
                    exec_order,
                    output_dim,
                    relu_name_to_idx,
                    alpha_state,
                    engine,
                    deadline: None, // No deadline for SPSA gradient estimation (#3393)
                },
            )?;
            let lower_plus: f32 = finite_lower_sum(bounds_plus.lower());

            // Apply -epsilon perturbation from original (both paths, #3393).
            for ((alpha, orig), pert) in alpha_state
                .alphas
                .iter_mut()
                .zip(original_alphas.iter())
                .zip(perturbations.iter())
            {
                for ((a, &o), &p) in alpha.iter_mut().zip(orig.iter()).zip(pert.iter()) {
                    *a = (o - SEQUENTIAL_GRAD_EPS * p).clamp(0.0, 1.0);
                }
            }
            for ((alpha, orig), pert) in alpha_state
                .alphas_upper
                .iter_mut()
                .zip(original_alphas_upper.iter())
                .zip(perturbations.iter())
            {
                for ((a, &o), &p) in alpha.iter_mut().zip(orig.iter()).zip(pert.iter()) {
                    *a = (o - SEQUENTIAL_GRAD_EPS * p).clamp(0.0, 1.0);
                }
            }
            // Propagate CROWN errors instead of swallowing them (#2064, #1941).
            let bounds_minus = graph.propagate_alpha_crown_single_pass_sequential_graph(
                SequentialSinglePassRequest {
                    input,
                    node_bounds,
                    exec_order,
                    output_dim,
                    relu_name_to_idx,
                    alpha_state,
                    engine,
                    deadline: None, // No deadline for SPSA gradient estimation (#3393)
                },
            )?;
            let lower_minus: f32 = finite_lower_sum(bounds_minus.lower());

            // SPSA gradient estimate (#2499: zip eliminates indexing).
            let diff = lower_plus - lower_minus;
            for ((grad, mask), pert) in avg_grads
                .iter_mut()
                .zip(alpha_state.unstable_mask.iter())
                .zip(perturbations.iter())
            {
                for ((g, &unstable), &p) in grad.iter_mut().zip(mask.iter()).zip(pert.iter()) {
                    if unstable && p.abs() > 0.5 {
                        *g += diff / (2.0 * SEQUENTIAL_GRAD_EPS * p);
                    }
                }
            }
        }
        Ok(())
    })();

    // Always restore alpha state, including early error returns from `?` (#3393).
    restore_alpha_snapshot(alpha_state, &original_alphas, &original_alphas_upper);
    result?;

    // Average the gradients.
    let num_samples = config.spsa_samples.max(1) as f32;
    for grad in &mut avg_grads {
        *grad /= num_samples;
    }
    Ok(avg_grads)
}

#[allow(clippy::too_many_arguments)]
fn finite_difference_gradients(
    graph: &GraphNetwork,
    alpha_state: &mut AlphaState,
    input: &BoundedTensor,
    node_bounds: &HashMap<String, BoundedTensor>,
    exec_order: &[String],
    output_dim: usize,
    relu_name_to_idx: &HashMap<String, usize>,
    engine: Option<&dyn GemmEngine>,
) -> Result<Vec<Array1<f32>>> {
    let original_alphas: Vec<Array1<f32>> = alpha_state.alphas.clone();
    let original_alphas_upper: Vec<Array1<f32>> = alpha_state.alphas_upper.clone();
    let result = (|| -> Result<Vec<Array1<f32>>> {
        // Per-neuron central differences (#2499: use enumerate+zip to
        // bounds-check relu_idx against alphas/mask vectors).
        let mut grads = Vec::with_capacity(alpha_state.alphas.len());
        for (relu_idx, mask) in alpha_state
            .unstable_mask
            .iter()
            .enumerate()
            .take(alpha_state.alphas.len())
        {
            let num_neurons = alpha_state.alphas[relu_idx].len();
            let mut grad = Array1::<f32>::zeros(num_neurons);

            for (neuron_idx, &unstable) in mask.iter().enumerate().take(num_neurons) {
                if !unstable {
                    continue;
                }

                let orig_alpha = alpha_state.alphas[relu_idx][neuron_idx];
                let orig_alpha_upper = alpha_state.alphas_upper[relu_idx][neuron_idx];

                // f(alpha + epsilon) — perturb both paths (#3393)
                alpha_state.alphas[relu_idx][neuron_idx] =
                    (orig_alpha + SEQUENTIAL_GRAD_EPS).clamp(0.0, 1.0);
                alpha_state.alphas_upper[relu_idx][neuron_idx] =
                    (orig_alpha_upper + SEQUENTIAL_GRAD_EPS).clamp(0.0, 1.0);
                // Propagate CROWN errors (#2064, #1941).
                let bounds_plus = graph.propagate_alpha_crown_single_pass_sequential_graph(
                    SequentialSinglePassRequest {
                        input,
                        node_bounds,
                        exec_order,
                        output_dim,
                        relu_name_to_idx,
                        alpha_state,
                        engine,
                        deadline: None, // No deadline for FiniteDiff gradient estimation (#3393)
                    },
                )?;
                let lower_plus: f32 = finite_lower_sum(bounds_plus.lower());

                // f(alpha - epsilon) — perturb both paths (#3393)
                alpha_state.alphas[relu_idx][neuron_idx] =
                    (orig_alpha - SEQUENTIAL_GRAD_EPS).clamp(0.0, 1.0);
                alpha_state.alphas_upper[relu_idx][neuron_idx] =
                    (orig_alpha_upper - SEQUENTIAL_GRAD_EPS).clamp(0.0, 1.0);
                // Propagate CROWN errors (#2064, #1941).
                let bounds_minus = graph.propagate_alpha_crown_single_pass_sequential_graph(
                    SequentialSinglePassRequest {
                        input,
                        node_bounds,
                        exec_order,
                        output_dim,
                        relu_name_to_idx,
                        alpha_state,
                        engine,
                        deadline: None, // No deadline for FiniteDiff gradient estimation (#3393)
                    },
                )?;
                let lower_minus: f32 = finite_lower_sum(bounds_minus.lower());

                // Restore original alpha.
                alpha_state.alphas[relu_idx][neuron_idx] = orig_alpha;
                alpha_state.alphas_upper[relu_idx][neuron_idx] = orig_alpha_upper;
                grad[neuron_idx] = (lower_plus - lower_minus) / (2.0 * SEQUENTIAL_GRAD_EPS);
            }

            grads.push(grad);
        }
        Ok(grads)
    })();

    // Always restore alpha state, including early error returns from `?` (#3393).
    restore_alpha_snapshot(alpha_state, &original_alphas, &original_alphas_upper);
    result
}

#[allow(clippy::too_many_arguments)]
fn analytic_chain_gradients(
    graph: &GraphNetwork,
    alpha_state: &AlphaState,
    input: &BoundedTensor,
    node_bounds: &HashMap<String, BoundedTensor>,
    exec_order: &[String],
    output_dim: usize,
    relu_name_to_idx: &HashMap<String, usize>,
    engine: Option<&dyn GemmEngine>,
    analytic_gradients: &[Array1<f32>],
    iter: usize,
) -> Result<Vec<Array1<f32>>> {
    // True chain-rule gradients using intermediate A matrices.
    // Run backward pass that stores A matrices at each ReLU node.
    let input_dim = input.len();
    let mut scratch: Vec<Array1<f32>> = analytic_gradients
        .iter()
        .map(|g| Array1::zeros(g.len()))
        .collect();

    let Some(relu_names) = relu_names_in_alpha_order(relu_name_to_idx, scratch.len()) else {
        record_chain_rule_fallback(
            iter,
            "relu_name_to_idx could not be reconstructed into contiguous alpha order",
        );
        return Ok(analytic_gradients.to_vec());
    };
    let Some(graph_alpha_state) = graph_alpha_state_from_sequential(alpha_state, relu_name_to_idx)
    else {
        record_chain_rule_fallback(
            iter,
            "sequential alpha state could not be projected into graph order",
        );
        return Ok(analytic_gradients.to_vec());
    };

    let mut scratch_upper: Vec<Array1<f32>> =
        scratch.iter().map(|g| Array1::zeros(g.len())).collect();

    match graph.dag_alpha_backward_pass_with_intermediates(
        input,
        node_bounds,
        exec_order,
        output_dim,
        input_dim,
        relu_name_to_idx,
        &graph_alpha_state,
        alpha_state.invprop(),
        &mut scratch,
        &mut scratch_upper,
        engine,
        None,
        None, // mul_binary_alphas — not used in AnalyticChain gradient computation
        None, // No deadline for AnalyticChain gradient computation (#3393)
    ) {
        Ok((_bounds, intermediate)) => Ok(graph.compute_graph_chain_rule_gradients_with_binding(
            input,
            &relu_names,
            &intermediate,
            Some(&graph_alpha_state),
            engine,
        )),
        Err(error) => {
            // Fall back to local gradients if intermediate storage failed.
            record_chain_rule_fallback(
                iter,
                &format!("backward pass with intermediates failed: {error}"),
            );
            Ok(analytic_gradients.to_vec())
        }
    }
}

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
mod tests;
