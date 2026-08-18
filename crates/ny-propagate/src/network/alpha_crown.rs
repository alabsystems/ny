// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Alpha-CROWN methods for sequential networks.
//!
//! This module contains alpha-CROWN propagation methods for `Network`.
//! Alpha-CROWN optimizes the linear relaxation parameters (alpha) to
//! achieve tighter bounds than standard CROWN.

mod backward;
mod gradients;
mod helpers;

use self::backward::{
    backward_pass_core, run_simple_backward_pass, BackwardPassConfig, BackwardPassResult,
};
use self::gradients::{compute_finite_difference_gradients, compute_spsa_gradients};
use self::helpers::{
    build_layer_to_relu_idx, compute_chain_rule_gradients, init_invprop_if_enabled,
};
use crate::bounds::{
    AlphaCrownConfig, AlphaCrownIntermediate, AlphaState, GradientMethod, LinearBounds,
};
use crate::layers::Layer;
use crate::network::alpha_crown_loop::{
    alpha_crown_optimize, AlphaCrownBackend, BackwardIterationResult,
};
use ndarray::{Array1, Array2};
use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;
use tracing::{debug, instrument, warn};

use super::core::Network;

/// Cheap pre-collection admission for the native output-seed INVPROP route.
///
/// The exact output dimension is only available after intermediate-bound
/// collection. This rejects every statically unsupported constraint form so a
/// pure-linear network pays for collection only when it can reach the checked
/// post-collection route.
fn native_invprop_route_candidate(config: &AlphaCrownConfig) -> bool {
    config.iterations > 0
        && config.invprop.enabled
        && config.invprop.optimize_gammas
        && config
            .output_constraints
            .as_ref()
            .is_some_and(|constraints| {
                constraints.is_conjunction
                    && constraints.clause_indices.is_none()
                    && constraints.num_constraints() > 0
                    && constraints.output_dim() > 0
                    && constraints.rhs.len() == constraints.num_constraints()
                    && constraints
                        .a_matrix
                        .iter()
                        .chain(constraints.rhs.iter())
                        .all(|value| value.is_finite())
            })
}

/// Extension trait for alpha-CROWN propagation on sequential networks.
pub(crate) trait NetworkAlphaCrownExt {
    /// Alpha-CROWN entry point implementation.
    fn propagate_alpha_crown_impl(&self, input: &BoundedTensor) -> Result<BoundedTensor>;

    /// Alpha-CROWN with optional GEMM acceleration engine.
    fn propagate_alpha_crown_with_engine_impl(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor>;

    /// Alpha-CROWN with custom configuration (no acceleration engine).
    fn propagate_alpha_crown_with_config_impl(
        &self,
        input: &BoundedTensor,
        config: &AlphaCrownConfig,
    ) -> Result<BoundedTensor>;

    /// Alpha-CROWN with custom configuration and optional GEMM acceleration engine.
    fn propagate_alpha_crown_with_config_and_engine_impl(
        &self,
        input: &BoundedTensor,
        config: &AlphaCrownConfig,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor>;

    /// Single forward+backward pass with given alpha state (for numerical gradient).
    fn propagate_alpha_crown_single_pass_impl(
        &self,
        input: &BoundedTensor,
        layer_bounds: &[BoundedTensor],
        alpha_state: &AlphaState,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor>;

    /// Single forward+backward pass with intermediates for chain-rule gradients.
    fn propagate_alpha_crown_with_intermediates_impl(
        &self,
        input: &BoundedTensor,
        layer_bounds: &[BoundedTensor],
        alpha_state: &AlphaState,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<AlphaCrownIntermediate>;

    /// Compute chain-rule gradients for alpha parameters.
    fn compute_chain_rule_gradients_impl(
        &self,
        alpha_state: &AlphaState,
        intermediate: &AlphaCrownIntermediate,
    ) -> Vec<Array1<f32>>;
}

impl NetworkAlphaCrownExt for Network {
    #[inline]
    fn propagate_alpha_crown_impl(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        self.propagate_alpha_crown_with_engine_impl(input, None)
    }

    #[inline]
    #[instrument(skip(self, input, engine), fields(num_layers = self.layers.len(), input_shape = ?input.shape()))]
    fn propagate_alpha_crown_with_engine_impl(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor> {
        self.propagate_alpha_crown_with_config_and_engine_impl(
            input,
            &AlphaCrownConfig::default(),
            engine,
        )
    }

    #[instrument(skip(self, input, config), fields(num_layers = self.layers.len(), iterations = config.iterations))]
    fn propagate_alpha_crown_with_config_impl(
        &self,
        input: &BoundedTensor,
        config: &AlphaCrownConfig,
    ) -> Result<BoundedTensor> {
        self.propagate_alpha_crown_with_config_and_engine_impl(input, config, None)
    }

    #[instrument(skip(self, input, config, engine), fields(num_layers = self.layers.len(), iterations = config.iterations))]
    fn propagate_alpha_crown_with_config_and_engine_impl(
        &self,
        input: &BoundedTensor,
        config: &AlphaCrownConfig,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor> {
        // Disable the L2/Cauchy–Schwarz lever for the sequential-Network
        // alpha-CROWN scope (chokepoint for all sequential alpha-CROWN variants).
        // The CROWN-IBP intermediate bound collection and per-iteration SPSA
        // single-pass evaluations run inside this scope on the driver thread, so
        // their lever-firing IBP forward passes are gated off. Sound; restored on
        // drop. See `crate::l2_lever_gate`.
        let _l2_lever_off = crate::l2_lever_gate::L2LeverGuard::disabled();
        if self.layers.is_empty() {
            return Ok(input.clone());
        }
        if self.has_self_attention() {
            return Err(NyError::UnsupportedConfiguration(
                "SelfAttention requires a graph network; alpha-CROWN only supports sequential networks"
                    .to_string(),
            ));
        }

        // Early exits BEFORE expensive collect_crown_ibp_bounds (#3218):
        // Check for unsupported layer types and deep networks first.

        // Check for Conv2d/ConvTranspose2d/MaxPool2d layers - fall back to CROWN for now
        for layer in &self.layers {
            if matches!(
                layer,
                Layer::Conv2d(_) | Layer::ConvTranspose2d(_) | Layer::MaxPool2d(_)
            ) {
                debug!(
                    "Alpha-CROWN: Conv2d/ConvTranspose2d/MaxPool2d detected, falling back to CROWN"
                );
                return self.propagate_crown_with_engine_and_deadline(
                    input,
                    engine,
                    config.deadline,
                );
            }
        }

        // Count ReLU layers cheaply before the expensive CROWN-IBP collection.
        let relu_count = self
            .layers
            .iter()
            .filter(|l| matches!(l, Layer::ReLU(_)))
            .count();

        let invprop_route_candidate = native_invprop_route_candidate(config);
        if relu_count == 0 && !invprop_route_candidate {
            return self.propagate_crown_with_engine_and_deadline(input, engine, config.deadline);
        }

        // Adaptive skip: for deep models, alpha-CROWN optimization provides
        // diminishing returns due to bound explosion through many ReLU layers.
        // Skip optimization and return CROWN bounds directly. Matches the
        // GraphNetwork DAG path check (propagate_dag.rs, bounds/alpha.rs).
        // Must be checked BEFORE collect_crown_ibp_bounds to avoid wasting
        // the O(L²) intermediate bound computation. #3218
        if config.adaptive_skip && relu_count > config.adaptive_skip_depth_threshold {
            debug!(
                "Sequential α-CROWN: adaptive skip — {} ReLU layers > threshold {}, \
                 falling back to CROWN",
                relu_count, config.adaptive_skip_depth_threshold
            );
            return self.propagate_crown_with_engine_and_deadline(input, engine, config.deadline);
        }

        // Deadline guard (#4321 / VNN-COMP no-JSON fix): the intermediate-bound
        // collection below is the O(L²) CROWN-IBP pass that dominates the
        // initial-bound cost on deep/wide networks (vggnet16, yolo, soundnessbench).
        // If the wall-clock deadline is already spent, do not even begin it —
        // return a fast deadline-aware CROWN bound (which itself falls back to IBP
        // once the deadline is hit) so the caller gets a sound bound promptly
        // rather than stalling here past the competition budget.
        if config.past_deadline() {
            debug!("Sequential α-CROWN: deadline already exceeded, returning CROWN fallback");
            return self.propagate_crown_with_engine_and_deadline(input, engine, config.deadline);
        }

        // Step 1: Run CROWN-IBP to collect tighter bounds at each layer
        // This matches the intermediate bound computation used in propagate_crown().
        // Thread the deadline (#4321): on deep/wide nets this pass can run for many
        // seconds; without a deadline it would blow past the competition --timeout
        // and get OS-killed before any JSON verdict is emitted. The deadline-aware
        // variant falls remaining layers back to (sound, looser) IBP bounds.
        let layer_bounds =
            self.collect_crown_ibp_bounds_with_engine_and_deadline(input, engine, config.deadline)?;
        let output_dim = layer_bounds.last().map(BoundedTensor::len).ok_or_else(|| {
            NyError::InvalidSpec(
                "Alpha-CROWN expected at least one intermediate bound for non-empty network"
                    .to_string(),
            )
        })?;

        // Step 2: Identify ReLU layers and initialize alpha state
        let (relu_layer_indices, layer_to_relu_idx) = build_layer_to_relu_idx(&self.layers);

        // Initialize alpha state
        // We need pre-activation bounds for ReLU layers
        // For ReLU at layer i, pre-activation is layer_bounds[i-1] (or input if i==0)
        let pre_activation_bounds: Vec<BoundedTensor> = relu_layer_indices
            .iter()
            .map(|&i| {
                if i == 0 {
                    input.clone()
                } else {
                    layer_bounds[i - 1].clone()
                }
            })
            .collect();

        let mut alpha_state = AlphaState::from_preactivation_bounds(
            &pre_activation_bounds,
            &(0..relu_layer_indices.len()).collect::<Vec<_>>(),
        )?;

        // Initialize INVPROP state if enabled and constraints provided (#371)
        let invprop_enabled = init_invprop_if_enabled(
            config,
            &mut alpha_state,
            &relu_layer_indices,
            &pre_activation_bounds,
            input.len(),
        )?;

        let num_unstable = alpha_state.num_unstable();
        let invprop_seed_treatment_eligible = config.invprop.optimize_gammas
            && crate::network::alpha_crown_loop::native_invprop_seed_treatment_eligible(
                &alpha_state,
                output_dim,
            );
        if num_unstable == 0 && !invprop_seed_treatment_eligible {
            // No unstable neurons, alpha-CROWN won't help
            debug!("Alpha-CROWN: No unstable neurons, using CROWN");
            return self.propagate_crown_with_engine_and_deadline(input, engine, config.deadline);
        }

        debug!(
            "Alpha-CROWN: Starting optimization with {} unstable neurons across {} ReLU layers{}",
            num_unstable,
            relu_layer_indices.len(),
            if invprop_enabled {
                " (INVPROP enabled)"
            } else {
                ""
            }
        );

        // Step 3: Optimization loop via shared backend (#2835)
        let mut backend = SequentialAlphaCrownBackend {
            network: self,
            layer_bounds: &layer_bounds,
            engine,
            config,
            output_dim,
            relu_layer_indices,
            layer_to_relu_idx,
        };

        alpha_crown_optimize(
            &mut backend,
            config,
            &mut alpha_state,
            input,
            invprop_enabled,
        )
    }

    fn propagate_alpha_crown_single_pass_impl(
        &self,
        input: &BoundedTensor,
        layer_bounds: &[BoundedTensor],
        alpha_state: &AlphaState,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<BoundedTensor> {
        self.propagate_alpha_crown_single_pass_with_deadline(
            input,
            layer_bounds,
            alpha_state,
            engine,
            None,
        )
    }

    fn propagate_alpha_crown_with_intermediates_impl(
        &self,
        input: &BoundedTensor,
        layer_bounds: &[BoundedTensor],
        alpha_state: &AlphaState,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<AlphaCrownIntermediate> {
        self.propagate_alpha_crown_with_intermediates_and_deadline(
            input,
            layer_bounds,
            alpha_state,
            engine,
            None,
        )
    }

    fn compute_chain_rule_gradients_impl(
        &self,
        alpha_state: &AlphaState,
        intermediate: &AlphaCrownIntermediate,
    ) -> Vec<Array1<f32>> {
        compute_chain_rule_gradients(alpha_state, intermediate)
    }
}

impl Network {
    fn propagate_alpha_crown_single_pass_with_deadline(
        &self,
        input: &BoundedTensor,
        layer_bounds: &[BoundedTensor],
        alpha_state: &AlphaState,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<std::time::Instant>,
    ) -> Result<BoundedTensor> {
        let bp_output = match run_simple_backward_pass(
            &self.layers,
            input,
            layer_bounds,
            alpha_state,
            engine,
            deadline,
            false,
            "single pass",
        )? {
            BackwardPassResult::Success(output) => *output,
            BackwardPassResult::Fallback => {
                return self.propagate_crown_with_engine_and_deadline(input, engine, deadline);
            }
        };
        Ok(bp_output.linear_bounds.concretize_sound(input))
    }

    fn propagate_alpha_crown_with_intermediates_and_deadline(
        &self,
        input: &BoundedTensor,
        layer_bounds: &[BoundedTensor],
        alpha_state: &AlphaState,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<std::time::Instant>,
    ) -> Result<AlphaCrownIntermediate> {
        let bp_output = match run_simple_backward_pass(
            &self.layers,
            input,
            layer_bounds,
            alpha_state,
            engine,
            deadline,
            true,
            "intermediates pass",
        )? {
            BackwardPassResult::Success(output) => *output,
            BackwardPassResult::Fallback => {
                // Unsupported layer: return constant LinearBounds from CROWN fallback.
                let crown_bounds =
                    self.propagate_crown_with_engine_and_deadline(input, engine, deadline)?;
                let crown_flat = crown_bounds.flatten();
                let num_outputs = crown_flat.len();
                // `len()` == `flatten().len()` (flatten preserves element count) with no allocation.
                let num_inputs = input.len();
                return Ok(AlphaCrownIntermediate {
                    a_at_relu: Vec::new(),
                    pre_relu_bounds: Vec::new(),
                    final_bounds: LinearBounds::new_or_conservative(
                        Array2::zeros((num_outputs, num_inputs)),
                        Array1::from_vec(crown_flat.lower().iter().copied().collect()),
                        Array2::zeros((num_outputs, num_inputs)),
                        Array1::from_vec(crown_flat.upper().iter().copied().collect()),
                    )?,
                });
            }
        };

        // Reverse to get forward layer order (backward_pass_core collects in reverse)
        let mut a_at_relu = bp_output.a_at_relu;
        let mut pre_relu_bounds = bp_output.pre_relu_bounds;
        a_at_relu.reverse();
        pre_relu_bounds.reverse();

        Ok(AlphaCrownIntermediate {
            a_at_relu,
            pre_relu_bounds,
            final_bounds: bp_output.linear_bounds,
        })
    }
}

/// Backend for the shared alpha-CROWN optimization loop on sequential `Network`.
///
/// Holds references to the network-specific state needed during the loop:
/// layers, intermediate bounds, engine, and layout mappings.
pub(crate) struct SequentialAlphaCrownBackend<'a> {
    network: &'a Network,
    layer_bounds: &'a [BoundedTensor],
    engine: Option<&'a dyn GemmEngine>,
    config: &'a AlphaCrownConfig,
    output_dim: usize,
    relu_layer_indices: Vec<usize>,
    layer_to_relu_idx: std::collections::HashMap<usize, usize>,
}

impl AlphaCrownBackend for SequentialAlphaCrownBackend<'_> {
    fn backward_iteration(
        &self,
        alpha_state: &AlphaState,
        input: &BoundedTensor,
        _iter: usize,
        invprop_enabled: bool,
        need_grad: bool,
    ) -> Result<Option<BackwardIterationResult>> {
        let bp_config = BackwardPassConfig {
            track_gradients: need_grad,
            store_intermediates: false,
            best_of_oc: self.config.invprop.best_of_oc_and_no_oc && invprop_enabled,
            engine: self.engine,
            deadline: self.config.deadline,
            layer_to_relu_idx: &self.layer_to_relu_idx,
            relu_layer_indices: &self.relu_layer_indices,
        };
        let bp_output = match backward_pass_core(
            &self.network.layers,
            input,
            self.layer_bounds,
            alpha_state,
            self.output_dim,
            &bp_config,
        )? {
            BackwardPassResult::Success(output) => *output,
            BackwardPassResult::Fallback => return Ok(None),
        };

        Ok(Some(BackwardIterationResult {
            linear_bounds: bp_output.linear_bounds,
            gradients: bp_output.gradients,
            gradients_upper: bp_output.gradients_upper,
            bounds_without_oc: bp_output.bounds_without_oc,
        }))
    }

    fn compute_gradients(
        &self,
        config: &AlphaCrownConfig,
        alpha_state: &mut AlphaState,
        input: &BoundedTensor,
        gradients: &[Array1<f32>],
        gradients_upper: &[Array1<f32>],
        _iter: usize,
    ) -> Result<(Vec<Array1<f32>>, Vec<Array1<f32>>)> {
        let eps = 1e-3;
        match config.gradient_method {
            // SPSA/FiniteDiff: joint perturbation of both alpha paths — same gradient for both (#3393).
            GradientMethod::Spsa => {
                let grads =
                    compute_spsa_gradients(alpha_state, eps, config.spsa_samples, |state| {
                        if config.deadline.is_some() {
                            self.network
                                .propagate_alpha_crown_single_pass_with_deadline(
                                    input,
                                    self.layer_bounds,
                                    state,
                                    self.engine,
                                    config.deadline,
                                )
                        } else {
                            self.network.propagate_alpha_crown_single_pass_impl(
                                input,
                                self.layer_bounds,
                                state,
                                self.engine,
                            )
                        }
                    })?;
                let upper = grads.clone();
                Ok((grads, upper))
            }
            GradientMethod::FiniteDifferences => {
                let grads = compute_finite_difference_gradients(alpha_state, eps, |state| {
                    if config.deadline.is_some() {
                        self.network
                            .propagate_alpha_crown_single_pass_with_deadline(
                                input,
                                self.layer_bounds,
                                state,
                                self.engine,
                                config.deadline,
                            )
                    } else {
                        self.network.propagate_alpha_crown_single_pass_impl(
                            input,
                            self.layer_bounds,
                            state,
                            self.engine,
                        )
                    }
                })?;
                let upper = grads.clone();
                Ok((grads, upper))
            }
            // Analytic: separate lower/upper gradients from backward pass (#3393).
            GradientMethod::Analytic => Ok((gradients.to_vec(), gradients_upper.to_vec())),
            // AnalyticChain: chain-rule only computes lower gradients; use same for upper.
            // Follow-up: extend chain-rule to produce separate upper gradients.
            GradientMethod::AnalyticChain => {
                let intermediate = if config.deadline.is_some() {
                    self.network
                        .propagate_alpha_crown_with_intermediates_and_deadline(
                            input,
                            self.layer_bounds,
                            alpha_state,
                            self.engine,
                            config.deadline,
                        )?
                } else {
                    self.network.propagate_alpha_crown_with_intermediates_impl(
                        input,
                        self.layer_bounds,
                        alpha_state,
                        self.engine,
                    )?
                };

                if intermediate.a_at_relu.is_empty() {
                    warn!("AnalyticChain: unsupported layer in backward pass, falling back to local gradients");
                    Ok((gradients.to_vec(), gradients_upper.to_vec()))
                } else {
                    let grads = self
                        .network
                        .compute_chain_rule_gradients_impl(alpha_state, &intermediate);
                    let upper = grads.clone();
                    Ok((grads, upper))
                }
            }
        }
    }

    fn crown_fallback(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        self.network.propagate_crown_with_engine_and_deadline(
            input,
            self.engine,
            self.config.deadline,
        )
    }

    fn log_label(&self) -> &str {
        "α-CROWN"
    }
}

#[cfg(test)]
mod tests;
