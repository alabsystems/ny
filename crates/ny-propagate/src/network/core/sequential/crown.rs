// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! CROWN propagation entry points for sequential networks.

// fast.rs adds Network::propagate_crown_fast and Network::propagate_crown_with_linear.
mod fast;

// Bounds validation predicates (Packet 1 of crown.rs decomposition, #3868).
mod bounds_validation;
pub(crate) use bounds_validation::has_degraded_bounds;

// CROWN output tightening via forward-bound intersection (Packet 2, #3880).
mod tighten;
pub(crate) use tighten::{tighten_crown_output, tighten_crown_output_with_provenance};

// GPU layer extraction and static cache helpers (Packet 3 of crown.rs decomposition).
mod gpu_extraction;
pub(crate) use gpu_extraction::{
    apply_bn_werr_to_host_relu, extract_relu_gpu_layer_with_alpha, try_extract_batch_norm_conv1x1,
    try_extract_single_gpu_layer, GpuCrownStaticCache,
};

// Dense backward-step dispatch and materialization budget helpers (Packet 4, #4162).
mod backward_step;

// Patches backward step dispatch (Packet 1 of crown.rs three-module extraction, #4005).
mod patches_step;
pub(crate) use patches_step::crown_backward_step_patches;

// Public CROWN entry-point wrappers (Packet B of #4233 crown.rs decomposition).
mod entry_points;

// SDP-CROWN ℓ2 ball propagation (Packet C of #4233 crown.rs decomposition).
mod sdp;

// Batched N-D CROWN propagation (Packet D of #4233 crown.rs decomposition).
mod batched;

use crate::bounds::patches::{CrownBounds, PatchesLinearBounds};
use crate::bounds::LinearBounds;
use crate::contiguous_flat_slice;
use crate::layers::Layer;
use crate::types::CrownIbpFallbackReason;
use ndarray::{ArrayD, IxDyn};
use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::{BoundedTensor, RepairStrategy};
use std::time::Instant;
use tracing::{debug, info, instrument};

use super::Network;
use backward_step::{
    crown_backward_step, dense_identity_budget_estimate, dense_materialization_budget_estimate,
    guard_dense_materialization_budget, log_dense_materialization_budget_fallback,
};
use gpu_extraction::extract_gpu_crown_layers_cached;

/// Result of a single CROWN backward step through one layer.
///
/// Used by [`crown_backward_step`] to communicate the outcome to callers,
/// who handle IBP fallback differently (e.g., `propagate_ibp` vs
/// `ibp_fallback_with_constant_linear`).
#[derive(Debug, Clone)]
pub(crate) struct CrownStepFallback {
    /// Structured reason for the fallback.
    pub reason: CrownIbpFallbackReason,
    /// Human-readable fallback details for logging / diagnostics.
    pub details: String,
}

pub(crate) enum CrownStepResult {
    /// Backward step succeeded; `linear_bounds` was updated in-place.
    Continue,
    /// This layer requires IBP fallback with structured diagnostics.
    IbpFallback(CrownStepFallback),
}

impl Network {
    /// Core CROWN implementation with optional pre-computed IBP bounds (#3397).
    #[instrument(skip(self, input, precomputed_ibp, engine, deadline), fields(num_layers = self.layers.len(), input_shape = ?input.shape()))]
    fn propagate_crown_core(
        &self,
        input: &BoundedTensor,
        precomputed_ibp: Option<Vec<BoundedTensor>>,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
        crown_backward_layers: Option<usize>,
    ) -> Result<BoundedTensor> {
        // Disable the L2/Cauchy–Schwarz lever for the sequential-Network
        // fixed-slope CROWN scope (chokepoint for all sequential CROWN entry
        // points). The CROWN-IBP intermediate collection and the deadline IBP
        // fallback below run inside this scope; their lever-firing IBP forward
        // passes are gated off. Sound; restored on drop. See `crate::l2_lever_gate`.
        let _l2_lever_off = crate::l2_lever_gate::L2LeverGuard::disabled();
        if self.layers.is_empty() {
            return Ok(input.clone());
        }
        if self.has_self_attention() {
            return Err(NyError::UnsupportedConfiguration(
                "SelfAttention requires a graph network; use GraphNetwork IBP or CROWN".to_string(),
            ));
        }

        // Deadline check before expensive CROWN-IBP collection (#3328).
        if let Some(d) = deadline {
            if Instant::now() >= d {
                info!("CROWN: deadline exceeded before CROWN-IBP collection, falling back to IBP");
                return self.propagate_ibp(input);
            }
        }

        // Step 1: Collect CROWN-IBP intermediate bounds.
        // When pre-computed IBP bounds are available, skip internal IBP (~59s savings
        // for soundnessbench). Otherwise, compute IBP internally as before (#3397).
        let layer_bounds = match precomputed_ibp {
            Some(ibp) => {
                self.collect_crown_ibp_bounds_with_precomputed_ibp(input, ibp, engine, deadline)?
            }
            None => {
                self.collect_crown_ibp_bounds_with_engine_and_deadline(input, engine, deadline)?
            }
        };
        if let Some(max_layers) = crown_backward_layers {
            self.propagate_crown_with_layer_bounds_and_engine_and_deadline_and_limits(
                input,
                &layer_bounds,
                engine,
                deadline,
                Some(max_layers),
            )
        } else {
            self.propagate_crown_with_layer_bounds_and_engine_and_deadline(
                input,
                &layer_bounds,
                engine,
                deadline,
            )
        }
    }

    /// Propagate CROWN using caller-provided intermediate bounds.
    ///
    /// This skips the internal CROWN-IBP collection pass when a caller already
    /// has one sound forward bound per layer, such as sequential β-CROWN input
    /// split domains that just recomputed their intermediate bounds.
    pub(crate) fn propagate_crown_with_layer_bounds_and_engine_and_deadline(
        &self,
        input: &BoundedTensor,
        layer_bounds: &[BoundedTensor],
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
    ) -> Result<BoundedTensor> {
        self.propagate_crown_with_layer_bounds_and_engine_and_deadline_and_limits(
            input,
            layer_bounds,
            engine,
            deadline,
            None,
        )
    }

    pub(crate) fn propagate_crown_with_layer_bounds_and_engine_and_deadline_and_limits(
        &self,
        input: &BoundedTensor,
        layer_bounds: &[BoundedTensor],
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
        crown_backward_layers: Option<usize>,
    ) -> Result<BoundedTensor> {
        if self.layers.is_empty() {
            return Ok(input.clone());
        }
        if self.has_self_attention() {
            return Err(NyError::UnsupportedConfiguration(
                "SelfAttention requires a graph network; use GraphNetwork IBP or CROWN".to_string(),
            ));
        }
        if layer_bounds.len() != self.layers.len() {
            return Err(NyError::InvalidSpec(format!(
                "pre-computed layer bounds have {} entries, expected {} (one per layer)",
                layer_bounds.len(),
                self.layers.len()
            )));
        }

        let output_bounds = layer_bounds
            .last()
            .ok_or_else(|| NyError::InvalidSpec("No layer bounds computed".to_string()))?;
        let output_dim = output_bounds.len();
        let output_shape = output_bounds.shape().to_vec();

        debug!(
            "CROWN: Starting backward propagation from {} outputs",
            output_dim
        );

        // Step 2: Initialize linear bounds at output as CrownBounds.
        // When the network has Conv2d and the output is 3D spatial (C, H, W),
        // start in Patches mode so Conv2d backward uses PatchesPropagation
        // (sparse receptive-field representation instead of full dense A-matrix).
        // For non-spatial output (e.g., 1D after Linear), start Dense.
        //
        // Note: Most VNN-COMP CNN classifiers end with Linear → 1D output,
        // so they start Dense here. This is correct — the final spec backward
        // operates on a small matrix (spec_dim × input_dim) that is efficient in
        // Dense mode. alpha-beta-CROWN likewise starts Dense for spec backward.
        //
        // The key Patches optimization occurs in CROWN-IBP intermediate bounds
        // (propagate_crown_partial_with_engine in ibp.rs), where each layer's
        // output IS 3D spatial and Patches mode activates automatically.
        //
        // Reference: designs/2026-02-28-patches-mode-wrapper-enum-design.md
        // output >= 1 * output + 0, output <= 1 * output + 0
        let has_conv2d = self.layers.iter().any(|l| matches!(l, Layer::Conv2d(_)));
        let mut crown_bounds = if has_conv2d && output_shape.len() == 3 {
            let spatial = (output_shape[0], output_shape[1], output_shape[2]);
            debug!(
                "CROWN: Patches mode — 3D spatial output {:?} with Conv2d",
                spatial
            );
            CrownBounds::Patches(Box::new(PatchesLinearBounds::identity(spatial, spatial)))
        } else {
            debug!(
                "CROWN: Dense mode — output_shape {:?}, has_conv2d={}",
                output_shape, has_conv2d
            );
            if let Some(estimate) =
                dense_identity_budget_estimate("initial_dense_identity", output_dim)
            {
                log_dense_materialization_budget_fallback("CROWN", estimate, None, None);
                return self.propagate_ibp(input);
            }
            CrownBounds::Dense(LinearBounds::identity(output_dim))
        };

        // GPU fast-path (#3397): if the engine supports full GPU CROWN backward and all
        // layers are GPU-compatible (Linear + ReLU), run the entire backward + concretize
        // on GPU. Only the final bounds are read back (eliminates N-1 host roundtrips).
        // Soundness gate (#vnncomp-gpu-crown-soundness): when soundness is required
        // (competition / verdict-deciding), `sound_gpu_crown_backward` returns `None`
        // so this whole GPU f32 fast-path is bypassed and the proven-sound CPU
        // backward+concretize loop below decides the bound. See `sound_gpu_gate`.
        if !crown_bounds.is_patches() {
            if let Some((gpu, use_sound)) = crate::sound_gpu_gate::gpu_crown_backward_route(engine)
            {
                if let Some(gpu_layers) = extract_gpu_crown_layers_cached(
                    &self.layers,
                    layer_bounds,
                    input,
                    &self.gpu_crown_cache,
                ) {
                    debug!(
                        "CROWN: GPU fast-path — {} layers all supported, {} specs",
                        gpu_layers.len(),
                        output_dim
                    );
                    let input_lower = contiguous_flat_slice(input.lower());
                    let input_upper = contiguous_flat_slice(input.upper());

                    // Build identity spec matrix: each row i has 1.0 at column i, 0.0 elsewhere.
                    // This propagates bounds for every output neuron independently.
                    let mut spec = vec![0.0f32; output_dim * output_dim];
                    for i in 0..output_dim {
                        spec[i * output_dim + i] = 1.0;
                    }

                    // Under the soundness gate `use_sound` is true: decide the
                    // bound on the SOUND GPU-resident backward (certified enclosure).
                    // Otherwise the existing fast (unsound) path. Either way, an
                    // Err or NaN below falls through to the proven CPU sound loop.
                    let gpu_result = if use_sound {
                        gpu.crown_backward_gpu_sound(
                            &gpu_layers,
                            &spec,
                            output_dim,
                            &input_lower,
                            &input_upper,
                        )
                    } else {
                        gpu.crown_backward_gpu(
                            &gpu_layers,
                            &spec,
                            output_dim,
                            &input_lower,
                            &input_upper,
                        )
                    };
                    match gpu_result {
                        Ok(result) => {
                            // Raw-NaN pre-check (#3757): Widen repair maps NaN to ±inf,
                            // which would erase the signal before tighten_crown_output()
                            // can reject the whole GPU result.
                            let has_nan = result
                                .lower_bounds
                                .iter()
                                .chain(result.upper_bounds.iter())
                                .any(|v| v.is_nan());
                            if has_nan {
                                info!("CROWN: NaN in raw GPU bounds, falling back to CPU");
                            } else {
                                let lower = ArrayD::from_shape_vec(
                                    IxDyn(&output_shape),
                                    result.lower_bounds,
                                )
                                .map_err(|e| {
                                    NyError::InvalidSpec(format!("GPU CROWN reshape: {e}"))
                                })?;
                                let upper = ArrayD::from_shape_vec(
                                    IxDyn(&output_shape),
                                    result.upper_bounds,
                                )
                                .map_err(|e| {
                                    NyError::InvalidSpec(format!("GPU CROWN reshape: {e}"))
                                })?;

                                let crown_output = BoundedTensor::new_repaired(
                                    lower,
                                    upper,
                                    RepairStrategy::Widen,
                                )?;
                                info!(
                                    "CROWN: GPU backward succeeded — {} specs × {} layers",
                                    output_dim,
                                    gpu_layers.len()
                                );
                                return tighten_crown_output(
                                    crown_output,
                                    output_bounds,
                                    "CROWN-GPU",
                                );
                            }
                        }
                        Err(e) => {
                            info!("CROWN: GPU backward failed ({}), falling back to CPU", e);
                            // Fall through to CPU backward loop below
                        }
                    }
                } else {
                    debug!("CROWN: GPU fast-path skipped — unsupported layer types");
                }
            }
        }

        // Step 3: Propagate backward through each layer using Patches dispatch
        // Track peak memory and mode transitions for Patches diagnostics (#2613).
        let mut peak_memory_bytes: usize = crown_bounds.memory_bytes();
        for (backward_steps, (i, layer)) in self.layers.iter().enumerate().rev().enumerate() {
            // Deadline check before each layer backward pass (#3328).
            // Reuse already-computed output_bounds instead of re-running IBP (#3397).
            if let Some(d) = deadline {
                if Instant::now() >= d {
                    info!(
                        "CROWN: deadline exceeded at backward layer {}/{}, reusing CROWN-IBP output bounds",
                        i,
                        self.layers.len()
                    );
                    return Ok(output_bounds.clone());
                }
            }

            if crown_backward_layers.is_some_and(|max_layers| backward_steps >= max_layers) {
                info!(
                    "CROWN: truncating backward after {} layers at frontier {} of {}",
                    backward_steps,
                    i,
                    self.layers.len()
                );
                if let Some(estimate) = dense_materialization_budget_estimate(
                    &crown_bounds,
                    "truncated_concretization",
                )? {
                    log_dense_materialization_budget_fallback("CROWN", estimate, None, None);
                    return Ok(output_bounds.clone());
                }
                let truncation_bounds = &layer_bounds[i];
                let crown_output = crown_bounds
                    .into_dense()?
                    .concretize_sound(truncation_bounds)
                    .reshape(&output_shape)?;
                return tighten_crown_output(crown_output, output_bounds, "CROWN");
            }

            debug!(
                "CROWN: backward through layer {} ({})",
                i,
                layer.layer_type()
            );

            // Get pre-activation bounds (bounds before this layer)
            // For layer i, pre-activation bounds are:
            // - layer_bounds[i-1] for i > 0
            // - input for i == 0
            let pre_activation = if i == 0 { input } else { &layer_bounds[i - 1] };

            match crown_backward_step_patches(
                layer,
                &mut crown_bounds,
                pre_activation,
                engine,
                i,
                "CROWN",
                deadline,
            )? {
                CrownStepResult::Continue => {}
                CrownStepResult::IbpFallback(fallback) => {
                    // Reuse already-computed output_bounds instead of re-running IBP (#3397).
                    info!(
                        "CROWN: {} — reusing CROWN-IBP output bounds",
                        fallback.details
                    );
                    return Ok(output_bounds.clone());
                }
            }

            let step_memory = crown_bounds.memory_bytes();
            peak_memory_bytes = peak_memory_bytes.max(step_memory);
            debug!(
                "CROWN: layer {} memory: {} bytes ({}), mode={}",
                i,
                step_memory,
                if step_memory > 1_048_576 {
                    format!("{:.1} MB", step_memory as f64 / 1_048_576.0)
                } else {
                    format!("{:.1} KB", step_memory as f64 / 1024.0)
                },
                if crown_bounds.is_patches() {
                    "Patches"
                } else {
                    "Dense"
                }
            );
        }
        debug!(
            "CROWN: backward complete — peak A-matrix memory: {} bytes ({:.1} MB)",
            peak_memory_bytes,
            peak_memory_bytes as f64 / 1_048_576.0
        );

        // Step 4: Convert to Dense (no-op when already Dense) and concretize
        debug!("CROWN: Concretizing linear bounds with input");
        if let Some(estimate) =
            dense_materialization_budget_estimate(&crown_bounds, "final_concretization")?
        {
            // Reuse already-computed output_bounds instead of re-running IBP (#3397).
            log_dense_materialization_budget_fallback("CROWN", estimate, None, None);
            return Ok(output_bounds.clone());
        }
        let linear_bounds = crown_bounds.into_dense()?;
        let crown_output = linear_bounds
            .concretize_sound(input)
            .reshape(&output_shape)?;

        // Step 5+6: Degrade check + forward-bound tightening (#3043 dedup).
        tighten_crown_output(crown_output, output_bounds, "CROWN")
    }
}

#[cfg(test)]
mod crown_tests;
