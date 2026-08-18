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
pub(crate) use tighten::{
    tighten_crown_output, tighten_crown_output_with_deadline,
    tighten_crown_output_with_provenance_and_deadline,
};

// GPU layer extraction and static cache helpers (Packet 3 of crown.rs decomposition).
mod gpu_extraction;
pub(crate) use gpu_extraction::{
    apply_bn_werr_to_host_relu, extract_relu_gpu_layer_with_alpha, gpu_relu_affine_cell,
    try_extract_batch_norm_conv1x1, try_extract_single_gpu_layer, GpuCrownStaticCache,
    GpuReluAffineVariant,
};

// Dense backward-step dispatch and materialization budget helpers (Packet 4, #4162).
mod backward_step;

// Patches backward step dispatch (Packet 1 of crown.rs three-module extraction, #4005).
pub(crate) mod patches_step;
pub(crate) use patches_step::{
    crown_backward_step_patches, crown_backward_step_patches_spec_crown,
    crown_backward_step_patches_with_deadline_authority, SpecPatchesStepError,
};

// Public CROWN entry-point wrappers (Packet B of #4233 crown.rs decomposition).
mod entry_points;

// SDP-CROWN ℓ2 ball propagation (Packet C of #4233 crown.rs decomposition).
mod sdp;

// Batched N-D CROWN propagation (Packet D of #4233 crown.rs decomposition).
mod batched;

use crate::bounds::patches::{CrownBounds, PatchesLinearBounds, PatchesMaterializationPurpose};
use crate::bounds::LinearBounds;
use crate::contiguous_flat_slice;
use crate::layers::Layer;
use crate::types::CrownIbpFallbackReason;
use ndarray::{ArrayD, IxDyn};
use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;
use std::mem::size_of;
use std::time::Instant;
use tracing::{debug, info, instrument};

use super::Network;
use backward_step::{
    crown_backward_step, crown_backward_step_with_dispatch_boundary,
    dense_identity_budget_estimate, dense_materialization_budget_estimate,
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

/// Materialize a terminal Patches carrier while classifying only a structured
/// host-memory refusal as an established forward-bound fallback. Shape,
/// numerical, deadline, and internal errors remain typed.
pub(crate) fn materialize_terminal_crown_bounds(
    bounds: CrownBounds,
) -> Result<Option<LinearBounds>> {
    materialize_terminal_crown_bounds_with_deadline(bounds, None)
}

/// Deadline-aware terminal materialization.  The same absolute authority used
/// by the backward walk covers validation, allocation, scatter, numeric
/// firewalls, and publication.  A deadline refusal is deliberately not mapped
/// to the memory fallback: callers must preserve its distinct terminal policy.
pub(crate) fn materialize_terminal_crown_bounds_with_deadline(
    bounds: CrownBounds,
    deadline: Option<Instant>,
) -> Result<Option<LinearBounds>> {
    match bounds.into_dense_with_deadline_for_purpose(
        deadline,
        PatchesMaterializationPurpose::NetworkInputTerminal,
    ) {
        Ok(bounds) => Ok(Some(bounds)),
        Err(NyError::CpuMemoryExceeded { .. }) => Ok(None),
        Err(error) => Err(error),
    }
}

#[inline]
fn check_gpu_crown_deadline(deadline: Option<Instant>, phase: &str) -> Result<()> {
    if deadline.is_some_and(|value| Instant::now() >= value) {
        Err(NyError::DeadlineExceeded(format!(
            "sequential full-GPU CROWN backward: deadline exceeded {phase}"
        )))
    } else {
        Ok(())
    }
}

fn check_crown_publication_deadline(deadline: Option<Instant>, phase: &'static str) -> Result<()> {
    if deadline.is_some_and(|limit| Instant::now() >= limit) {
        Err(NyError::DeadlineExceeded(format!(
            "CROWN: deadline exceeded {phase}"
        )))
    } else {
        Ok(())
    }
}

/// Clone an already-computed forward fallback without hiding an
/// uninterruptible endpoint-pair copy inside a finite CROWN request. The
/// no-deadline route keeps the historical `Clone` implementation exactly.
fn clone_crown_forward_fallback(
    bounds: &BoundedTensor,
    deadline: Option<Instant>,
) -> Result<BoundedTensor> {
    if deadline.is_some() {
        tighten::clone_forward_bounds_with_deadline(bounds, deadline)
    } else {
        Ok(bounds.clone())
    }
}

/// Allocate the dense identity seed without making allocation failure a
/// process abort. The GPU lane is optional; overflow or allocator refusal must
/// select the already-materialized CPU CROWN path below.
fn try_gpu_identity_spec(output_dim: usize) -> Option<Vec<f32>> {
    let elements = output_dim.checked_mul(output_dim)?;
    let mut spec = Vec::new();
    spec.try_reserve_exact(elements).ok()?;
    spec.resize(elements, 0.0);
    for row in 0..output_dim {
        let diagonal = row.checked_mul(output_dim)?.checked_add(row)?;
        spec[diagonal] = 1.0;
    }
    Some(spec)
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
            return clone_crown_forward_fallback(input, deadline);
        }
        // Deadline check before expensive CROWN-IBP collection (#3328). An
        // owned precomputed IBP vector already contains the certified output,
        // so that result can be moved out in O(1) without doing any post-expiry
        // work. A fresh request has no such publishable artifact and remains a
        // typed deadline refusal.
        let precomputed_ibp = if deadline.is_some_and(|d| Instant::now() >= d) {
            match precomputed_ibp {
                Some(mut bounds) => {
                    if bounds.len() != self.layers.len() {
                        return Err(NyError::InvalidSpec(format!(
                            "pre-computed IBP bounds have {} entries, expected {} (one per layer)",
                            bounds.len(),
                            self.layers.len()
                        )));
                    }
                    return bounds.pop().ok_or_else(|| {
                        NyError::InvalidSpec("No pre-computed layer bounds supplied".to_string())
                    });
                }
                None => {
                    return Err(NyError::DeadlineExceeded(
                        "CROWN: deadline exceeded before CROWN-IBP collection".into(),
                    ));
                }
            }
        } else {
            precomputed_ibp
        };
        if self.has_self_attention() {
            return Err(NyError::UnsupportedConfiguration(
                "SelfAttention requires a graph network; use GraphNetwork IBP or CROWN".to_string(),
            ));
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
            return clone_crown_forward_fallback(input, deadline);
        }
        if deadline.is_some_and(|limit| Instant::now() >= limit) {
            // `layer_bounds` is borrowed. Publishing its output would require
            // an O(output) clone, which cannot begin after authority expires.
            return Err(NyError::DeadlineExceeded(
                "CROWN: deadline exceeded before borrowed layer-bound propagation".into(),
            ));
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
            let seed = match PatchesLinearBounds::try_identity_with_deadline(
                spatial, spatial, deadline, 0,
            ) {
                Ok(seed) => seed,
                Err(error) if error.is_deadline_exceeded() || error.is_cpu_memory_exceeded() => {
                    debug!(
                        "CROWN: Patches identity admission refused ({error}); reusing CROWN-IBP output bounds"
                    );
                    return clone_crown_forward_fallback(output_bounds, deadline);
                }
                Err(error) => return Err(error),
            };
            CrownBounds::Patches(Box::new(seed))
        } else {
            debug!(
                "CROWN: Dense mode — output_shape {:?}, has_conv2d={}",
                output_shape, has_conv2d
            );
            if let Some(estimate) =
                dense_identity_budget_estimate("initial_dense_identity", output_dim)
            {
                log_dense_materialization_budget_fallback("CROWN", estimate, None, None);
                return clone_crown_forward_fallback(output_bounds, deadline);
            }
            CrownBounds::Dense(LinearBounds::try_identity_with_deadline(
                output_dim,
                deadline,
                output_bounds
                    .len()
                    .saturating_mul(2)
                    .saturating_mul(size_of::<f32>()),
            )?)
        };

        // GPU fast-path (#3397): if the engine supports full GPU CROWN backward and all
        // layers are GPU-compatible (Linear + ReLU), run the entire backward + concretize
        // on GPU. Only the final bounds are read back (eliminates N-1 host roundtrips).
        // Soundness gate (#vnncomp-gpu-crown-soundness): when soundness is required
        // (competition / verdict-deciding), `sound_gpu_crown_backward` returns `None`
        // so this whole GPU f32 fast-path is bypassed and the proven-sound CPU
        // backward+concretize loop below decides the bound. See `sound_gpu_gate`.
        if !crown_bounds.is_patches() {
            'gpu_fast_path: {
                // The backend deadline contract begins at device dispatch, but
                // this optional lane first extracts every layer, may copy both
                // input endpoint arrays, and builds an O(output_dim^2) host
                // identity. Under FINITE authority the route below therefore
                // consults only ALREADY-MATERIALIZED backends (the caller's
                // engine, or a process-global slot prewarmed at qualification
                // time — `Some(deadline)` never invokes a lazy factory, see
                // `select_lazy_backend_for_deadline`), and admission
                // additionally requires a SOUND backend that honors
                // cooperative cancellation (#charged-metal-engagement): the
                // unpollable host stretches are bracketed by the explicit
                // checkpoints below and the device walk polls the leased
                // deadline between work units. A backend that leaves the
                // cooperative-deadline capability at its default (e.g. the
                // CUDA engine, whose claim is deliberately narrow) is refused
                // under a deadline exactly as the earlier blanket skip did —
                // fail-closed to the deadline-aware CPU path, byte-identical
                // bounds for those hosts. The no-deadline GPU route remains
                // unchanged.
                let Some((gpu, use_sound)) =
                    crate::sound_gpu_gate::gpu_crown_backward_route_with_deadline(engine, deadline)
                else {
                    break 'gpu_fast_path;
                };
                if deadline.is_some() && !use_sound {
                    debug!(
                        "CROWN: GPU fast-path skipped — finite authority admits only a sound backend"
                    );
                    break 'gpu_fast_path;
                }
                // A finite verifier deadline may enter this multi-dispatch fast
                // path only when the exact routed backend advertises cooperative
                // cancellation. Otherwise skip it and use the deadline-aware CPU
                // backward below. Do not route a second time when installing the
                // lease: a process-global sound backend may differ from `engine`.
                if !crate::sound_gpu_gate::gpu_crown_backend_honors_deadline(gpu, deadline) {
                    debug!(
                        "CROWN: GPU fast-path skipped — routed backend does not honor deadlines"
                    );
                    break 'gpu_fast_path;
                }
                if check_gpu_crown_deadline(deadline, "before layer extraction").is_err() {
                    debug!("CROWN: GPU fast-path deadline refusal; falling back to CPU");
                    break 'gpu_fast_path;
                }
                let Some(gpu_layers) = extract_gpu_crown_layers_cached(
                    &self.layers,
                    layer_bounds,
                    input,
                    &self.gpu_crown_cache,
                ) else {
                    debug!("CROWN: GPU fast-path skipped — unsupported layer types");
                    break 'gpu_fast_path;
                };
                debug!(
                    "CROWN: GPU fast-path — {} layers all supported, {} specs",
                    gpu_layers.len(),
                    output_dim
                );
                let input_lower = contiguous_flat_slice(input.lower());
                let input_upper = contiguous_flat_slice(input.upper());

                // Build identity spec matrix: each row i has 1.0 at column i.
                // Overflow/OOM is an optional-lane refusal, not a verifier error.
                let Some(spec) = try_gpu_identity_spec(output_dim) else {
                    info!("CROWN: GPU identity allocation refused, falling back to CPU");
                    break 'gpu_fast_path;
                };

                if check_gpu_crown_deadline(deadline, "before backend launch").is_err() {
                    debug!("CROWN: GPU launch deadline refusal; falling back to CPU");
                    break 'gpu_fast_path;
                }
                let _gpu_deadline_scope =
                    crate::sound_gpu_gate::GpuCrownBackendDeadlineScope::set(gpu, deadline);

                // Under the soundness gate `use_sound` is true. Every backend
                // error and every malformed payload is a whole-result refusal;
                // the proven CPU loop below remains the sole fallback authority.
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
                if check_gpu_crown_deadline(deadline, "after backend launch").is_err() {
                    debug!("CROWN: GPU completion missed deadline; falling back to CPU");
                    break 'gpu_fast_path;
                }
                let result = match gpu_result {
                    Ok(result) => result,
                    Err(error) => {
                        info!(
                            "CROWN: GPU backward failed ({}), falling back to CPU",
                            error
                        );
                        break 'gpu_fast_path;
                    }
                };
                if !crate::sound_gpu_gate::gpu_crown_result_is_publishable(&result, output_dim) {
                    info!("CROWN: malformed GPU bounds, falling back to CPU");
                    break 'gpu_fast_path;
                }

                let (Ok(lower), Ok(upper)) = (
                    ArrayD::from_shape_vec(IxDyn(&output_shape), result.lower_bounds),
                    ArrayD::from_shape_vec(IxDyn(&output_shape), result.upper_bounds),
                ) else {
                    info!("CROWN: GPU result reshape refused, falling back to CPU");
                    break 'gpu_fast_path;
                };
                let Ok(crown_output) = BoundedTensor::new(lower, upper) else {
                    info!("CROWN: GPU result validation refused, falling back to CPU");
                    break 'gpu_fast_path;
                };
                info!(
                    "CROWN: GPU backward succeeded — {} specs × {} layers",
                    output_dim,
                    gpu_layers.len()
                );
                return match tighten_crown_output_with_deadline(
                    crown_output,
                    output_bounds,
                    "CROWN-GPU",
                    deadline,
                ) {
                    Ok(bounds) => Ok(bounds),
                    Err(NyError::CpuMemoryExceeded { .. } | NyError::DeadlineExceeded(_)) => {
                        clone_crown_forward_fallback(output_bounds, deadline)
                    }
                    Err(error) => Err(error),
                };
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
                    return clone_crown_forward_fallback(output_bounds, deadline);
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
                    return clone_crown_forward_fallback(output_bounds, deadline);
                }
                let truncation_bounds = &layer_bounds[i];
                let dense =
                    match materialize_terminal_crown_bounds_with_deadline(crown_bounds, deadline) {
                        Ok(Some(bounds)) => bounds,
                        Ok(None) | Err(NyError::DeadlineExceeded(_)) => {
                            return clone_crown_forward_fallback(output_bounds, deadline);
                        }
                        Err(error) => return Err(error),
                    };
                let crown_output =
                    match dense.concretize_sound_with_deadline(truncation_bounds, deadline) {
                        Ok(bounds) => bounds,
                        Err(NyError::CpuMemoryExceeded { .. } | NyError::DeadlineExceeded(_)) => {
                            return clone_crown_forward_fallback(output_bounds, deadline);
                        }
                        Err(error) => return Err(error),
                    };
                let crown_output = match crown_output.into_reshape_with_poll(&output_shape, || {
                    check_crown_publication_deadline(deadline, "during truncated reshape")
                }) {
                    Ok(bounds) => bounds,
                    Err(NyError::CpuMemoryExceeded { .. } | NyError::DeadlineExceeded(_)) => {
                        return clone_crown_forward_fallback(output_bounds, deadline);
                    }
                    Err(error) => return Err(error),
                };
                return match tighten_crown_output_with_deadline(
                    crown_output,
                    output_bounds,
                    "CROWN",
                    deadline,
                ) {
                    Ok(bounds) => Ok(bounds),
                    Err(NyError::CpuMemoryExceeded { .. } | NyError::DeadlineExceeded(_)) => {
                        clone_crown_forward_fallback(output_bounds, deadline)
                    }
                    Err(error) => Err(error),
                };
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
                    return clone_crown_forward_fallback(output_bounds, deadline);
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
            return clone_crown_forward_fallback(output_bounds, deadline);
        }
        let linear_bounds =
            match materialize_terminal_crown_bounds_with_deadline(crown_bounds, deadline) {
                Ok(Some(bounds)) => bounds,
                Ok(None) | Err(NyError::DeadlineExceeded(_)) => {
                    return clone_crown_forward_fallback(output_bounds, deadline);
                }
                Err(error) => return Err(error),
            };
        let crown_output = match linear_bounds.concretize_sound_with_deadline(input, deadline) {
            Ok(bounds) => bounds,
            Err(NyError::CpuMemoryExceeded { .. } | NyError::DeadlineExceeded(_)) => {
                return clone_crown_forward_fallback(output_bounds, deadline);
            }
            Err(error) => return Err(error),
        };
        let crown_output = match crown_output.into_reshape_with_poll(&output_shape, || {
            check_crown_publication_deadline(deadline, "during final reshape")
        }) {
            Ok(bounds) => bounds,
            Err(NyError::CpuMemoryExceeded { .. } | NyError::DeadlineExceeded(_)) => {
                return clone_crown_forward_fallback(output_bounds, deadline);
            }
            Err(error) => return Err(error),
        };

        // Step 5+6: Degrade check + forward-bound tightening (#3043 dedup).
        match tighten_crown_output_with_deadline(crown_output, output_bounds, "CROWN", deadline) {
            Ok(bounds) => Ok(bounds),
            Err(NyError::CpuMemoryExceeded { .. } | NyError::DeadlineExceeded(_)) => {
                clone_crown_forward_fallback(output_bounds, deadline)
            }
            Err(error) => Err(error),
        }
    }
}

#[cfg(test)]
mod crown_tests;
