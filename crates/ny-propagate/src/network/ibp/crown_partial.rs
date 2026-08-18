// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Partial CROWN backward dispatch for CROWN-IBP tightening.
//!
//! Contains the CPU Patches/Dense backward loop. GPU fast-path is in
//! `crown_partial_gpu.rs`.

use super::crown_partial_gpu::try_gpu_crown_partial_backward;
use super::helpers::{memory_budget_partial_fallback, patches_dense_materialization_fallback};
use super::sparse_merge::{
    find_unstable_dense_indices, merge_sparse_crown_with_ibp, scatter_sparse_crown_into_ibp,
};
use crate::bounds::patches::{CrownBounds, PatchesLinearBounds};
use crate::bounds::LinearBounds;
use crate::contiguous_flat_slice;
use crate::layers::{BoundPropagation, Layer};
use crate::network::core::{
    crown_backward_step_patches, materialize_terminal_crown_bounds_with_deadline,
    CrownStepFallback, CrownStepResult,
};
use crate::network::crown_memory::{cpu_crown_dense_budget_bytes, DenseMaterializationEstimate};
use ndarray::{Array1, Array2};
use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;
use std::time::Instant;
use tracing::debug;

pub(super) enum PartialCrownPropagationResult {
    Crown(Box<BoundedTensor>),
    ForwardFallback(CrownStepFallback),
}

/// Publish a completed CPU CROWN result only while its node budget is live.
///
/// The backward loop polls the deadline between layers, but concretization and
/// sparse post-processing can themselves cross the cutoff. Treat that late
/// result exactly like any other per-node timeout so the collection caller
/// keeps the sound forward bound instead.
fn publish_concretized_crown(
    result: BoundedTensor,
    deadline: Option<Instant>,
    completed_at: Instant,
) -> Result<BoundedTensor> {
    if deadline.is_some_and(|limit| completed_at >= limit) {
        return Err(NyError::DeadlineExceeded(
            "CROWN-IBP partial: per-node deadline exceeded after concretization".to_string(),
        ));
    }
    Ok(result)
}

fn concretization_memory_fallback(error: &NyError) -> Option<CrownStepFallback> {
    matches!(error, NyError::CpuMemoryExceeded { .. }).then(|| CrownStepFallback {
        reason: crate::types::CrownIbpFallbackReason::MemoryBudgetExceeded,
        details: format!("CROWN-IBP partial concretization exceeded its CPU budget: {error}"),
    })
}

fn check_partial_deadline(deadline: Option<Instant>) -> Result<()> {
    if deadline.is_some_and(|limit| Instant::now() >= limit) {
        Err(NyError::DeadlineExceeded(
            "CROWN-IBP partial: deadline exceeded during final reshape".into(),
        ))
    } else {
        Ok(())
    }
}

/// Propagate CROWN bounds through a partial network (subset of layers).
///
/// Uses Patches-aware backward dispatch via [`crown_backward_step_patches`]
/// to avoid materializing dense A-matrices for Conv2d layers. When the output
/// is 3D spatial and the sub-network contains Conv2d, bounds are initialized
/// in Patches mode (O(out_c * kH * kW) per spatial position) instead of
/// Dense mode (O(out_dim²) total).
///
/// Part of #2613: Patches Mode Phase 1
pub(super) fn propagate_crown_partial_with_engine(
    _all_layers: &[Layer],
    input: &BoundedTensor,
    layers: &[Layer],
    prior_bounds: &[BoundedTensor],
    engine: Option<&dyn GemmEngine>,
    deadline: Option<Instant>,
) -> Result<PartialCrownPropagationResult> {
    if deadline.is_some_and(|limit| Instant::now() >= limit) {
        return Err(NyError::DeadlineExceeded(
            "CROWN-IBP partial: deadline exceeded before entry".to_string(),
        ));
    }
    if layers.is_empty() {
        return Ok(PartialCrownPropagationResult::Crown(Box::new(
            input.clone(),
        )));
    }

    // Get output dimension from last layer via IBP forward pass.
    let last_layer = layers.last().ok_or_else(|| {
        NyError::InvalidSpec("CROWN partial propagation requires at least one layer".to_string())
    })?;
    let pre_last = if layers.len() == 1 {
        input.clone()
    } else if prior_bounds.len() >= layers.len() - 1 {
        prior_bounds[layers.len() - 2].clone()
    } else {
        // Compute IBP for intermediate if not available
        let mut current = input.clone();
        for layer in layers.iter().take(layers.len() - 1) {
            current = layer.propagate_ibp(&current)?;
        }
        current
    };
    let output_bounds = last_layer.propagate_ibp(&pre_last)?;
    let output_dim = output_bounds.len();
    let output_shape = output_bounds.shape().to_vec();

    // GPU fast path (#3599 Phase 1): dispatch the entire per-node backward to GPU
    // when all layers in the sub-network are GPU-extractable. This avoids the
    // CPU-bound Patches/Dense backward loop and host-side concretization.
    // A finite request deliberately skips this optional GPU route. Although a
    // backend can advertise cooperative device cancellation, the legacy host
    // preparation still extracts/copies every layer and input endpoint and
    // builds the full (or sparse) specification matrix without cooperative
    // polls. The CPU Patches/Dense path below observes the same absolute
    // deadline through its admitted materialization boundaries.
    // Soundness gate (#vnncomp-gpu-crown-soundness): under the gate, route to the
    // SOUND GPU-resident backward (`use_sound = true`) when the engine advertises
    // it — that path carries the certified `γ_n·S` coefficient-rounding error
    // through every on-device AW GEMM, so this verdict-relevant INTERMEDIATE CROWN
    // bound is a sound enclosure decided on GPU instead of bailing to the CPU sound
    // loop. Without the gate (`use_sound = false`) the existing fast unsound path
    // runs. Either way an `Err`/NaN inside falls back to the proven CPU loop below.
    // See `sound_gpu_gate`.
    let gpu_route = deadline
        .is_none()
        .then(|| crate::sound_gpu_gate::gpu_crown_backward_route_with_deadline(engine, deadline));
    if let Some(Some((gpu, use_sound))) = gpu_route {
        if crate::sound_gpu_gate::gpu_crown_backend_honors_deadline(gpu, deadline) {
            if deadline.is_some_and(|limit| Instant::now() >= limit) {
                return Err(NyError::DeadlineExceeded(
                    "CROWN-IBP partial: deadline exceeded before GPU dispatch".to_string(),
                ));
            }
            let _gpu_deadline_scope =
                crate::sound_gpu_gate::GpuCrownBackendDeadlineScope::set(gpu, deadline);
            let gpu_result = try_gpu_crown_partial_backward(
                layers,
                prior_bounds,
                input,
                gpu,
                use_sound,
                output_dim,
                &output_shape,
                &output_bounds,
                deadline,
            )?;
            if deadline.is_some_and(|limit| Instant::now() >= limit) {
                return Err(NyError::DeadlineExceeded(
                    "CROWN-IBP partial: deadline exceeded after GPU dispatch".to_string(),
                ));
            }
            if let Some(gpu_result) = gpu_result {
                return Ok(gpu_result);
            }
        }
    }
    if deadline.is_some_and(|limit| Instant::now() >= limit) {
        return Err(NyError::DeadlineExceeded(
            "CROWN-IBP partial: deadline exceeded before CPU fallback".to_string(),
        ));
    }

    // Initialize CROWN bounds — use Patches mode when the output is 3D spatial
    // (channels, height, width) and the sub-network contains Conv2d. This avoids
    // materializing dense A-matrices of size O(out_dim²) in favor of sparse
    // patch coefficients of size O(out_c * kH * kW) per spatial position.
    //
    // Sparse patches optimization (#2613 Phase 4): when IBP bounds show >90%
    // of output neurons are stable (all positive or all negative), only track
    // unstable neurons in the backward pass. This reduces memory from
    // O(total * receptive_field) to O(unstable * receptive_field).
    let has_conv2d = layers.iter().any(|l| matches!(l, Layer::Conv2d(_)));
    let is_sparse_mode;
    // Phase 2 (#3599): track unstable indices for Dense sparse mode.
    // When set, only unstable neurons have spec rows in the backward pass.
    let mut dense_unstable_indices: Option<Vec<usize>> = None;
    // Flat output positions the SPARSE PATCHES seed actually tracks. The merge
    // below must know these positionally: an untracked neuron's concretized
    // bound is `[bias, bias]`, which is indistinguishable by value from a
    // tracked neuron that legitimately lands there. See
    // `merge_sparse_crown_with_ibp`.
    let mut patches_tracked_flat: Option<Vec<usize>> = None;
    let mut crown_bounds = if has_conv2d && output_shape.len() == 3 {
        let spatial = (output_shape[0], output_shape[1], output_shape[2]);
        // Try sparse patches from IBP output bounds
        // Sparse discovery and seed construction retain legacy infallible
        // `Vec` growth and full endpoint scans. Keep finite authority on the
        // admitted full virtual-identity route; no-deadline behavior is exact.
        let sparse_patches = if deadline.is_none() {
            crate::bounds::patches::UnstableIdx::from_ibp_bounds(
                output_bounds.lower().as_slice().unwrap_or(&[]),
                output_bounds.upper().as_slice().unwrap_or(&[]),
                spatial,
                0.9,
            )
        } else {
            None
        };
        if let Some(unstable_idx) = sparse_patches {
            let n = unstable_idx.len();
            let total = spatial.0 * spatial.1 * spatial.2;
            debug!(
                "CROWN-IBP partial: Sparse Patches mode — {}/{} unstable ({:.1}% sparse), {:?}",
                n,
                total,
                (1.0 - n as f64 / total as f64) * 100.0,
                spatial,
            );
            is_sparse_mode = true;
            // Capture the tracked positions BEFORE the seed consumes the index
            // set; the merge after concretization needs them.
            patches_tracked_flat = Some(
                (0..unstable_idx.len())
                    .map(|i| unstable_idx.flat_index(i, spatial.1, spatial.2))
                    .collect(),
            );
            CrownBounds::Patches(Box::new(PatchesLinearBounds::sparse_identity(
                spatial,
                spatial,
                unstable_idx,
            )))
        } else {
            debug!(
                "CROWN-IBP partial: Patches mode — 3D spatial output {:?} with Conv2d",
                spatial
            );
            is_sparse_mode = false;
            let seed = match PatchesLinearBounds::try_identity_with_deadline(
                spatial, spatial, deadline, 0,
            ) {
                Ok(seed) => seed,
                Err(error @ NyError::CpuMemoryExceeded { .. }) => {
                    return Ok(PartialCrownPropagationResult::ForwardFallback(
                        CrownStepFallback {
                            reason: crate::types::CrownIbpFallbackReason::MemoryBudgetExceeded,
                            details: format!(
                                "CROWN-IBP partial identity seed exceeded its CPU budget: {error}"
                            ),
                        },
                    ));
                }
                Err(error) => return Err(error),
            };
            CrownBounds::Patches(Box::new(seed))
        }
    } else {
        is_sparse_mode = false;

        // Phase 2 (#3599): try unstable-only Dense specs. If >90% of output
        // neurons are stable, build a reduced (n_unstable × output_dim) identity
        // instead of the full (output_dim × output_dim). Reduces GEMM work
        // proportionally to the stability ratio.
        //
        // Reference: alpha-beta-CROWN backward_bound.py uses sparse C matrices
        // sized by unstable neuron count for the same optimization.
        let unstable_idx = find_unstable_dense_indices(
            output_bounds.lower().as_slice().unwrap_or(&[]),
            output_bounds.upper().as_slice().unwrap_or(&[]),
            0.9,
        );

        let seed_rows = if let Some(ref idx) = unstable_idx {
            debug!(
                "CROWN-IBP partial: Dense sparse mode — {}/{} unstable ({:.1}% sparse)",
                idx.len(),
                output_dim,
                (1.0 - idx.len() as f64 / output_dim as f64) * 100.0,
            );
            idx.len()
        } else {
            output_dim
        };

        // Memory estimate uses actual seed rows (potentially reduced).
        let required_bytes = seed_rows
            .checked_mul(output_dim)
            .and_then(|n: usize| n.checked_mul(2 * size_of::<f32>()))
            .unwrap_or(usize::MAX);
        let estimate = DenseMaterializationEstimate {
            site: "initial_dense_identity",
            rows: seed_rows,
            cols: output_dim,
            required_bytes,
        };
        if estimate.exceeds_budget(cpu_crown_dense_budget_bytes()) {
            return Ok(PartialCrownPropagationResult::ForwardFallback(
                memory_budget_partial_fallback(estimate, "CROWN-IBP partial"),
            ));
        }

        let seed = if let Some(ref idx) = unstable_idx {
            // Sparse identity: each row selects one unstable output neuron.
            let n = idx.len();
            let mut lower_a = Array2::zeros((n, output_dim));
            let mut upper_a = Array2::zeros((n, output_dim));
            for (row, &col) in idx.iter().enumerate() {
                lower_a[(row, col)] = 1.0;
                upper_a[(row, col)] = 1.0;
            }
            LinearBounds {
                lower_a,
                lower_b: Array1::zeros(n),
                upper_a,
                upper_b: Array1::zeros(n),
                // Exact sparse-identity selection: no accumulated f32 error.
                lower_a_err: None,
                upper_a_err: None,
            }
        } else {
            LinearBounds::identity(output_dim)
        };
        dense_unstable_indices = unstable_idx;
        CrownBounds::Dense(seed)
    };

    // Propagate backward through each layer using Patches-aware dispatch.
    let backward_start = Instant::now();
    for (i, layer) in layers.iter().enumerate().rev() {
        if deadline.is_some_and(|d| Instant::now() >= d) {
            return Err(NyError::DeadlineExceeded(format!(
                "CROWN-IBP partial: per-node deadline exceeded at layer {} ({})",
                i,
                layer.layer_type()
            )));
        }

        // Get pre-activation bounds for this layer.
        let pre_activation = if i == 0 {
            input
        } else if i - 1 < prior_bounds.len() {
            &prior_bounds[i - 1]
        } else {
            return Err(NyError::InvalidSpec(
                "Missing prior bounds for CROWN-IBP".to_string(),
            ));
        };

        match crown_backward_step_patches(
            layer,
            &mut crown_bounds,
            pre_activation,
            engine,
            i,
            "CROWN-IBP",
            deadline,
        )? {
            CrownStepResult::Continue => {}
            CrownStepResult::IbpFallback(fallback) => {
                return Ok(PartialCrownPropagationResult::ForwardFallback(fallback));
            }
        }
    }
    let backward_secs = backward_start.elapsed().as_secs_f64();

    // Convert to Dense for concretization (Patches → Dense is a no-op if already Dense).
    let dense_start = Instant::now();
    if let Some(fallback) = patches_dense_materialization_fallback(
        &crown_bounds,
        "final_concretization",
        "CROWN-IBP partial",
    )? {
        return Ok(PartialCrownPropagationResult::ForwardFallback(fallback));
    }
    let Some(linear_bounds) =
        materialize_terminal_crown_bounds_with_deadline(crown_bounds, deadline)?
    else {
        return Ok(PartialCrownPropagationResult::ForwardFallback(
            CrownStepFallback {
                reason: crate::types::CrownIbpFallbackReason::MemoryBudgetExceeded,
                details:
                    "CROWN-IBP partial final Patches materialization exceeded its full peak budget"
                        .to_string(),
            },
        ));
    };
    let dense_secs = dense_start.elapsed().as_secs_f64();

    // Concretize linear bounds with input bounds and reshape to the IBP output shape.
    //
    // LinearBounds::concretize returns a flat 1D tensor. For ONNX models (and any model
    // with non-1D activations), this causes a shape mismatch against IBP bounds and
    // prevents CROWN-IBP from intersecting/tightening intermediate bounds.
    let concretize_start = Instant::now();

    // Phase 2 (#3599): Dense sparse mode — concretize gives n_unstable values,
    // not output_dim. Scatter CROWN bounds to unstable positions; use IBP for
    // stable neurons where CROWN tightening has no benefit (exact ReLU relaxation).
    if let Some(ref idx) = dense_unstable_indices {
        let sparse_crown = match linear_bounds.concretize_sound_with_deadline(input, deadline) {
            Ok(bounds) => bounds,
            Err(error) => {
                if let Some(fallback) = concretization_memory_fallback(&error) {
                    return Ok(PartialCrownPropagationResult::ForwardFallback(fallback));
                }
                return Err(error);
            }
        };
        let crown_lower = contiguous_flat_slice(sparse_crown.lower());
        let crown_upper = contiguous_flat_slice(sparse_crown.upper());
        let concretize_secs = concretize_start.elapsed().as_secs_f64();
        let partial_total = backward_secs + dense_secs + concretize_secs;
        if partial_total > 0.5 {
            debug!(
                "CROWN-IBP partial ({} layers, dense_sparse): backward={backward_secs:.3}s dense_conv={dense_secs:.3}s concretize={concretize_secs:.3}s total={partial_total:.3}s",
                layers.len(),
            );
        }
        let crown_result = scatter_sparse_crown_into_ibp(
            &crown_lower,
            &crown_upper,
            &output_bounds,
            idx,
            &output_shape,
        )?;
        let crown_result = publish_concretized_crown(crown_result, deadline, Instant::now())?;
        return Ok(PartialCrownPropagationResult::Crown(Box::new(crown_result)));
    }

    let crown_result = match linear_bounds.concretize_sound_with_deadline(input, deadline) {
        Ok(bounds) => bounds,
        Err(error) => {
            if let Some(fallback) = concretization_memory_fallback(&error) {
                return Ok(PartialCrownPropagationResult::ForwardFallback(fallback));
            }
            return Err(error);
        }
    };
    let crown_result =
        crown_result.into_reshape_with_poll(&output_shape, || check_partial_deadline(deadline))?;
    let concretize_secs = concretize_start.elapsed().as_secs_f64();

    // Per-partial-pass timing (#3599): log when significant.
    let partial_total = backward_secs + dense_secs + concretize_secs;
    if partial_total > 0.5 {
        let mode = if is_sparse_mode {
            "sparse"
        } else if has_conv2d && output_shape.len() == 3 {
            "patches"
        } else {
            "dense"
        };
        debug!(
            "CROWN-IBP partial ({} layers, {mode}): backward={backward_secs:.3}s dense_conv={dense_secs:.3}s concretize={concretize_secs:.3}s total={partial_total:.3}s",
            layers.len(),
        );
    }

    // Sparse mode merge (#2613 Phase 4): CROWN only computed bounds for unstable
    // neurons (sparse rows). Stable neurons got zero rows, producing [0, 0] bounds
    // after concretization. Replace those with IBP bounds, which are already tight
    // for stable neurons (all positive or all negative).
    let crown_result = if is_sparse_mode {
        // Fail closed if the tracked-index set is somehow absent: without it the
        // merge cannot tell tracked from untracked, and guessing by value is the
        // defect this threading exists to remove.
        let tracked = patches_tracked_flat.as_deref().ok_or_else(|| {
            NyError::InternalError(
                "sparse CROWN merge: sparse mode without a tracked-index set".into(),
            )
        })?;
        merge_sparse_crown_with_ibp(&crown_result, &output_bounds, tracked)?
    } else {
        crown_result
    };
    let crown_result = publish_concretized_crown(crown_result, deadline, Instant::now())?;
    Ok(PartialCrownPropagationResult::Crown(Box::new(crown_result)))
}

#[cfg(test)]
mod deadline_tests {
    use super::*;
    use ndarray::arr1;
    use std::time::Duration;

    #[test]
    fn valid_result_completed_after_deadline_is_rejected() {
        let valid = BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn()).unwrap();
        let deadline = Instant::now();
        let completed_at = deadline + Duration::from_nanos(1);

        let error = match publish_concretized_crown(valid, Some(deadline), completed_at) {
            Ok(_) => panic!("a valid but late CROWN result must not be published"),
            Err(error) => error,
        };

        assert!(
            matches!(error, NyError::DeadlineExceeded(ref message)
                if message.contains("after concretization")),
            "late CROWN result should fail open as DeadlineExceeded, got {error:?}"
        );
    }
}
