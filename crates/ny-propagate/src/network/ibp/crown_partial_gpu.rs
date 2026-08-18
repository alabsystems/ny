// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! GPU-accelerated CROWN backward for partial sub-networks.
//!
//! Extracted from crown_partial.rs to stay within the 500-line limit.

use super::crown_partial::PartialCrownPropagationResult;
use super::sparse_merge::{find_unstable_dense_indices, scatter_sparse_crown_into_ibp};
use crate::contiguous_flat_slice;
use crate::layers::Layer;
use crate::network::core::try_extract_single_gpu_layer;
use ndarray::{ArrayD, IxDyn};
use ny_core::{GpuCrownBackward, NyError, Result};
use ny_tensor::BoundedTensor;
use std::time::Instant;
use tracing::{debug, info};

/// Build a full or unstable-row identity seed without allowing an optional GPU
/// allocation to abort the process. Any overflow, invalid sparse index, or
/// allocator refusal selects the caller's CPU partial-CROWN path.
fn try_gpu_partial_spec(
    unstable_indices: Option<&[usize]>,
    n_specs: usize,
    output_dim: usize,
) -> Option<Vec<f32>> {
    let elements = n_specs.checked_mul(output_dim)?;
    let mut spec = Vec::new();
    spec.try_reserve_exact(elements).ok()?;
    spec.resize(elements, 0.0);
    if let Some(indices) = unstable_indices {
        if indices.len() != n_specs {
            return None;
        }
        for (row, &column) in indices.iter().enumerate() {
            if column >= output_dim {
                return None;
            }
            let offset = row.checked_mul(output_dim)?.checked_add(column)?;
            spec[offset] = 1.0;
        }
    } else {
        if n_specs != output_dim {
            return None;
        }
        for row in 0..output_dim {
            let offset = row.checked_mul(output_dim)?.checked_add(row)?;
            spec[offset] = 1.0;
        }
    }
    Some(spec)
}

/// Attempt GPU-accelerated CROWN backward for a partial sub-network.
///
/// Extracts [`GpuCrownLayer`] descriptors for every layer in the sub-network
/// (in backward order), builds an identity specification matrix, and dispatches
/// the entire backward pass + concretization to the GPU. This avoids the
/// CPU-bound Patches/Dense backward loop and N host-side round-trips.
///
/// Returns `Ok(Some(result))` when the GPU path succeeds, `Ok(None)` when the
/// sub-network contains unsupported layer types (caller falls through to CPU),
/// or `Err(...)` on a hard GPU error.
///
/// `use_sound`: when `true` (the soundness gate is engaged and the engine
/// advertises a sound GPU-resident backward), dispatch the SOUND resident
/// backward [`crown_backward_gpu_sound`] — which carries the certified
/// `γ_n·S` coefficient-rounding error through the on-device backward GEMM
/// chain — so this verdict-relevant INTERMEDIATE CROWN bound is a sound
/// enclosure decided on GPU instead of on the CPU fallback. When `false`
/// (speed-only callers, no gate), the existing fast unsound `crown_backward_gpu`
/// is used. Either way, an `Err`/NaN below falls back to the proven CPU sound
/// loop (the 0-wrong moat holds).
///
/// Part of #3599 Phase 1
#[allow(clippy::too_many_arguments)]
pub(super) fn try_gpu_crown_partial_backward(
    layers: &[Layer],
    prior_bounds: &[BoundedTensor],
    input: &BoundedTensor,
    gpu: &dyn GpuCrownBackward,
    use_sound: bool,
    output_dim: usize,
    output_shape: &[usize],
    output_bounds: &BoundedTensor,
    deadline: Option<Instant>,
) -> Result<Option<PartialCrownPropagationResult>> {
    if deadline.is_some_and(|limit| Instant::now() >= limit) {
        return Err(NyError::DeadlineExceeded(
            "CROWN-IBP GPU partial: deadline exceeded before setup".to_string(),
        ));
    }
    // Skip GPU dispatch for very small sub-networks — the identity spec
    // construction and GPU upload/download overhead exceeds the CPU backward
    // cost for ≤2 layers.
    if layers.len() < 3 {
        return Ok(None);
    }

    // Phase 2 (#3599): compute unstable indices to determine effective spec count.
    // With >90% stable neurons, only bound unstable ones — reduces GPU GEMM work
    // and allows layers with large output_dim but few unstable neurons to fit.
    let unstable_indices = find_unstable_dense_indices(
        output_bounds.lower().as_slice().unwrap_or(&[]),
        output_bounds.upper().as_slice().unwrap_or(&[]),
        0.9,
    );

    let n_specs = if let Some(ref idx) = unstable_indices {
        if idx.is_empty() {
            // All stable — CROWN cannot improve on IBP, skip GPU entirely.
            return Ok(None);
        }
        idx.len()
    } else {
        output_dim
    };

    // Guard: the GPU spec matrix is n_specs × output_dim × 4 bytes.
    // Phase 2 allows layers with large output_dim through when n_unstable is
    // small (e.g., output_dim=10K but only 2K unstable → 2K specs fit GPU).
    //
    // wgpu dispatch group limit is 65535, so n_specs must fit.
    const GPU_PARTIAL_MAX_SPECS: usize = 8192;
    if n_specs > GPU_PARTIAL_MAX_SPECS {
        debug!(
            "CROWN-IBP GPU partial: {} specs (output_dim={}) exceeds threshold {}, falling back to CPU",
            n_specs, output_dim, GPU_PARTIAL_MAX_SPECS,
        );
        return Ok(None);
    }

    // Step 1: Extract GPU crown layer descriptors in backward (reverse) order.
    // Each layer needs the pre-activation bounds for activation relaxation slopes.
    let mut gpu_layers = Vec::with_capacity(layers.len());
    for (i, layer) in layers.iter().enumerate().rev() {
        if deadline.is_some_and(|limit| Instant::now() >= limit) {
            return Err(NyError::DeadlineExceeded(format!(
                "CROWN-IBP GPU partial: deadline exceeded while extracting layer {i}"
            )));
        }
        let pre_activation = if i == 0 { input } else { &prior_bounds[i - 1] };
        if try_extract_single_gpu_layer(layer, pre_activation, &mut gpu_layers).is_none() {
            debug!(
                "CROWN-IBP GPU partial: unsupported layer {} ({}), falling back to CPU",
                i,
                layer.layer_type(),
            );
            return Ok(None);
        }
    }

    // Step 2: Build specification matrix.
    // Phase 2 (#3599): when unstable indices are available, build a sparse
    // (n_unstable × output_dim) spec instead of the full (output_dim × output_dim)
    // identity. Each row selects one unstable output neuron.
    let spec = if let Some(ref idx) = unstable_indices {
        debug!(
            "CROWN-IBP GPU partial: sparse spec {}/{} unstable ({:.1}% reduction)",
            n_specs,
            output_dim,
            (1.0 - n_specs as f64 / output_dim as f64) * 100.0,
        );
        let Some(spec) = try_gpu_partial_spec(Some(idx), n_specs, output_dim) else {
            info!("CROWN-IBP GPU partial: sparse seed allocation refused, falling back to CPU");
            return Ok(None);
        };
        spec
    } else {
        let Some(spec) = try_gpu_partial_spec(None, n_specs, output_dim) else {
            info!("CROWN-IBP GPU partial: identity seed allocation refused, falling back to CPU");
            return Ok(None);
        };
        spec
    };

    // Step 3: Get contiguous input bounds for concretization.
    let input_lower = contiguous_flat_slice(input.lower());
    let input_upper = contiguous_flat_slice(input.upper());

    // Step 4: Dispatch to GPU with n_specs (potentially reduced).
    // Under the soundness gate (`use_sound`), take the SOUND resident backward,
    // whose bounds are a certified enclosure of the proven CPU sound bound (the
    // `γ_n·S` coefficient-error term is carried through every on-device AW GEMM,
    // validated against the host reference in
    // `crown_backward_sound_resident::tests`). Otherwise the fast unsound path.
    if deadline.is_some_and(|limit| Instant::now() >= limit) {
        return Err(NyError::DeadlineExceeded(
            "CROWN-IBP GPU partial: deadline exceeded before launch".to_string(),
        ));
    }
    let dispatch = if use_sound {
        gpu.crown_backward_gpu_sound(&gpu_layers, &spec, n_specs, &input_lower, &input_upper)
    } else {
        gpu.crown_backward_gpu(&gpu_layers, &spec, n_specs, &input_lower, &input_upper)
    };
    let gpu_result = match dispatch {
        Ok(r) => r,
        Err(e) => {
            if deadline.is_some_and(|limit| Instant::now() >= limit) {
                return Err(NyError::DeadlineExceeded(
                    "CROWN-IBP GPU partial: deadline exceeded during launch".to_string(),
                ));
            }
            info!(
                "CROWN-IBP GPU partial: backward failed ({}), falling back to CPU",
                e
            );
            return Ok(None);
        }
    };
    if deadline.is_some_and(|limit| Instant::now() >= limit) {
        return Err(NyError::DeadlineExceeded(
            "CROWN-IBP GPU partial: deadline exceeded after launch".to_string(),
        ));
    }

    // Step 4b: validate the complete raw device payload before any reshape,
    // repair, scatter, or intersection. Wrong shape, NaN/Inf, and inversion
    // all refuse the GPU result as a unit and preserve the CPU oracle.
    if !crate::sound_gpu_gate::gpu_crown_result_is_publishable(&gpu_result, n_specs) {
        info!("CROWN-IBP GPU partial: malformed raw GPU bounds, falling back to CPU");
        return Ok(None);
    }

    // Step 5: Build BoundedTensor from GPU results.
    // Phase 2 (#3599): sparse mode returns n_unstable bounds — scatter into
    // full output_dim using IBP for stable positions before intersection.
    let crown_result = if let Some(ref idx) = unstable_indices {
        let scattered = scatter_sparse_crown_into_ibp(
            &gpu_result.lower_bounds,
            &gpu_result.upper_bounds,
            output_bounds,
            idx,
            output_shape,
        );
        let Ok(scattered) = scattered else {
            info!("CROWN-IBP GPU partial: sparse result validation refused, falling back to CPU");
            return Ok(None);
        };
        scattered
    } else {
        let (Ok(lower), Ok(upper)) = (
            ArrayD::from_shape_vec(IxDyn(output_shape), gpu_result.lower_bounds),
            ArrayD::from_shape_vec(IxDyn(output_shape), gpu_result.upper_bounds),
        ) else {
            info!("CROWN-IBP GPU partial: result reshape refused, falling back to CPU");
            return Ok(None);
        };
        let Ok(bounds) = BoundedTensor::new(lower, upper) else {
            info!("CROWN-IBP GPU partial: result validation refused, falling back to CPU");
            return Ok(None);
        };
        bounds
    };

    // Step 6: Intersect GPU-computed CROWN bounds with IBP bounds for tightening.
    // Per-element intersection keeps the tighter of CROWN vs IBP per neuron,
    // with disjoint fallback to union (conservative). Same pattern as the
    // CPU CROWN-IBP tightening path.
    let tightened = match output_bounds.intersection_per_element(&crown_result) {
        Some((result, disjoint)) => {
            if disjoint > 0 {
                debug!(
                    "CROWN-IBP GPU partial: {disjoint}/{} disjoint elements (union fallback)",
                    output_dim,
                );
            }
            result
        }
        None => {
            // Shape mismatch — should not happen since both tensors are
            // constructed from the same output_shape. NaN is caught by the
            // Step 4b pre-check above (#3752).
            info!("CROWN-IBP GPU partial: intersection failed (shape mismatch?), using IBP bounds");
            return Ok(None);
        }
    };

    info!(
        "CROWN-IBP GPU partial: {} specs × {} layers succeeded (output_dim={})",
        n_specs,
        gpu_layers.len(),
        output_dim,
    );

    if deadline.is_some_and(|limit| Instant::now() >= limit) {
        return Err(NyError::DeadlineExceeded(
            "CROWN-IBP GPU partial: deadline exceeded before publish".to_string(),
        ));
    }
    Ok(Some(PartialCrownPropagationResult::Crown(Box::new(
        tightened,
    ))))
}
