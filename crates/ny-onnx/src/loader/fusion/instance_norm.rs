// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! InstanceNorm/LayerNorm discrimination for proto-level fusion.
//!
//! Part of #3591: decomposed AdaIN patterns (e.g., kokoro vocoder) produce the
//! same ReduceMean→Sub→Pow/Mul→ReduceMean→Add(eps)→Sqrt→Div/Reciprocal+Mul
//! sequence as LayerNorm. The only discriminator is ny shape vs input shape:
//!
//! - **InstanceNorm**: input `[B, C, T]`, ny shape `[C]` (channel dim, axis 1)
//! - **LayerNorm**:    input `[B, T, D]`, ny shape `[D]` (reduced/last dim)

use crate::{LayerSpec, WeightStore};
use ny_core::LayerType;
use std::collections::HashMap;
use tracing::debug;

/// If `spec` (produced by `try_fuse_layer_norm`) is actually a decomposed
/// InstanceNorm pattern, remap its `layer_type` from `LayerNorm` → `InstanceNorm`.
///
/// Returns `true` if the spec was remapped.
///
/// Discrimination logic:
/// - Requires 3D+ input with shape info available
/// - Ny length must match channel dim (axis 1), not the reduced dim (last axis)
/// - If channel_dim == last_dim (ambiguous), leave as LayerNorm (conservative)
pub(crate) fn try_discriminate_instance_norm(
    spec: &mut LayerSpec,
    tensor_shapes: &HashMap<String, Vec<i64>>,
    weights: &WeightStore,
) -> bool {
    if spec.inputs.len() < 2 {
        return false;
    }

    let x_name = &spec.inputs[0];
    let ny_name = &spec.inputs[1];

    let (Some(x_shape), Some(ny)) = (tensor_shapes.get(x_name), weights.get(ny_name)) else {
        return false;
    };

    if x_shape.len() < 3 {
        return false;
    }

    let channel_dim = x_shape[1] as usize;
    let last_dim = *x_shape.last().unwrap_or(&0) as usize;

    if ny.len() == channel_dim && channel_dim != last_dim {
        debug!(
            "Remapping LayerNorm -> InstanceNorm: ny.len()={} \
             matches channel_dim={} (last_dim={})",
            ny.len(),
            channel_dim,
            last_dim
        );
        spec.layer_type = LayerType::InstanceNorm;
        true
    } else {
        false
    }
}
