// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Helper functions for backward CROWN dispatch.
//!
//! Extracted from `dispatch.rs` to stay under the 500-line file limit (#3424).

use std::borrow::Cow;
use std::time::Instant;

use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;

use crate::bounds::LinearBounds;
use crate::layers::BoundPropagation;
use crate::NETWORK_INPUT;

use super::types::{BackwardDispatchResult, DispatchContext};

/// Preserve structured error types (ShapeMismatch, UnsupportedOp, etc.) so callers
/// can match on them for IBP fallback. Only wrap unstructured errors in InvalidSpec.
/// #3602: Without this, ShapeMismatch from complex DAG topologies is wrapped in
/// InvalidSpec, preventing the CROWN-IBP catch from falling back to IBP.
/// #conv-crown-oom: CpuMemoryExceeded (the Conv2d backward per-buffer memory-cap
/// backstop) must likewise stay structured — every CROWN-IBP / spec-root catch site
/// (crown_tighten.rs, spec_propagation, warm_start, crown_batched) matches it to
/// degrade that node to sound IBP (a valid over-approximation) instead of OOMing.
/// Wrapping it in InvalidSpec defeated those catches and turned the intended sound
/// IBP fallback into a fatal "CROWN failed at node" error on conv-heavy specs.
pub(super) fn preserve_structured_error(e: NyError, node_name: &str, layer_name: &str) -> NyError {
    match e {
        NyError::ShapeMismatch { .. }
        | NyError::UnsupportedOp(_)
        | NyError::UnsupportedConfiguration(_)
        | NyError::NumericalInstability(_)
        | NyError::SoundnessRefusal(_)
        | NyError::DeadlineExceeded(_)
        | NyError::CpuMemoryExceeded { .. }
        | NyError::InternalError(_) => e,
        _ => NyError::InvalidSpec(format!(
            "CROWN failed at node '{node_name}' ({layer_name}): {e}"
        )),
    }
}

/// Resolve input bounds for a named node, returning either network input
/// or cached node bounds.
pub(super) fn resolve_input_bounds<'a>(
    input_name: &str,
    network_input: &'a BoundedTensor,
    node_bounds: super::types::NodeBoundsView<'a>,
    node_name: &str,
    label: &str,
) -> Result<&'a BoundedTensor> {
    if input_name == NETWORK_INPUT {
        Ok(network_input)
    } else {
        node_bounds.get(input_name).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "{label} '{input_name}' not found at node '{node_name}'"
            ))
        })
    }
}

/// Engine-aware convolution dispatch helper (#3598).
///
/// Threads `GemmEngine` for GPU acceleration instead of going through the
/// `BoundPropagation` trait (which discards engine). The closure receives the
/// cloned layer, input shape, bounds, and engine, and is responsible for
/// shape setup + engine-aware propagation.
pub(super) fn dispatch_conv_engine_aware<'a, C, F>(
    conv: &C,
    ctx: &DispatchContext<'_>,
    node_lb: &'a LinearBounds,
    layer_name: &str,
    min_dims: usize,
    setup_and_propagate: F,
) -> Result<BackwardDispatchResult>
where
    C: Clone,
    F: FnOnce(
        &mut C,
        &[usize],
        &'a LinearBounds,
        Option<&dyn GemmEngine>,
        Option<Instant>,
    ) -> Result<Cow<'a, LinearBounds>>,
{
    let input_shape = ctx.pre_activation.shape();
    if input_shape.len() < min_dims {
        return Err(NyError::UnsupportedOp(format!(
            "{layer_name} CROWN backward requires >= {min_dims}D input shape, got {:?} at node '{}'",
            input_shape, ctx.node_name,
        )));
    }
    let mut conv_with_shape = conv.clone();
    let new_lb = setup_and_propagate(
        &mut conv_with_shape,
        input_shape,
        node_lb,
        ctx.engine,
        ctx.deadline,
    )
    .map_err(|e| preserve_structured_error(e, ctx.node_name, layer_name))?;
    Ok(BackwardDispatchResult::Single(Box::new(
        new_lb.into_owned(),
    )))
}

/// Dispatch a layer through its `propagate_linear` trait method.
pub(super) fn dispatch_propagate_linear(
    layer: &dyn BoundPropagation,
    node_lb: &LinearBounds,
    node_name: &str,
    layer_type: &str,
) -> Result<BackwardDispatchResult> {
    let new_lb = layer
        .propagate_linear(node_lb)
        .map_err(|e| preserve_structured_error(e, node_name, layer_type))?
        .into_owned();
    Ok(BackwardDispatchResult::Single(Box::new(new_lb)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// #conv-crown-oom regression: the Conv2d backward per-buffer memory-cap
    /// backstop raises `CpuMemoryExceeded`, and every downstream catch site
    /// (crown_tighten, spec_propagation, warm_start, crown_batched) matches that
    /// structured variant to degrade the node to sound IBP. If
    /// `preserve_structured_error` wraps it in `InvalidSpec`, those matches miss
    /// and the intended sound fallback becomes a fatal "CROWN failed at node"
    /// error on conv-heavy specs (observed on soundnessbench). Guard it.
    #[test]
    fn preserve_keeps_cpu_memory_exceeded_structured() {
        let e = NyError::CpuMemoryExceeded {
            required_bytes: 603_979_776,
            budget_bytes: 536_870_912,
            site: "conv2d::ops_transpose_gemm::backward_result",
        };
        let out = preserve_structured_error(e, "/model/model.6/Conv", "Conv2d");
        assert!(
            out.is_cpu_memory_exceeded(),
            "CpuMemoryExceeded must remain structured for the IBP-fallback catches; got {out:?}"
        );
        assert!(
            !matches!(out, NyError::InvalidSpec(_)),
            "CpuMemoryExceeded must not be wrapped in InvalidSpec"
        );
    }

    /// Companion control (#3602): ShapeMismatch is likewise preserved.
    #[test]
    fn preserve_keeps_shape_mismatch_structured() {
        let e = NyError::ShapeMismatch {
            expected: vec![1, 2],
            got: vec![3, 4],
        };
        let out = preserve_structured_error(e, "n", "Conv2d");
        assert!(matches!(out, NyError::ShapeMismatch { .. }));
    }
}
