// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Concat backward dispatch helper.
//!
//! Extracted from `dispatch.rs` to keep that module under 500 lines (#3287).

use ny_core::{NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32};

use crate::bounds::LinearBounds;
use crate::NETWORK_INPUT;

use super::helpers::preserve_structured_error;
use super::types::{BackwardDispatchResult, DispatchContext};

/// Concat dispatch: N-ary split with separate bias channel (#2617).
///
/// Concat is a linear layer and introduces no local relaxation bias. Incoming
/// bias is carried directly in the separate channel, while per-input bounds are
/// propagated with zero bias (`lower_b = upper_b = 0`). This eliminates the old
/// constant-input bias redistribution path (#2529).
pub(super) fn dispatch_concat(
    concat: &crate::layers::ConcatLayer,
    ctx: &DispatchContext<'_>,
    node_lb: &LinearBounds,
) -> Result<BackwardDispatchResult> {
    // Build input_shapes for ALL original inputs (constants + dynamic).
    // constant_inputs is indexed by original ONNX order; ctx.inputs only has
    // graph edges for non-constant inputs.
    let mut input_shapes = Vec::new();
    let mut is_constant = Vec::new(); // Track which positions are constants
    if let Some(ref ci) = concat.constant_inputs {
        let mut graph_idx = 0;
        for (i, const_opt) in ci.iter().enumerate() {
            if let Some(constant_tensor) = const_opt {
                input_shapes.push(constant_tensor.shape().to_vec());
                is_constant.push(true);
            } else {
                let inp_name = ctx.inputs.get(graph_idx).ok_or_else(|| {
                    NyError::InternalError(format!(
                        "Concat '{}': ran out of graph inputs at graph_idx {}",
                        ctx.node_name, graph_idx
                    ))
                })?;
                graph_idx += 1;
                let shape = if inp_name == NETWORK_INPUT {
                    ctx.network_input.shape().to_vec()
                } else if let Some(shape) = concat.input_shape(i) {
                    shape.to_vec()
                } else if let Some(bounds) = ctx.node_bounds.get(inp_name.as_str()) {
                    bounds.shape().to_vec()
                } else {
                    return Err(NyError::InvalidSpec(format!(
                        "CROWN failed at node '{}' (Concat): missing shape for input '{}' (index {})",
                        ctx.node_name, inp_name, i
                    )));
                };
                input_shapes.push(shape);
                is_constant.push(false);
            }
        }
    } else {
        // No embedded constants — all inputs from graph
        for (i, inp_name) in ctx.inputs.iter().enumerate() {
            let shape = if inp_name == NETWORK_INPUT {
                ctx.network_input.shape().to_vec()
            } else if let Some(shape) = concat.input_shape(i) {
                shape.to_vec()
            } else if let Some(bounds) = ctx.node_bounds.get(inp_name.as_str()) {
                bounds.shape().to_vec()
            } else {
                return Err(NyError::InvalidSpec(format!(
                    "CROWN failed at node '{}' (Concat): missing shape for input '{}' (index {})",
                    ctx.node_name, inp_name, i
                )));
            };
            input_shapes.push(shape);
            is_constant.push(false);
        }
    }

    // #2617/#2530: Concat introduces no local relaxation bias. Carry incoming
    // bias directly and propagate only zero-bias A-paths.
    let bias_lower = node_lb.lower_b().clone();
    let bias_upper = node_lb.upper_b().clone();
    let mut zero_bias_lb = node_lb.clone();
    zero_bias_lb.lower_b_mut().fill(0.0);
    zero_bias_lb.upper_b_mut().fill(0.0);

    let mut bounds_vec = concat
        .propagate_linear_nary(&zero_bias_lb, &input_shapes)
        .map_err(|e| preserve_structured_error(e, ctx.node_name, "Concat"))?;

    // Defense in depth: enforce zero per-input bias even if a future
    // Concat implementation regresses.
    for lb in &mut bounds_vec {
        lb.lower_b_mut().fill(0.0);
        lb.upper_b_mut().fill(0.0);
    }

    // For constant inputs, compute W_i * constant_value and fold the result
    // into the bias before dropping the weight sub-matrix.  Without this,
    // constant-only Concat nodes lose their values during CROWN backward
    // propagation (#4112).
    let mut bias_lower = bias_lower;
    let mut bias_upper = bias_upper;
    let bounds: Vec<Option<LinearBounds>> = bounds_vec
        .into_iter()
        .enumerate()
        .map(|(i, lb)| {
            if *is_constant.get(i).unwrap_or(&false) {
                if let Some(ref ci) = concat.constant_inputs {
                    if let Some(Some(ct)) = ci.get(i) {
                        let c_flat: ndarray::Array1<f32> =
                            ndarray::Array1::from_iter(ct.lower().iter().copied());
                        // lower_a * c → lower bias contribution (round down)
                        let lower_contrib = lb.lower_a().dot(&c_flat);
                        // upper_a * c → upper bias contribution (round up)
                        let upper_contrib = lb.upper_a().dot(&c_flat);
                        bias_lower += &lower_contrib.mapv(next_down_f32);
                        bias_upper += &upper_contrib.mapv(next_up_f32);
                    }
                }
                None
            } else {
                Some(lb)
            }
        })
        .collect();
    Ok(BackwardDispatchResult::Nary {
        bounds,
        bias_lower,
        bias_upper,
    })
}
