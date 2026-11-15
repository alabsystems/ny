// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Concat backward dispatch helper.
//!
//! Extracted from `dispatch.rs` to keep that module under 500 lines.

use std::time::Instant;

use ny_core::{
    dd::{next_down_f64, next_up_f64},
    f64_to_f32_down, f64_to_f32_up, require_f64_interval_proof_environment, NyError, Result,
};
use ny_tensor::BoundedTensor;

use crate::bounds::{certified_affine_sum_f32_with_poll, LinearBounds, OutwardDirection};
use crate::NETWORK_INPUT;

use super::helpers::preserve_structured_error;
use super::types::{BackwardDispatchResult, DispatchContext, NaryInputBounds};

/// Concat dispatch: N-ary split with separate bias channel.
///
/// Concat is a linear layer and introduces no local relaxation bias. Incoming
/// bias is carried directly in the separate channel, while per-input bounds are
/// propagated with zero bias (`lower_b = upper_b = 0`). This eliminates the old
/// constant-input bias redistribution path.
pub(super) fn dispatch_concat(
    concat: &crate::layers::ConcatLayer,
    ctx: &DispatchContext<'_>,
    node_lb: &LinearBounds,
) -> Result<BackwardDispatchResult> {
    concat_checkpoint(ctx.deadline)?;
    node_lb.validate_internal_shapes()?;
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

    // Concat introduces no local relaxation bias. Carry incoming
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
    // propagation.
    let mut bias_lower = bias_lower;
    let mut bias_upper = bias_upper;
    let mut bounds = Vec::with_capacity(bounds_vec.len());
    for (i, lb) in bounds_vec.into_iter().enumerate() {
        concat_checkpoint(ctx.deadline)?;
        if *is_constant.get(i).unwrap_or(&false) {
            let ct = concat
                .constant_inputs
                .as_ref()
                .and_then(|ci| ci.get(i))
                .and_then(Option::as_ref)
                .ok_or_else(|| NyError::InternalError("Concat constant slot changed".into()))?;
            fold_constant(&lb, ct, &mut bias_lower, &mut bias_upper, ctx.deadline)?;
            bounds.push(None);
        } else {
            bounds.push(Some(lb));
        }
    }
    concat_checkpoint(ctx.deadline)?;
    Ok(BackwardDispatchResult::Nary {
        // ONNX-ordered: `bounds[i]` is ONNX input i, `None` at every embedded
        // constant whose `W_i · c_i` was just folded into the bias above. The
        // newtype is what stops a consumer pairing this against the
        // dynamic-only `node.inputs` positionally (#nary-zip).
        bounds: NaryInputBounds::from_onnx_order(bounds),
        bias_lower,
        bias_upper,
    })
}

/// Discharge each frozen constant's complete independent A/E planes over its
/// declared enclosure. Products of binary32 operands are exact in binary64;
/// the shared certified reduction encloses cancellation and accumulation.
/// Incoming bias participates in that same reduction, rather than undergoing
/// an uncharged binary32 `+=` after a rounded dot product.
fn fold_constant(
    bounds: &LinearBounds,
    constant: &BoundedTensor,
    lower_bias: &mut ndarray::Array1<f32>,
    upper_bias: &mut ndarray::Array1<f32>,
    deadline: Option<Instant>,
) -> Result<()> {
    concat_checkpoint(deadline)?;
    require_f64_interval_proof_environment()?;
    bounds.validate_internal_shapes()?;
    if bounds.num_inputs() != constant.lower().len()
        || constant.lower().shape() != constant.upper().shape()
        || bounds.num_outputs() != lower_bias.len()
        || bounds.num_outputs() != upper_bias.len()
    {
        return Err(NyError::InvalidSpec(
            "Concat constant coefficient/enclosure shape mismatch".into(),
        ));
    }
    for row in 0..bounds.num_outputs() {
        concat_checkpoint(deadline)?;
        for (coefficients, errors, bias, direction) in [
            (
                bounds.lower_a(),
                bounds.lower_a_err(),
                &mut *lower_bias,
                OutwardDirection::Lower,
            ),
            (
                bounds.upper_a(),
                bounds.upper_a_err(),
                &mut *upper_bias,
                OutwardDirection::Upper,
            ),
        ] {
            let coefficients = coefficients.row(row);
            let terms = coefficients
                .iter()
                .zip(constant.lower().iter().zip(constant.upper().iter()))
                .map(|(&coefficient, (&lo, &hi))| {
                    let bits = coefficient.to_bits();
                    let nonnegative = bits >> 31 == 0 || bits & 0x7fff_ffff == 0;
                    let use_lo = match direction {
                        OutwardDirection::Lower => nonnegative,
                        OutwardDirection::Upper => !nonnegative,
                    };
                    (coefficient, if use_lo { lo } else { hi })
                });
            let nominal = certified_affine_sum_f32_with_poll(bias[row], terms, direction, |_| {
                concat_checkpoint(deadline)
            })?;
            let penalty = if let Some(errors) = errors {
                let errors = errors.row(row);
                let terms = errors
                    .iter()
                    .zip(constant.lower().iter().zip(constant.upper().iter()))
                    .map(|(&error, (&lo, &hi))| {
                        let bits = error.to_bits();
                        let magnitude = bits & 0x7fff_ffff;
                        let error = if magnitude >= f32::INFINITY.to_bits()
                            || (bits >> 31 != 0 && magnitude != 0)
                        {
                            f32::INFINITY
                        } else {
                            error
                        };
                        let magnitude = f32::from_bits(
                            (lo.to_bits() & 0x7fff_ffff).max(hi.to_bits() & 0x7fff_ffff),
                        );
                        (error, magnitude)
                    });
                certified_affine_sum_f32_with_poll(0.0, terms, OutwardDirection::Upper, |_| {
                    concat_checkpoint(deadline)
                })?
            } else {
                0.0
            };
            // A poisoned error lane cannot turn into a negative penalty or
            // an inf-inf NaN that would subsequently be treated as a value.
            bias[row] = if !penalty.is_finite() || penalty < 0.0 {
                match direction {
                    OutwardDirection::Lower => f32::NEG_INFINITY,
                    OutwardDirection::Upper => f32::INFINITY,
                }
            } else {
                match direction {
                    OutwardDirection::Lower => f64_to_f32_down(if penalty == 0.0 {
                        nominal
                    } else {
                        next_down_f64(nominal - penalty)
                    }),
                    OutwardDirection::Upper => f64_to_f32_up(if penalty == 0.0 {
                        nominal
                    } else {
                        next_up_f64(nominal + penalty)
                    }),
                }
            };
        }
    }
    require_f64_interval_proof_environment()?;
    concat_checkpoint(deadline)
}

fn concat_checkpoint(deadline: Option<Instant>) -> Result<()> {
    if deadline.is_some_and(|limit| Instant::now() >= limit) {
        return Err(NyError::DeadlineExceeded(
            "Concat constant folding exceeded its inherited deadline".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
#[path = "concat_tests.rs"]
mod tests;
