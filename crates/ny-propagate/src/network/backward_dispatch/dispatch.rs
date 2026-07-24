// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Core backward CROWN dispatch logic.
//!
//! Contains [`dispatch_backward_layer`] — the canonical dispatch function.
//! Helper functions are in [`super::helpers`].

use ndarray::Array1;
use ny_core::{NyError, Result};

use crate::bounds::LinearBounds;
use crate::layers::{BoundPropagation, Layer};

use super::helpers::{
    dispatch_conv_engine_aware, dispatch_propagate_linear, preserve_structured_error,
    resolve_input_bounds,
};
use super::types::{BackwardDispatchResult, DispatchContext};

/// Whether the fused self-attention ternary CROWN backward is enabled. ON by
/// default: the directional O(radius^2) McCormick margin makes it STRICTLY TIGHTER
/// than the simplex-aware IBP end-to-end (attention->Linear 0.72x, attention->ReLU
/// ->Linear 0.70x), it is memory-bounded (~30 MB at seq=64 vs ~20 GB de-fused), and
/// it is gated to the provably mask-matched class (windowed-causal sq!=sk falls back
/// to sound IBP). Soundness verified: 18M+ interior samples 0 violations, two
/// adversarially-found false-proof holes (underflow visible key; windowed-causal
/// mask mismatch) closed, a third adversarial audit round reached consensus-sound.
/// Kill-switch: set `NY_ATTN_CROWN_TERNARY=0` to fall back to the prior IBP path.
fn attn_crown_ternary_enabled() -> bool {
    !matches!(
        std::env::var("NY_ATTN_CROWN_TERNARY").ok().as_deref(),
        Some("0")
    )
}

/// Dispatch backward CROWN propagation through a single layer.
///
/// This is the canonical layer dispatch function. It handles:
/// - Linear layers (with optional GEMM engine)
/// - Conv layers (1d/2d, transpose variants — shape setup + propagation)
/// - Transpose (shape setup + propagation)
/// - Flatten, Reshape (linear propagation)
/// - Add, Sub, ExpandLikeLastAxis (binary split)
/// - Concat (N-ary split)
/// - MatMul, BilinearCrown, MulBinary (binary with both input bounds)
/// - SkipMerge, OpaqueSkip (pass-through / identity)
/// - All other unary layers via `propagate_crown_backward` trait
///
/// **Not handled here:** ReLU (alpha/beta/gradient logic varies per site),
/// Where (complex ternary conditional with concretization).
///
/// Returns `BackwardDispatchResult` or an error. `Unsupported` means the
/// layer explicitly doesn't support CROWN backward (multi-input without
/// handler); the caller decides whether to fall back to IBP or propagate
/// the error.
pub(crate) fn dispatch_backward_layer(
    ctx: &DispatchContext<'_>,
    node_lb: &LinearBounds,
) -> Result<BackwardDispatchResult> {
    // SOUNDNESS SAFETY NET (#vnncomp-aw-soundness). When the incoming bounds
    // carry a certified coefficient-error interval, only layers taught to
    // propagate it (Linear, Conv2d, element-wise activations, pass-through
    // reshapes) keep it sound. For EXACT-linear graph ops (Add/Sub/Concat/Slice/
    // Transpose/Tile/Gather/Pad/Conv1d/ConvTranspose/constant-arithmetic), the
    // backward applies a fixed linear column transform `T`; we propagate the error
    // by re-running the SAME backward on the incoming error matrices (as a
    // non-negative carrier) and attaching the carried result as the output error.
    // For any OTHER error-incompatible layer, soundly discharge the error by
    // degrading the affected rows to `[-inf, +inf]` BEFORE dispatch. Rows without
    // error stay precise either way.
    if node_lb.has_coeff_err()
        && !ctx.layer.propagates_coeff_err()
        && ctx.layer.is_exact_linear_coeff_err_carrier()
    {
        return dispatch_exact_linear_with_err(ctx, node_lb);
    }
    // For any other error-incompatible layer (e.g. a nonlinear binary McCormick
    // envelope), discharge the certified error PRECISELY by folding it into the
    // bias over this node's own output box (the value `node_lb`'s coefficients
    // multiply) instead of degrading the whole row to `[-inf, +inf]`. The penalty
    // `Σ_j max(|yl_j|,|yu_j|)·err_ij` exactly bounds the coefficient-interval
    // contribution and remains sound when the layer further transforms the
    // (now error-free) bounds. Falls back to the conservative row degrade only
    // when this node's output box is unavailable.
    let discharged;
    let node_lb: &LinearBounds = if node_lb.has_coeff_err() && !ctx.layer.propagates_coeff_err() {
        let mut tmp = node_lb.clone();
        if let Some(out_box) = ctx.node_bounds.get(ctx.node_name) {
            let flat = out_box.flatten();
            if let (Some(l), Some(u)) = (flat.lower().as_slice(), flat.upper().as_slice()) {
                tmp.fold_coeff_err_into_bias(l, u);
            } else {
                let l: Vec<f32> = flat.lower().iter().copied().collect();
                let u: Vec<f32> = flat.upper().iter().copied().collect();
                tmp.fold_coeff_err_into_bias(&l, &u);
            }
        } else {
            tmp.discharge_coeff_err_to_conservative();
        }
        discharged = tmp;
        &discharged
    } else if node_lb.has_coeff_err()
        && matches!(
            ctx.layer,
            Layer::Conv1d(_)
                | Layer::Conv2d(_)
                | Layer::ConvTranspose1d(_)
                | Layer::ConvTranspose2d(_)
        )
    {
        // #cgan-conv-err-compose: EAGER per-row discharge of the INCOMING
        // certified error over this conv node's OWN OUTPUT box (the value the
        // incoming coefficients multiply — the identical fold identity as the
        // non-carrier discharge above), PREFERRED over letting the conv
        // backward compose the error through `|kernel|`:
        //   - TIGHTER on real chains: the conv's output cut is a pre-ReLU node
        //     that per-node CROWN collection tightens (crown∩IBP map), so the
        //     u-scale relative error discharges at ~u·Σ|e|·mag(tight box),
        //     while the |K|-composition carries it against the ABS kernel mass
        //     (growing by the signed/abs cancellation ratio per layer) until a
        //     later discharge.
        //   - CHEAPER: skips the extra |kernel| GEMM pair per side (measured:
        //     that pair pushed cgan_2023's 28,800-dim BatchNormalization_11
        //     chunked collection past its per-node time share, degrading it to
        //     IBP in-pipeline and costing the root-unsat verdict).
        // Rows with a non-finite penalty keep their error and take the exact
        // |K|-composition fallback in the conv backward; a missing output box
        // (e.g. the sequential collector's empty node map) leaves the error
        // untouched for the same fallback. Sound either way.
        if let Some(out_box) = ctx.node_bounds.get(ctx.node_name) {
            let mut tmp = node_lb.clone();
            tmp.fold_coeff_err_over_box_eager(out_box);
            discharged = tmp;
            &discharged
        } else {
            node_lb
        }
    } else {
        node_lb
    };
    dispatch_backward_layer_inner(ctx, node_lb)
}

/// Carry the certified coefficient error through an EXACT-linear graph op
/// (#vnncomp-aw-soundness).
///
/// Runs the op's backward twice: once on the error-free coefficients (the real
/// result), once on the [`coeff_err_carrier`](LinearBounds::coeff_err_carrier)
/// (the carried error). The carried result is then attached to the real result —
/// its coefficient magnitudes become the per-coefficient error and its bias
/// magnitudes widen the bias OUTWARD. Both runs go through the identical transform,
/// so the carried error exactly tracks `T_abs(err_in)` of the op's column map.
fn dispatch_exact_linear_with_err(
    ctx: &DispatchContext<'_>,
    node_lb: &LinearBounds,
) -> Result<BackwardDispatchResult> {
    let carrier = match node_lb.coeff_err_carrier() {
        Some(c) => c,
        None => {
            // No error to carry (shouldn't happen given has_coeff_err gate).
            let mut plain = node_lb.clone();
            plain.discharge_coeff_err_to_conservative();
            return dispatch_backward_layer_inner(ctx, &plain);
        }
    };
    // Real run on error-free coefficients.
    let mut plain = node_lb.clone();
    plain.lower_a_err = None;
    plain.upper_a_err = None;
    let real = dispatch_backward_layer_inner(ctx, &plain)?;
    // Carrier run produces the transformed error in its coefficients/bias.
    let carried = dispatch_backward_layer_inner(ctx, &carrier)?;
    Ok(merge_carried_err(real, carried))
}

/// Merge a carried-error dispatch result into the real dispatch result by
/// attaching coefficient error and folding bias error OUTWARD.
///
/// If the two results' variants do not line up (which would indicate the op
/// transformed the error carrier differently from the coefficients — it should
/// never happen for a fixed linear transform), the real result is returned
/// undisturbed but with its error discharged to conservative so no tightness is
/// claimed without the penalty.
fn merge_carried_err(
    real: BackwardDispatchResult,
    carried: BackwardDispatchResult,
) -> BackwardDispatchResult {
    use BackwardDispatchResult::{Binary, Nary, PassThrough, Single};
    match (real, carried) {
        (Single(mut a), Single(c)) => {
            a.attach_err_from_carried(&c);
            Single(a)
        }
        (
            Binary {
                mut bounds_a,
                mut bounds_b,
                mut bias_lower,
                mut bias_upper,
            },
            Binary {
                bounds_a: ca,
                bounds_b: cb,
                bias_lower: cbl,
                bias_upper: cbu,
            },
        ) => {
            bounds_a.attach_err_from_carried(&ca);
            bounds_b.attach_err_from_carried(&cb);
            fold_bias_err_outward(&mut bias_lower, &mut bias_upper, &cbl, &cbu);
            Binary {
                bounds_a,
                bounds_b,
                bias_lower,
                bias_upper,
            }
        }
        (
            Nary {
                mut bounds,
                mut bias_lower,
                mut bias_upper,
            },
            Nary {
                bounds: cbounds,
                bias_lower: cbl,
                bias_upper: cbu,
            },
        ) if bounds.len() == cbounds.len() => {
            for (b, c) in bounds.iter_mut().zip(cbounds.iter()) {
                if let (Some(b), Some(c)) = (b.as_mut(), c.as_ref()) {
                    b.attach_err_from_carried(c);
                }
            }
            fold_bias_err_outward(&mut bias_lower, &mut bias_upper, &cbl, &cbu);
            Nary {
                bounds,
                bias_lower,
                bias_upper,
            }
        }
        (PassThrough, _) => PassThrough,
        (other, _) => {
            // Variant mismatch or Unsupported — return real but discharge any
            // residual error rows to stay sound (no tightness without penalty).
            match other {
                Single(mut a) => {
                    a.discharge_coeff_err_to_conservative();
                    Single(a)
                }
                Binary {
                    mut bounds_a,
                    mut bounds_b,
                    bias_lower,
                    bias_upper,
                } => {
                    bounds_a.discharge_coeff_err_to_conservative();
                    bounds_b.discharge_coeff_err_to_conservative();
                    Binary {
                        bounds_a,
                        bounds_b,
                        bias_lower,
                        bias_upper,
                    }
                }
                Nary {
                    mut bounds,
                    bias_lower,
                    bias_upper,
                } => {
                    for b in bounds.iter_mut().flatten() {
                        b.discharge_coeff_err_to_conservative();
                    }
                    Nary {
                        bounds,
                        bias_lower,
                        bias_upper,
                    }
                }
                r => r,
            }
        }
    }
}

/// Fold a carried bias-error pair OUTWARD into the real separate-bias channel:
/// lower decreases by `max(|cbl|,|cbu|)`, upper increases by the same.
fn fold_bias_err_outward(
    bias_lower: &mut Array1<f32>,
    bias_upper: &mut Array1<f32>,
    cbl: &Array1<f32>,
    cbu: &Array1<f32>,
) {
    if bias_lower.len() != cbl.len() || bias_upper.len() != cbu.len() {
        return;
    }
    for i in 0..bias_lower.len() {
        let mag = cbl[i].abs().max(cbu[i].abs());
        if mag != 0.0 && mag.is_finite() {
            bias_lower[i] = ny_tensor::next_down_f32(bias_lower[i] - mag);
            bias_upper[i] = ny_tensor::next_up_f32(bias_upper[i] + mag);
        } else if !mag.is_finite() {
            bias_lower[i] = f32::NEG_INFINITY;
            bias_upper[i] = f32::INFINITY;
        }
    }
}

/// The canonical per-layer backward dispatch match (error-free coefficients).
fn dispatch_backward_layer_inner(
    ctx: &DispatchContext<'_>,
    node_lb: &LinearBounds,
) -> Result<BackwardDispatchResult> {
    match ctx.layer {
        // === Linear: uses engine for GEMM acceleration ===
        // Thread the per-node deadline (#4321): a wide classifier-head GEMM with
        // many spec rows is the single longest uninterrupted op on the spec-matrix
        // root output-bound path. The deadline-aware variant chunks the GEMM over
        // output rows so an over-budget node aborts (DeadlineExceeded -> sound IBP
        // fallback) instead of overrunning the verifier timeout. Bit-identical
        // bounds when not aborting.
        Layer::Linear(l) => {
            let new_lb = l
                .propagate_linear_with_engine_and_deadline(node_lb, ctx.engine, ctx.deadline)
                .map_err(|e| preserve_structured_error(e, ctx.node_name, "Linear"))?
                .into_owned();
            Ok(BackwardDispatchResult::Single(Box::new(new_lb)))
        }
        // === Transpose/Tile/Slice: need input_shape before dispatch (#3105) ===
        Layer::Transpose(t) => {
            let mut t = t.clone();
            t.set_input_shape(ctx.pre_activation.shape().to_vec());
            dispatch_propagate_linear(&t, node_lb, ctx.node_name, "Transpose")
        }
        Layer::Tile(t) => {
            let mut t = t.clone();
            t.set_input_shape(ctx.pre_activation.shape().to_vec());
            dispatch_propagate_linear(&t, node_lb, ctx.node_name, "Tile")
        }
        Layer::Slice(s) => {
            let mut s = s.clone();
            s.set_input_shape(ctx.pre_activation.shape().to_vec());
            dispatch_propagate_linear(&s, node_lb, ctx.node_name, "Slice")
        }
        // === Gather: need input_shape before dispatch (#3400) ===
        Layer::Gather(g) => {
            let mut g = g.clone();
            g.set_input_shape(ctx.pre_activation.shape().to_vec());
            dispatch_propagate_linear(&g, node_lb, ctx.node_name, "Gather")
        }
        // === Conv1d/ConvTranspose1d: 1D shape setup + GPU-accelerated propagation (#3598) ===
        Layer::Conv1d(c) => dispatch_conv_engine_aware(
            c,
            ctx,
            node_lb,
            "Conv1d",
            2,
            |conv, shape, lb, engine, _deadline| {
                conv.set_input_length(shape[shape.len() - 1]);
                conv.propagate_linear_with_engine(lb, engine)
            },
        ),
        Layer::ConvTranspose1d(c) => dispatch_conv_engine_aware(
            c,
            ctx,
            node_lb,
            "ConvTranspose1d",
            2,
            |conv, shape, lb, engine, _deadline| {
                conv.set_input_length(shape[shape.len() - 1]);
                conv.propagate_linear_with_engine(lb, engine)
            },
        ),
        // === Conv2d: use dispatch_conv_engine_aware for preserve_structured_error (#3720) ===
        Layer::Conv2d(c) => dispatch_conv_engine_aware(
            c,
            ctx,
            node_lb,
            "Conv2d",
            3,
            |conv, shape, lb, engine, deadline| {
                conv.set_input_shape(shape[shape.len() - 2], shape[shape.len() - 1]);
                conv.propagate_linear_with_engine_and_deadline(lb, engine, deadline)
            },
        ),
        // === ConvTranspose2d: 2D shape setup + GPU-accelerated propagation ===
        // Previously used dispatch_conv_with_shape_setup which goes through the
        // BoundPropagation trait, discarding the GemmEngine. Now threads engine
        // AND the per-node deadline directly like Conv2d (#wall-deadwork
        // ConvTranspose port): expiry inside the dominant f64 recompute
        // surfaces as DeadlineExceeded, which the collector maps to its sound
        // reference-bounds fallback instead of finishing a doomed walk.
        Layer::ConvTranspose2d(c) => dispatch_conv_engine_aware(
            c,
            ctx,
            node_lb,
            "ConvTranspose2d",
            3,
            |conv, shape, lb, engine, deadline| {
                conv.set_input_shape(shape[shape.len() - 2], shape[shape.len() - 1]);
                conv.propagate_linear_with_engine_and_deadline(lb, engine, deadline)
            },
        ),
        // === Add: binary split with separate bias channel (#2617) ===
        Layer::Add(add) => {
            if ctx.inputs.len() != 2 {
                return Err(NyError::InvalidSpec(format!(
                    "Add node '{}' requires exactly 2 inputs, got {}",
                    ctx.node_name,
                    ctx.inputs.len()
                )));
            }
            // #2617/#2530: Add introduces no local relaxation bias. Carry incoming
            // bias directly via the separate channel and propagate zero-bias A-paths.
            let bias_lower = node_lb.lower_b().clone();
            let bias_upper = node_lb.upper_b().clone();
            let mut zero_bias_lb = node_lb.clone();
            zero_bias_lb.lower_b_mut().fill(0.0);
            zero_bias_lb.upper_b_mut().fill(0.0);
            let (mut lb_a, mut lb_b) = add
                .propagate_linear_binary(&zero_bias_lb)
                .map_err(|e| preserve_structured_error(e, ctx.node_name, "Add"))?;
            lb_a.lower_b_mut().fill(0.0);
            lb_a.upper_b_mut().fill(0.0);
            lb_b.lower_b_mut().fill(0.0);
            lb_b.upper_b_mut().fill(0.0);
            Ok(BackwardDispatchResult::Binary {
                bounds_a: Box::new(lb_a),
                bounds_b: Box::new(lb_b),
                bias_lower,
                bias_upper,
            })
        }
        // === Sub: binary split with sign flip on B, separate bias channel (#2617) ===
        Layer::Sub(sub) => {
            if ctx.inputs.len() != 2 {
                return Err(NyError::InvalidSpec(format!(
                    "Sub node '{}' requires exactly 2 inputs, got {}",
                    ctx.node_name,
                    ctx.inputs.len()
                )));
            }
            // #2617/#2530: Sub introduces no local relaxation bias. Carry incoming
            // bias directly via the separate channel and propagate zero-bias A-paths.
            let bias_lower = node_lb.lower_b().clone();
            let bias_upper = node_lb.upper_b().clone();
            let mut zero_bias_lb = node_lb.clone();
            zero_bias_lb.lower_b_mut().fill(0.0);
            zero_bias_lb.upper_b_mut().fill(0.0);
            let (mut lb_a, mut lb_b) = sub
                .propagate_linear_binary(&zero_bias_lb)
                .map_err(|e| preserve_structured_error(e, ctx.node_name, "Sub"))?;
            lb_a.lower_b_mut().fill(0.0);
            lb_a.upper_b_mut().fill(0.0);
            lb_b.lower_b_mut().fill(0.0);
            lb_b.upper_b_mut().fill(0.0);
            Ok(BackwardDispatchResult::Binary {
                bounds_a: Box::new(lb_a),
                bounds_b: Box::new(lb_b),
                bias_lower,
                bias_upper,
            })
        }
        Layer::ExpandLikeLastAxis(expand) => {
            if ctx.inputs.len() != 2 {
                return Err(NyError::InvalidSpec(format!(
                    "ExpandLikeLastAxis node '{}' requires exactly 2 inputs, got {}",
                    ctx.node_name,
                    ctx.inputs.len()
                )));
            }
            let source_bounds = resolve_input_bounds(
                &ctx.inputs[0],
                ctx.network_input,
                ctx.node_bounds,
                ctx.node_name,
                "ExpandLikeLastAxis source",
            )?;
            let reference_bounds = resolve_input_bounds(
                &ctx.inputs[1],
                ctx.network_input,
                ctx.node_bounds,
                ctx.node_name,
                "ExpandLikeLastAxis reference",
            )?;
            let (mut lb_a, mut lb_b) = expand
                .propagate_linear_binary(node_lb, source_bounds, reference_bounds)
                .map_err(|e| preserve_structured_error(e, ctx.node_name, "ExpandLikeLastAxis"))?;
            let bias_lower = lb_a.lower_b() + lb_b.lower_b();
            let bias_upper = lb_a.upper_b() + lb_b.upper_b();
            lb_a.lower_b_mut().fill(0.0);
            lb_a.upper_b_mut().fill(0.0);
            lb_b.lower_b_mut().fill(0.0);
            lb_b.upper_b_mut().fill(0.0);
            Ok(BackwardDispatchResult::Binary {
                bounds_a: Box::new(lb_a),
                bounds_b: Box::new(lb_b),
                bias_lower,
                bias_upper,
            })
        }
        // === Concat: N-ary split (extracted to concat.rs for file size, #3287) ===
        Layer::Concat(concat) => super::concat::dispatch_concat(concat, ctx, node_lb),
        // === MatMul: binary with both input bounds, separate bias (#2617) ===
        Layer::MatMul(matmul) => {
            if ctx.inputs.len() != 2 {
                return Err(NyError::InvalidSpec(format!(
                    "MatMul node '{}' requires exactly 2 inputs, got {}",
                    ctx.node_name,
                    ctx.inputs.len()
                )));
            }
            let input_a_bounds = resolve_input_bounds(
                &ctx.inputs[0],
                ctx.network_input,
                ctx.node_bounds,
                ctx.node_name,
                "MatMul input A",
            )?;
            let input_b_bounds = resolve_input_bounds(
                &ctx.inputs[1],
                ctx.network_input,
                ctx.node_bounds,
                ctx.node_name,
                "MatMul input B",
            )?;
            // #2617/#2530: MatMul is a relaxation-producing layer (McCormick). Pass full
            // incoming bounds — the layer places all bias (incoming + McCormick) on
            // bounds_a and zeros bounds_b (#2520). Extract total bias post-call via
            // the separate bias channel.
            let (mut lb_a, mut lb_b) = matmul
                .propagate_linear_binary(node_lb, input_a_bounds, input_b_bounds)
                .map_err(|e| preserve_structured_error(e, ctx.node_name, "MatMul"))?;
            let bias_lower = lb_a.lower_b() + lb_b.lower_b();
            let bias_upper = lb_a.upper_b() + lb_b.upper_b();
            lb_a.lower_b_mut().fill(0.0);
            lb_a.upper_b_mut().fill(0.0);
            lb_b.lower_b_mut().fill(0.0);
            lb_b.upper_b_mut().fill(0.0);
            Ok(BackwardDispatchResult::Binary {
                bounds_a: Box::new(lb_a),
                bounds_b: Box::new(lb_b),
                bias_lower,
                bias_upper,
            })
        }
        // === BilinearCrown: binary with both input bounds, separate bias (#2617) ===
        Layer::BilinearCrown(bilinear) => {
            if ctx.inputs.len() != 2 {
                return Err(NyError::InvalidSpec(format!(
                    "BilinearCrown node '{}' requires exactly 2 inputs, got {}",
                    ctx.node_name,
                    ctx.inputs.len()
                )));
            }
            let input_a_bounds = resolve_input_bounds(
                &ctx.inputs[0],
                ctx.network_input,
                ctx.node_bounds,
                ctx.node_name,
                "BilinearCrown input A",
            )?;
            let input_b_bounds = resolve_input_bounds(
                &ctx.inputs[1],
                ctx.network_input,
                ctx.node_bounds,
                ctx.node_name,
                "BilinearCrown input B",
            )?;
            // Use alpha-parameterized McCormick if bilinear alphas are available
            // for this node (#3287). Falls back to fixed midpoint when absent.
            let node_alpha = ctx.bilinear_alphas.and_then(|m| m.get(ctx.node_name));
            let (mut lb_a, mut lb_b) = bilinear
                .propagate_linear_binary_with_alpha(
                    node_lb,
                    input_a_bounds,
                    input_b_bounds,
                    node_alpha,
                )
                .map_err(|e| preserve_structured_error(e, ctx.node_name, "BilinearCrown"))?;
            let bias_lower = lb_a.lower_b() + lb_b.lower_b();
            let bias_upper = lb_a.upper_b() + lb_b.upper_b();
            lb_a.lower_b_mut().fill(0.0);
            lb_a.upper_b_mut().fill(0.0);
            lb_b.lower_b_mut().fill(0.0);
            lb_b.upper_b_mut().fill(0.0);
            Ok(BackwardDispatchResult::Binary {
                bounds_a: Box::new(lb_a),
                bounds_b: Box::new(lb_b),
                bias_lower,
                bias_upper,
            })
        }
        // === SkipMerge: pass bounds directly to single input ===
        Layer::SkipMerge(_) => {
            if ctx.inputs.len() != 1 {
                return Err(NyError::InvalidSpec(format!(
                    "SkipMerge node {} expects exactly 1 input, got {}. \
                     Use OpaqueSkip for multi-input skipped ops.",
                    ctx.node_name,
                    ctx.inputs.len()
                )));
            }
            Ok(BackwardDispatchResult::PassThrough)
        }
        // === OpaqueSkip: identity propagation ===
        Layer::OpaqueSkip(os) => {
            let new_lb = os
                .propagate_linear(node_lb)
                .map_err(|e| preserve_structured_error(e, ctx.node_name, "OpaqueSkip"))?
                .into_owned();
            if ctx.inputs.is_empty() {
                return Err(NyError::InvalidSpec(format!(
                    "OpaqueSkip node {} has no inputs",
                    ctx.node_name
                )));
            }
            if ctx.inputs.len() == 1 {
                Ok(BackwardDispatchResult::Single(Box::new(new_lb)))
            } else {
                // OpaqueSkip is a linear identity — no relaxation bias.
                // Extract incoming bias into separate channel, distribute
                // zero-biased bounds to all inputs.
                let (la, lb, ua, ub) = new_lb.into_parts();
                let bias_lower = lb;
                let bias_upper = ub;
                // Phase 4 audit: coefficients from upstream propagation + zero biases.
                let zeroed = LinearBounds::new_or_conservative(
                    la,
                    Array1::zeros(bias_lower.len()),
                    ua,
                    Array1::zeros(bias_upper.len()),
                )?;
                let bounds: Vec<Option<LinearBounds>> =
                    ctx.inputs.iter().map(|_| Some(zeroed.clone())).collect();
                Ok(BackwardDispatchResult::Nary {
                    bounds,
                    bias_lower,
                    bias_upper,
                })
            }
        }
        // === ReLU: NOT handled here ===
        // ReLU dispatch requires alpha/beta/gradient state that varies per site.
        // Each caller must handle ReLU before calling this function.
        Layer::ReLU(_) => Ok(BackwardDispatchResult::Unsupported(
            "ReLU must be handled by caller (requires alpha/beta state)".to_string(),
        )),
        // === MulBinary: McCormick envelope CROWN backward (#3439) ===
        // Element-wise z = x * y with McCormick relaxation. Follows the same
        // binary split pattern as MatMul/BilinearCrown with separate bias channel.
        // Reference: auto_LiRPA operators/bivariate.py:40-75 (McCormick envelopes),
        //            layers/binary_ops/mul/mod.rs:252 (existing implementation).
        Layer::MulBinary(mul) => {
            if ctx.inputs.len() != 2 {
                return Err(NyError::InvalidSpec(format!(
                    "MulBinary node '{}' requires exactly 2 inputs, got {}",
                    ctx.node_name,
                    ctx.inputs.len()
                )));
            }
            let input_a_bounds = resolve_input_bounds(
                &ctx.inputs[0],
                ctx.network_input,
                ctx.node_bounds,
                ctx.node_name,
                "MulBinary input A",
            )?;
            let input_b_bounds = resolve_input_bounds(
                &ctx.inputs[1],
                ctx.network_input,
                ctx.node_bounds,
                ctx.node_name,
                "MulBinary input B",
            )?;
            // #3439 Phase 2: alpha-parameterized McCormick when available,
            // fixed relaxation mode when not.
            let node_alpha = ctx.mul_binary_alphas.and_then(|m| m.get(ctx.node_name));
            let (mut lb_a, mut lb_b) = if node_alpha.is_some() {
                mul.propagate_linear_binary_with_alpha(
                    node_lb,
                    input_a_bounds,
                    input_b_bounds,
                    node_alpha,
                )
            } else {
                mul.propagate_linear_binary(
                    node_lb,
                    input_a_bounds,
                    input_b_bounds,
                    ctx.mul_binary_relaxation,
                )
            }
                .map_err(|e| preserve_structured_error(e, ctx.node_name, "MulBinary"))?;
            let bias_lower = lb_a.lower_b() + lb_b.lower_b();
            let bias_upper = lb_a.upper_b() + lb_b.upper_b();
            lb_a.lower_b_mut().fill(0.0);
            lb_a.upper_b_mut().fill(0.0);
            lb_b.lower_b_mut().fill(0.0);
            lb_b.upper_b_mut().fill(0.0);
            Ok(BackwardDispatchResult::Binary {
                bounds_a: Box::new(lb_a),
                bounds_b: Box::new(lb_b),
                bias_lower,
                bias_upper,
            })
        }
        // === Where: NOT handled here ===
        // Where requires ternary conditional logic with concretization.
        Layer::Where(_) => Ok(BackwardDispatchResult::Unsupported(
            "Where must be handled by caller (requires ternary conditional logic)".to_string(),
        )),
        // === ScatterAdd / IndexAdd / ScatterND: exact linear CROWN backward when
        // indices are constant and exactly one operand is variable (#yolo). These
        // are gather-scatter linear ops; the backward is the exact transpose of the
        // scatter. Data-dependent indices or multiple variable operands fall back to
        // the sound IBP union via the Unsupported path below.
        Layer::ScatterAdd(layer) => match layer.propagate_linear(node_lb) {
            Ok(new_lb) => Ok(BackwardDispatchResult::Single(Box::new(new_lb.into_owned()))),
            Err(
                NyError::UnsupportedOp(msg)
                | NyError::UnsupportedConfiguration(msg)
                | NyError::NumericalInstability(msg),
            ) => Ok(BackwardDispatchResult::Unsupported(msg)),
            Err(err @ NyError::ShapeMismatch { .. }) => {
                Ok(BackwardDispatchResult::Unsupported(format!(
                    "ScatterAdd shape: {err}"
                )))
            }
            Err(err) => Err(preserve_structured_error(err, ctx.node_name, "ScatterAdd")),
        },
        Layer::IndexAdd(layer) => match layer.propagate_linear(node_lb) {
            Ok(new_lb) => Ok(BackwardDispatchResult::Single(Box::new(new_lb.into_owned()))),
            Err(
                NyError::UnsupportedOp(msg)
                | NyError::UnsupportedConfiguration(msg)
                | NyError::NumericalInstability(msg),
            ) => Ok(BackwardDispatchResult::Unsupported(msg)),
            Err(err @ NyError::ShapeMismatch { .. }) => {
                Ok(BackwardDispatchResult::Unsupported(format!(
                    "IndexAdd shape: {err}"
                )))
            }
            Err(err) => Err(preserve_structured_error(err, ctx.node_name, "IndexAdd")),
        },
        Layer::ScatterNd(layer) => match layer.propagate_linear(node_lb) {
            Ok(new_lb) => Ok(BackwardDispatchResult::Single(Box::new(new_lb.into_owned()))),
            Err(
                NyError::UnsupportedOp(msg)
                | NyError::UnsupportedConfiguration(msg)
                | NyError::NumericalInstability(msg),
            ) => Ok(BackwardDispatchResult::Unsupported(msg)),
            Err(err @ NyError::ShapeMismatch { .. }) => {
                Ok(BackwardDispatchResult::Unsupported(format!(
                    "ScatterND shape: {err}"
                )))
            }
            Err(err) => Err(preserve_structured_error(err, ctx.node_name, "ScatterND")),
        },
        // === Variable-style AdaIN: ternary CROWN ===
        // Must be before the unary catch-all. Fixed-style AdaIN stays in the unary arm.
        Layer::AdaIN1d(adain) if adain.requires_style_inputs() => {
            if ctx.inputs.len() < 3 {
                return Err(NyError::InvalidSpec(format!(
                    "Variable-style AdaIN1d node '{}' requires 3 inputs, got {}",
                    ctx.node_name,
                    ctx.inputs.len()
                )));
            }
            let g_bounds = resolve_input_bounds(
                &ctx.inputs[1],
                ctx.network_input,
                ctx.node_bounds,
                ctx.node_name,
                "style_gamma input",
            )?;
            let b_bounds = resolve_input_bounds(
                &ctx.inputs[2],
                ctx.network_input,
                ctx.node_bounds,
                ctx.node_name,
                "style_beta input",
            )?;
            match adain.propagate_crown_ternary(node_lb, ctx.pre_activation, g_bounds, b_bounds) {
                Ok((bounds, bias_lower, bias_upper)) => {
                    Ok(BackwardDispatchResult::Nary {
                        bounds,
                        bias_lower,
                        bias_upper,
                    })
                }
                Err(
                    NyError::UnsupportedOp(msg)
                    | NyError::UnsupportedConfiguration(msg)
                    | NyError::NumericalInstability(msg),
                ) => Ok(BackwardDispatchResult::Unsupported(msg)),
                Err(err) => Err(err),
            }
        }
        // === Fused self-attention: ternary CROWN (DEFAULT-ON) ===
        // A SOUND, memory-bounded center-point linearization with a provably
        // POINTWISE-sound IBP-validated margin (mirrors the AdaIN ternary
        // surface). It un-blocks CROWN continuation past a fused attention node
        // (which otherwise returns `UnsupportedOp`, aborting the whole graph to
        // IBP). STATUS (corrected 2026-07-05): this is now **enabled by default**
        // — `attn_crown_ternary_enabled()` is `true` unless `NY_ATTN_CROWN_TERNARY=0`
        // — because on the boxes verification actually uses it is *tighter* than
        // the simplex-aware IBP (an earlier note claiming it merely ties IBP is
        // stale; the default was flipped on). Set `NY_ATTN_CROWN_TERNARY=0` to
        // force the byte-identical IBP fallback. Returns `Unsupported` on any
        // unsupported shape / non-finite intermediate, so the caller always falls
        // back to the same sound IBP. (#attn-crown)
        Layer::SelfAttention(attn) => {
            if !attn_crown_ternary_enabled() {
                return Ok(BackwardDispatchResult::Unsupported(
                    "SelfAttention CROWN ternary disabled (set NY_ATTN_CROWN_TERNARY=1)".to_string(),
                ));
            }
            if ctx.inputs.len() < 3 {
                return Err(NyError::InvalidSpec(format!(
                    "SelfAttention node '{}' requires 3 inputs (Q, K, V), got {}",
                    ctx.node_name,
                    ctx.inputs.len()
                )));
            }
            let k_bounds = resolve_input_bounds(
                &ctx.inputs[1],
                ctx.network_input,
                ctx.node_bounds,
                ctx.node_name,
                "attention K input",
            )?;
            let v_bounds = resolve_input_bounds(
                &ctx.inputs[2],
                ctx.network_input,
                ctx.node_bounds,
                ctx.node_name,
                "attention V input",
            )?;
            // ctx.pre_activation is the first input's (Q) bounds.
            match attn.propagate_crown_ternary(node_lb, ctx.pre_activation, k_bounds, v_bounds) {
                Ok((bounds, bias_lower, bias_upper)) => Ok(BackwardDispatchResult::Nary {
                    bounds,
                    bias_lower,
                    bias_upper,
                }),
                Err(
                    NyError::UnsupportedOp(msg)
                    | NyError::UnsupportedConfiguration(msg)
                    | NyError::NumericalInstability(msg),
                ) => Ok(BackwardDispatchResult::Unsupported(msg)),
                Err(err) => Err(err),
            }
        }
        // === RmsNorm: GenBaB inv_rms norm branching (#norm-genbab) ===
        // Special-cased ahead of the generic unary arm so a per-node `inv_rms`
        // range override (from a GenBaB norm split) can be threaded into the
        // decomposed RmsNorm CROWN. With no override this is byte-identical to
        // the generic `propagate_crown_backward` path (which routes IbpValidated
        // RmsNorm through the same decomposed backward). The override only ever
        // NARROWS the certified inv_rms interval (intersection), so it cannot
        // make any row's relaxation unsound; the sibling child reclaims the
        // excluded inv_rms range (see `InvRmsOverride`).
        Layer::RmsNorm(rms) => {
            let inv_rms = ctx
                .norm_inv_rms_override
                .and_then(|map| map.get(ctx.node_name))
                .map(|windows| windows.as_slice());
            match rms.propagate_linear_with_bounds_inv_rms(node_lb, ctx.pre_activation, inv_rms) {
                Ok(new_lb) => Ok(BackwardDispatchResult::Single(Box::new(new_lb))),
                Err(
                    NyError::UnsupportedOp(msg)
                    | NyError::UnsupportedConfiguration(msg)
                    | NyError::NumericalInstability(msg),
                ) => Ok(BackwardDispatchResult::Unsupported(msg)),
                Err(err @ NyError::ShapeMismatch { .. }) => Err(err),
                Err(err @ NyError::SoundnessRefusal(_) | err @ NyError::InternalError(_)) => {
                    Err(err)
                }
                Err(err) => Err(NyError::InvalidSpec(format!(
                    "CROWN failed at node '{}' (RmsNorm): {}",
                    ctx.node_name, err
                ))),
            }
        }
        // === Remaining unary layers: unified trait dispatch (#3424) ===
        // Every variant listed explicitly — no catch-all. Adding a new Layer
        // variant without a dispatch arm here is a compile error.
        //
        // Elementwise activations:
        Layer::GELU(_) | Layer::SiLU(_) | Layer::Tanh(_) | Layer::Sigmoid(_)
        | Layer::Exp(_) | Layer::Log(_) | Layer::Sqrt(_) | Layer::Reciprocal(_)
        | Layer::Softplus(_) | Layer::HardSwish(_) | Layer::Mish(_) | Layer::Selu(_)
        | Layer::Softsign(_) | Layer::Arctan(_) | Layer::Tan(_) | Layer::Sin(_)
        | Layer::Cos(_) | Layer::Elu(_) | Layer::Celu(_) | Layer::LeakyReLU(_)
        | Layer::HardSigmoid(_) | Layer::Clip(_) | Layer::ThresholdedRelu(_)
        | Layer::Abs(_) | Layer::PowConstant(_) | Layer::Floor(_) | Layer::Ceil(_)
        | Layer::Round(_) | Layer::Trunc(_) | Layer::Sign(_) | Layer::PRelu(_) | Layer::Shrink(_)
        | Layer::Snake(_) | Layer::Compare(_)
        // Softmax family:
        | Layer::Softmax(_) | Layer::CausalSoftmax(_) | Layer::LogSoftmax(_)
        | Layer::LogSumExp(_)
        // Normalization (fixed-style AdaIN stays here as unary; RmsNorm has its
        // own arm above for GenBaB inv_rms branching):
        | Layer::LayerNorm(_) | Layer::InstanceNorm1d(_)
        | Layer::GroupNorm(_) | Layer::AdaIN1d(_) | Layer::BatchNorm(_)
        // Constant arithmetic:
        | Layer::AddConstant(_) | Layer::MulConstant(_) | Layer::DivConstant(_)
        | Layer::SubConstant(_)
        // Reductions:
        | Layer::ReduceMean(_) | Layer::ReduceSum(_) | Layer::CumSum(_)
        | Layer::ReduceMax(_) | Layer::ReduceMin(_)
        | Layer::Topk(_) | Layer::ArgMax(_) | Layer::ArgMin(_) | Layer::ArgSort(_)
        // Shape transforms:
        | Layer::Flatten(_) | Layer::Reshape(_) | Layer::Squeeze(_)
        | Layer::Unsqueeze(_) | Layer::Pad(_) | Layer::Resize(_)
        | Layer::QdqPerturbation(_)
        // Pooling:
        | Layer::AveragePool(_) | Layer::MaxPool2d(_)
        // Positional encoding:
        | Layer::RoPE(_)
        // Special unary (trait handles rejection for data-dependent ops):
        | Layer::NonZero(_)
        // Binary comparison (IBP-only, CROWN returns UnsupportedOp):
        | Layer::CompareTensor(_) => {
            match ctx
                .layer
                .propagate_crown_backward(node_lb, Some(ctx.pre_activation))
            {
                Ok(new_lb) => Ok(BackwardDispatchResult::Single(Box::new(new_lb))),
                // #3166, #2888, #3602: Unsupported/shape errors degrade to IBP fallback.
                Err(
                    NyError::UnsupportedOp(msg)
                    | NyError::UnsupportedConfiguration(msg)
                    | NyError::NumericalInstability(msg),
                ) => Ok(BackwardDispatchResult::Unsupported(msg)),
                Err(err @ NyError::ShapeMismatch { .. }) => Err(err),
                Err(err @ NyError::SoundnessRefusal(_) | err @ NyError::InternalError(_)) => {
                    Err(err)
                }
                Err(err) => Err(NyError::InvalidSpec(format!(
                    "CROWN failed at node '{}' ({}): {}",
                    ctx.node_name,
                    ctx.layer.layer_type(),
                    err
                ))),
            }
        }
        // === MinBinary / MaxBinary / Atan2: sound linear CROWN backward ===
        // z = min(x, y) / max(x, y) are piecewise-linear (concave / convex);
        // z = atan2(y, x) is C^1 with bounded gradient away from the origin and
        // branch cut. All three admit sound linear envelopes over the input box
        // and use the same binary split + separate bias channel convention as
        // MulBinary. Atan2 falls back to IBP (UnsupportedOp) for boxes near the
        // origin or straddling the branch cut.
        Layer::MinBinary(_) | Layer::MaxBinary(_) | Layer::Atan2(_) => {
            if ctx.inputs.len() != 2 {
                return Err(NyError::InvalidSpec(format!(
                    "{} node '{}' requires exactly 2 inputs, got {}",
                    ctx.layer.layer_type(),
                    ctx.node_name,
                    ctx.inputs.len()
                )));
            }
            let input_a_bounds = resolve_input_bounds(
                &ctx.inputs[0],
                ctx.network_input,
                ctx.node_bounds,
                ctx.node_name,
                "binary linear-envelope input A",
            )?;
            let input_b_bounds = resolve_input_bounds(
                &ctx.inputs[1],
                ctx.network_input,
                ctx.node_bounds,
                ctx.node_name,
                "binary linear-envelope input B",
            )?;
            let op_label = ctx.layer.layer_type();
            let res = match ctx.layer {
                Layer::MinBinary(layer) => {
                    layer.propagate_linear_binary(node_lb, input_a_bounds, input_b_bounds)
                }
                Layer::MaxBinary(layer) => {
                    layer.propagate_linear_binary(node_lb, input_a_bounds, input_b_bounds)
                }
                Layer::Atan2(layer) => {
                    layer.propagate_linear_binary(node_lb, input_a_bounds, input_b_bounds)
                }
                _ => unreachable!("matched MinBinary | MaxBinary | Atan2 above"),
            };
            match res {
                Ok((mut lb_a, mut lb_b)) => {
                    let bias_lower = lb_a.lower_b() + lb_b.lower_b();
                    let bias_upper = lb_a.upper_b() + lb_b.upper_b();
                    lb_a.lower_b_mut().fill(0.0);
                    lb_a.upper_b_mut().fill(0.0);
                    lb_b.lower_b_mut().fill(0.0);
                    lb_b.upper_b_mut().fill(0.0);
                    Ok(BackwardDispatchResult::Binary {
                        bounds_a: Box::new(lb_a),
                        bounds_b: Box::new(lb_b),
                        bias_lower,
                        bias_upper,
                    })
                }
                // Non-finite box or shape issue → sound IBP fallback.
                Err(
                    NyError::UnsupportedOp(msg)
                    | NyError::UnsupportedConfiguration(msg)
                    | NyError::NumericalInstability(msg),
                ) => Ok(BackwardDispatchResult::Unsupported(msg)),
                Err(err @ NyError::ShapeMismatch { .. }) => Ok(
                    BackwardDispatchResult::Unsupported(format!("{op_label} shape: {err}")),
                ),
                Err(err) => Err(preserve_structured_error(err, ctx.node_name, op_label)),
            }
        }
        // === Div without explicit handler (#3424) ===
        // Div has a site-specific reciprocal-scaling handler in the graph CROWN
        // coordinator; it reports Unsupported here so callers fall back to IBP
        // if they reach this arm. (Atan2 now has a linear-envelope handler in
        // the arm above; only the origin / branch-cut cases fall back to IBP.)
        Layer::Div(_) => {
            Ok(BackwardDispatchResult::Unsupported(format!(
                "{} CROWN backward requires multi-input handling (not in canonical dispatch)",
                ctx.layer.layer_type()
            )))
        }
        // NO catch-all: compiler catches new Layer variants.
    }
}
