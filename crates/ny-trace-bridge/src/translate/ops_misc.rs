// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Misc elementwise / indexing / masking family: `SwiGlu`, `Powf`, `Fract`,
//! `Atan2`, `Cumsum`, `Flip`, `Roll`, `RepeatInterleave`, `Arange`, `Triu`,
//! `Tril`, `SliceSet`, `Unfold`, `IndexSelect`, `Gather`, `Compare`,
//! `CompareTensor`, `WhereCond`, `ScatterAdd`, `IndexAdd`, `IndexPut`,
//! `MoeGating`.
//!
//! Ported from NN's `trace_to_graph_layerspec_dispatch_extended.rs` /
//! `trace_to_graph_layerspec_decompose{,_scan}.rs`, retaining compatible
//! emissions where they preserve semantics and failing closed where the legacy
//! lowering or serialized data is insufficient.
//!
//! ## Implemented and audited behavior
//!
//! - `SwiGlu` → `SiLU(gate) * up` (SiLU + Mul).
//! - `Powf` → exact primitives for exponents 0 / 0.5 / 1 / 2 / −1 / −0.5;
//!   every other exponent is refused because the former
//!   `Exp(n · Log(Abs(x)))` lowering lost the sign of negative bases.
//! - `Fract` is refused: source semantics use truncation toward zero, while
//!   the former `x − floor(x)` lowering was wrong for negative inputs.
//! - `Atan2` → `LayerType::Atan2` binary spec.
//! - `Cumsum` → N × Slice + (N−1) × Add + Concat; dims above the decompose cap
//!   are refused instead of receiving a semantics-changing identity shortcut.
//! - `Flip` → N × Slice (reversed) + Concat.
//! - `RepeatInterleave` is refused because its serialized op omits the
//!   per-element repeat counts needed to reconstruct source semantics.
//! - `Arange` → constant weight tensor (data-independent, fixed at trace time).
//! - `Triu` / `Tril` → `LayerType::Triu` / `Tril` with `diagonal` attribute.
//! - `SliceSet` → up to 3 × Slice + Concat (before / src / after).
//! - `Unfold` → per-window Slice (+ Transpose) + Reshape, Concat, final Reshape.
//! - `IndexSelect` / `Gather` → `LayerType::Gather` with `axis` attribute.
//!   Unlike NN (which emits unconditionally and lets the graph build fail on a
//!   missing index tensor), a < 2-input node is refused here fail-closed.
//!
//! ## Refused (fail-closed, deliberate)
//!
//! - `Roll`: NN has no dedicated lowering — it falls through to NN's opaque
//!   `LayerType::Unknown` → OpaqueSkip catch-all, a vacuous `[-inf, +inf]`
//!   layer. The bridge reserves the vacuous OpaqueSkip lowering for
//!   `TraceOp::Custom` — the *explicitly* opaque escape hatch, where ±inf is
//!   the only sound treatment ([`crate::translate`] module docs). `Roll` is a
//!   KNOWN op with computable semantics, so it keeps the `UnsupportedOp`
//!   refusal: a real lowering, not a precision-destroying fallback, is the fix.
//! - `ScatterAdd` / `IndexAdd` / `IndexPut` / `MoeGating`: NN lowers these as
//!   identity passthroughs (`Add(x, 0)`), but that is **not** a sound
//!   over-approximation — scatter/index accumulation and replacement can move
//!   output values outside the first input's bounds, and MoE routing is
//!   data-dependent top-k. [`crate::coverage`] classifies all four
//!   `Unsupported` (must always be refused).
//! - `Compare` / `CompareTensor`: lowered to ny-propagate's Compare /
//!   CompareTensor layers ({0,1}-interval IBP, sound; no CROWN linear
//!   relaxation exists for a step, so these are IBP-only). `WhereCond` stays
//!   refused: NN's `m*x + (1-m)*y` decompose is unsound when a realizable
//!   mask value is outside {0,1}.
//!
//! (`ToDtype` is NOT in scope: `ops_core` fails closed because the trace omits
//! its source dtype. `Scatter`,
//! `Topk`, `Argmax`, `Argmin`, `ArgSort`, `Sort` are NOT in scope either:
//! they stay in the dispatch's refused-forever arm.)

use std::collections::HashMap;

use ndarray::{ArrayD, IxDyn};
use ny_build::{AttributeValue, LayerSpec};
use ny_core::{checked_shape_product, LayerType, NyError, Result};

use crate::schema::{CompareOp, TraceNode, TraceOp};

use super::{
    checked_f64_to_f32, dim_as_i64, first_input, insert_scalar_constant, op_name, ops_core,
    shape_to_i64, simple_spec, Ctx, NodeOutput,
};

// O(N) decompositions (Flip, Cumsum, Unfold) emit N nodes per dimension-size
// element. Capped to avoid exploding graph sizes. (Historically mirrored the
// legacy nn translator's `MAX_DECOMPOSE_DIM`; that translator is deleted and
// this constant is now the single source of truth — nn's
// `HarmonicSourceBounds::MAX_DECOMPOSE_DIM` matches it.)
const MAX_DECOMPOSE_DIM: usize = 2048;

/// Match the loader's constant-folding resource limit for materialized ranges.
const MAX_ARANGE_ELEMENTS: usize = 10_000_000;

/// Translate a misc-family op (`SwiGlu`, `Powf`, `Fract`, `Atan2`, `Cumsum`, `Flip`, `Roll`, `RepeatInterleave`, `Arange`, `Triu`, `Tril`, `SliceSet`, `Unfold`, `IndexSelect`, `Gather`, `Compare`, `CompareTensor`, `WhereCond`, `ScatterAdd`, `IndexAdd`, `IndexPut`, `MoeGating`) node.
///
/// See the module docs for the implemented / refused split. Refusals return
/// the exact [`NyError::UnsupportedOp`] error (same type, same message shape)
/// the pre-split catch-all arm produced.
pub(super) fn translate_misc(
    node: &TraceNode,
    name: &str,
    input_tensors: &[String],
    output_tensor: &str,
    _node_names: &HashMap<u64, String>,
    ctx: &mut Ctx,
) -> Result<NodeOutput> {
    let output_shape = &node.output_shape;
    match &node.op {
        TraceOp::SwiGlu => translate_swiglu(name, input_tensors, output_tensor),
        TraceOp::Powf { exponent } => {
            translate_powf(name, *exponent, input_tensors, output_tensor, ctx)
        }
        TraceOp::Fract => Err(NyError::UnsupportedOp(
            "Fract has no sound lowering for truncation-toward-zero semantics".to_string(),
        )),
        TraceOp::Atan2 => translate_atan2(name, input_tensors, output_tensor),
        // Cumsum preserves rank: trailing-relative axis vs the output rank.
        // Mirrors NN's post-rework dispatch (nn d7144ea7).
        TraceOp::Cumsum { dim } => translate_cumsum(
            name,
            *dim,
            super::trailing_axis(*dim, output_shape.len(), "Cumsum axis")?,
            input_tensors,
            output_tensor,
            output_shape,
        ),
        // Flip preserves rank: trailing-relative axis vs the output rank.
        TraceOp::Flip { dim } => translate_flip(
            name,
            *dim,
            super::trailing_axis(*dim, output_shape.len(), "Flip axis")?,
            input_tensors,
            output_tensor,
            output_shape,
        ),
        TraceOp::RepeatInterleave { .. } => Err(NyError::UnsupportedOp(
            "RepeatInterleave cannot be lowered because per-element repeat counts are not serialized"
                .to_string(),
        )),
        TraceOp::Arange { start, end, step } => {
            translate_arange(name, *start, *end, *step, output_shape, output_tensor, ctx)
        }
        TraceOp::Triu { diagonal } => {
            translate_triangular(name, *diagonal, false, input_tensors, output_tensor)
        }
        TraceOp::Tril { diagonal } => {
            translate_triangular(name, *diagonal, true, input_tensors, output_tensor)
        }
        TraceOp::SliceSet { dim, start } => {
            // Decompose: Slice(before) + src + Slice(after) → Concat.
            // Requires both self and src input shapes from context.
            if input_tensors.len() < 2 {
                return Err(NyError::InternalError(
                    "SliceSet requires 2 inputs (self, src)".to_string(),
                ));
            }
            let self_shape = lookup_tensor_shape(ctx, &input_tensors[0], "SliceSet: self")?;
            let src_shape = lookup_tensor_shape(ctx, &input_tensors[1], "SliceSet: src")?;
            // Trailing-relative axis vs the self tensor's rank (nn d7144ea7).
            translate_slice_set(
                name,
                *dim,
                super::trailing_axis(*dim, self_shape.len(), "SliceSet axis")?,
                *start,
                input_tensors,
                output_tensor,
                output_shape,
                &self_shape,
                &src_shape,
            )
        }
        TraceOp::Unfold { dim, size, step } => {
            // Decompose: N x Slice + Transpose + Reshape → Concat.
            // Requires input shape from context.
            let data_input = first_input(input_tensors, "Unfold")?;
            let input_shape = lookup_tensor_shape(ctx, &data_input, "Unfold")?;
            // Trailing-relative axis vs the input tensor's rank (nn d7144ea7).
            translate_unfold(
                name,
                *dim,
                super::trailing_axis(*dim, input_shape.len(), "Unfold axis")?,
                *size,
                *step,
                input_tensors,
                output_tensor,
                output_shape,
                &input_shape,
            )
        }
        // Both map to LayerType::Gather with a trailing-relative (negative)
        // axis attribute vs the data tensor's rank. The graph-build backend's
        // convert_gather handles constant vs dynamic indices.
        TraceOp::IndexSelect { dim } => {
            translate_gather("IndexSelect", name, *dim, input_tensors, output_tensor, ctx)
        }
        TraceOp::Gather { dim } => {
            translate_gather("Gather", name, *dim, input_tensors, output_tensor, ctx)
        }

        // Both lower to real ny-propagate comparison layers with sound
        // {0,1}-interval IBP semantics (exact when the operands' intervals
        // do not straddle, [0,1] hull when they do); IBP-only by design.
        TraceOp::Compare { op, value } => {
            translate_compare_scalar(name, *op, *value, input_tensors, output_tensor, ctx)
        }
        TraceOp::CompareTensor { op } => {
            translate_compare_tensor(name, *op, input_tensors, output_tensor)
        }

        // -- Refused fail-closed (see module docs for the per-op rationale):
        //    Roll (NN lowers it only via the vacuous OpaqueSkip catch-all);
        //    ScatterAdd / IndexAdd / IndexPut / MoeGating (NN's identity
        //    passthrough is not a sound over-approximation); WhereCond
        //    (NN's m*x + (1-m)*y decompose is unsound for non-binary masks). --
        op => Err(NyError::UnsupportedOp(format!(
            "{} not supported in NY trace translation",
            op_name(op)
        ))),
    }
}

// ---------------------------------------------------------------------------
// Elementwise decompositions
// ---------------------------------------------------------------------------

/// Translate `TraceOp::SwiGlu` by decomposing into `SiLU(gate) * up`.
///
/// SwiGlu requires exactly 2 inputs: gate (`input[0]`) and up (`input[1]`).
/// Mirrors NN's `translate_swiglu` (#3557).
fn translate_swiglu(
    name: &str,
    input_tensors: &[String],
    output_tensor: &str,
) -> Result<NodeOutput> {
    if input_tensors.len() < 2 {
        return Err(NyError::UnsupportedOp(format!(
            "SwiGlu requires 2 inputs (gate, up), got {}",
            input_tensors.len()
        )));
    }
    let gate = input_tensors[0].clone();
    let up = input_tensors[1].clone();

    // Step 1: SiLU(gate).
    let silu_name = format!("{name}_silu");
    let silu_out = format!("{silu_name}_out");
    let silu_spec = simple_spec(
        &silu_name,
        LayerType::SiLU,
        vec![gate],
        &silu_out,
        HashMap::new(),
    );

    // Step 2: silu_out * up.
    let mul_spec = simple_spec(
        name,
        LayerType::Mul,
        vec![silu_out, up],
        output_tensor,
        HashMap::new(),
    );

    Ok(NodeOutput {
        specs: vec![silu_spec, mul_spec],
    })
}

/// Translate only independently sound `TraceOp::Powf` special cases.
///
/// Special cases map to exact primitives (mirrors NN's `translate_powf`,
/// #3557):
///   - 0 → Clip\[1,1\] (constant 1.0)
///   - 0.5 → Sqrt
///   - 1.0 → identity (`Add + 0`)
///   - 2.0 → Sqr (Pow(2))
///   - −1.0 → Reciprocal
///   - −0.5 → Sqrt + Reciprocal
fn translate_powf(
    name: &str,
    exponent: f64,
    input_tensors: &[String],
    output_tensor: &str,
    ctx: &mut Ctx,
) -> Result<NodeOutput> {
    let data_input = first_input(input_tensors, "Powf")?;

    // Special cases for exact primitives.
    if exponent == 0.0 {
        // x^0 = 1.0 — emit Clip[1,1] to produce constant 1.0.
        let mut attrs = HashMap::new();
        attrs.insert("min".to_string(), AttributeValue::Float(1.0));
        attrs.insert("max".to_string(), AttributeValue::Float(1.0));
        return Ok(NodeOutput::one(simple_spec(
            name,
            LayerType::Clip,
            vec![data_input],
            output_tensor,
            attrs,
        )));
    }
    if exponent == 1.0 {
        return ops_core::translate_identity_add_zero(
            name,
            "Powf exponent 1",
            input_tensors,
            output_tensor,
            ctx,
        );
    }
    if exponent == 2.0 {
        return ops_core::translate_sqr(name, input_tensors, output_tensor, ctx);
    }
    if exponent == 0.5 {
        return ops_core::translate_unary_activation(
            &TraceOp::Sqrt,
            name,
            input_tensors,
            output_tensor,
        );
    }
    if exponent == -1.0 {
        // x^(-1) = 1/x → Reciprocal.
        return Ok(NodeOutput::one(simple_spec(
            name,
            LayerType::Reciprocal,
            vec![data_input],
            output_tensor,
            HashMap::new(),
        )));
    }
    if exponent == -0.5 {
        // x^(-0.5) = 1/sqrt(x) → Sqrt + Reciprocal.
        let sqrt_name = format!("{name}_sqrt");
        let sqrt_out = format!("{sqrt_name}_out");
        let sqrt_spec = simple_spec(
            &sqrt_name,
            LayerType::Sqrt,
            vec![data_input],
            &sqrt_out,
            HashMap::new(),
        );
        let recip_spec = simple_spec(
            name,
            LayerType::Reciprocal,
            vec![sqrt_out],
            output_tensor,
            HashMap::new(),
        );
        return Ok(NodeOutput {
            specs: vec![sqrt_spec, recip_spec],
        });
    }

    Err(NyError::UnsupportedOp(format!(
        "Powf exponent {exponent} has no sound lowering outside the supported special cases"
    )))
}

/// Translate `TraceOp::Atan2` — binary `LayerType::Atan2` spec.
///
/// Mirrors NN's `translate_binary` arm `TraceOp::Atan2 => LayerType::Atan2`
/// (Kokoro STFT polar-to-rect, NN #2218 F9). Kept local because the bridge's
/// `ops_core::translate_binary` covers only the core Add/Mul/Div/Max/Min set.
fn translate_atan2(
    name: &str,
    input_tensors: &[String],
    output_tensor: &str,
) -> Result<NodeOutput> {
    if input_tensors.len() < 2 {
        return Err(NyError::UnsupportedOp(format!(
            "Atan2 requires two inputs, got {}",
            input_tensors.len()
        )));
    }
    Ok(NodeOutput::one(simple_spec(
        name,
        LayerType::Atan2,
        input_tensors.to_vec(),
        output_tensor,
        HashMap::new(),
    )))
}

// ---------------------------------------------------------------------------
// O(N) scan decompositions (mirror NN's decompose_scan)
// ---------------------------------------------------------------------------

/// Flip → N × Slice (reverse order) + Concat.
///
/// `dim` is the raw trace dimension (for `output_shape[dim]` indexing);
/// `axis` is the trailing-relative negative encoding of `dim` (pre-encoded by
/// the caller via `super::trailing_axis`): correct in every ny-build
/// conversion regime. Mirrors NN's post-rework `translate_flip` (nn d7144ea7).
fn translate_flip(
    name: &str,
    dim: usize,
    axis: i64,
    input_tensors: &[String],
    output_tensor: &str,
    output_shape: &[usize],
) -> Result<NodeOutput> {
    let data_input = first_input(input_tensors, "Flip")?;

    if dim >= output_shape.len() {
        return Err(NyError::UnsupportedOp(format!(
            "Flip: dim {dim} >= rank {}",
            output_shape.len()
        )));
    }
    let n = output_shape[dim];
    if n == 0 {
        return Err(NyError::UnsupportedOp("Flip: dim size is 0".to_string()));
    }
    if n > MAX_DECOMPOSE_DIM {
        return Err(NyError::UnsupportedOp(format!(
            "Flip: dim size {n} exceeds decomposition limit {MAX_DECOMPOSE_DIM}"
        )));
    }

    let axis_i64 = axis;
    let mut specs = Vec::new();
    let mut slice_outs = Vec::new();

    for i in (0..n).rev() {
        let sn = format!("{name}_s{i}");
        let so = format!("{sn}_out");
        specs.push(slice_spec(
            &sn,
            vec![data_input.clone()],
            &so,
            axis_i64,
            i,
            i + 1,
        )?);
        slice_outs.push(so);
    }

    if n == 1 {
        specs.push(reshape_to_output(
            name,
            slice_outs.remove(0),
            output_tensor,
            output_shape,
        )?);
    } else {
        specs.push(concat_spec(name, slice_outs, output_tensor, axis_i64));
    }

    Ok(NodeOutput { specs })
}

/// Cumsum → N × Slice + (N−1) × Add + Concat.
///
/// Dimensions above the decomposition cap are refused. Replacing Cumsum with
/// an identity or Clamp changes its values and is unsound even when one known
/// caller happens to feed the result into a bounded activation.
fn translate_cumsum(
    name: &str,
    dim: usize,
    axis: i64,
    input_tensors: &[String],
    output_tensor: &str,
    output_shape: &[usize],
) -> Result<NodeOutput> {
    let data_input = first_input(input_tensors, "Cumsum")?;

    if dim >= output_shape.len() {
        return Err(NyError::UnsupportedOp(format!(
            "Cumsum: dim {dim} >= rank {}",
            output_shape.len()
        )));
    }
    let n = output_shape[dim];
    if n == 0 {
        return Err(NyError::UnsupportedOp("Cumsum: dim size is 0".to_string()));
    }

    if n > MAX_DECOMPOSE_DIM {
        return Err(NyError::UnsupportedOp(format!(
            "Cumsum: dim size {n} exceeds decomposition limit {MAX_DECOMPOSE_DIM}"
        )));
    }

    let axis_i64 = axis;
    let mut specs = Vec::new();
    let mut cumsum_outs: Vec<String> = Vec::new();

    for i in 0..n {
        let sn = format!("{name}_n{i}");
        let so = format!("{sn}_out");
        specs.push(slice_spec(
            &sn,
            vec![data_input.clone()],
            &so,
            axis_i64,
            i,
            i + 1,
        )?);

        if i == 0 {
            cumsum_outs.push(so);
        } else {
            let an = format!("{name}_a{i}");
            let ao = format!("{an}_out");
            specs.push(simple_spec(
                &an,
                LayerType::Add,
                vec![cumsum_outs[i - 1].clone(), so],
                &ao,
                HashMap::new(),
            ));
            cumsum_outs.push(ao);
        }
    }

    if n == 1 {
        specs.push(reshape_to_output(
            name,
            cumsum_outs.remove(0),
            output_tensor,
            output_shape,
        )?);
    } else {
        specs.push(concat_spec(name, cumsum_outs, output_tensor, axis_i64));
    }

    Ok(NodeOutput { specs })
}

/// Unfold → N × Slice (+ Transpose) + Reshape, Concat, final Reshape.
///
/// For input shape `[d0, ..., d_dim, ..., dN]` with `unfold(dim, size, step)`,
/// produces `[d0, ..., n_windows, ..., dN, size]` where
/// `n_windows = (d_dim - size) / step + 1`. Mirrors NN's `translate_unfold`
/// (#3094).
#[allow(clippy::too_many_arguments)]
fn translate_unfold(
    name: &str,
    dim: usize,
    axis: i64,
    size: usize,
    step: usize,
    input_tensors: &[String],
    output_tensor: &str,
    output_shape: &[usize],
    input_shape: &[i64],
) -> Result<NodeOutput> {
    let data_input = first_input(input_tensors, "Unfold")?;

    if step == 0 {
        return Err(NyError::UnsupportedOp(
            "Unfold: step must be > 0".to_string(),
        ));
    }
    if size == 0 {
        return Err(NyError::UnsupportedOp(
            "Unfold: size must be > 0".to_string(),
        ));
    }
    let rank = input_shape.len();
    if dim >= rank {
        return Err(NyError::UnsupportedOp(format!(
            "Unfold: dim {dim} >= input rank {rank}"
        )));
    }
    let dim_size = checked_i64_to_usize(input_shape[dim], &format!("Unfold input dim {dim}"))?;
    if size > dim_size {
        return Err(NyError::UnsupportedOp(format!(
            "Unfold: size ({size}) > dim size ({dim_size})"
        )));
    }
    let n_windows = (dim_size - size) / step + 1;
    if n_windows == 0 {
        return Err(NyError::UnsupportedOp("Unfold: no windows fit".to_string()));
    }
    if n_windows > MAX_DECOMPOSE_DIM {
        return Err(NyError::UnsupportedOp(format!(
            "Unfold: n_windows ({n_windows}) exceeds decomposition limit {MAX_DECOMPOSE_DIM}"
        )));
    }

    let axis_i64 = axis;
    let need_permute = dim < rank - 1;

    // After slicing along dim: same shape but dim axis has `size` elements.
    let narrow_shape: Vec<usize> = input_shape
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            if i == dim {
                Ok(size)
            } else {
                checked_i64_to_usize(v, &format!("Unfold dim {i}"))
            }
        })
        .collect::<Result<_>>()?;

    // Permute axes: move dim to end [0, ..., dim-1, dim+1, ..., rank-1, dim].
    let perm_axes: Vec<usize> = if need_permute {
        let mut a: Vec<usize> = (0..rank).filter(|&x| x != dim).collect();
        a.push(dim);
        a
    } else {
        (0..rank).collect()
    };
    let perm_shape: Vec<usize> = perm_axes.iter().map(|&a| narrow_shape[a]).collect();

    // Window shape: insert 1 at position `dim` (window count axis).
    let mut window_shape = perm_shape;
    window_shape.insert(dim, 1);

    // Transpose perm VERBATIM in trace-rank convention: ny-build's
    // convert_transpose applies perms as-is and the runtime TransposeLayer
    // validates them against the actual rank, so the historic `+1`
    // pretend-batched perm (values 1..=rank on a rank-`rank` tensor) was
    // never a valid permutation. Mirrors NN's post-rework `translate_unfold`
    // (nn d7144ea7).
    let perm_i64: Vec<i64> = perm_axes
        .iter()
        .map(|&a| dim_as_i64(a, "Unfold perm"))
        .collect::<Result<_>>()?;

    // Concat axis trailing-relative w.r.t. the window tensors' rank
    // (`rank + 1`: the per-window Reshape inserts a size-1 window-count axis
    // at position `dim`).
    let cat_onnx_axis = dim as i64 - (rank as i64 + 1);

    let mut specs = Vec::new();
    let mut window_outs = Vec::new();

    for w in 0..n_windows {
        let start = w * step;
        let end = start + size;

        // Slice: extract [start, end) along dim.
        let sn = format!("{name}_w{w}_sl");
        let so = format!("{sn}_out");
        specs.push(slice_spec(
            &sn,
            vec![data_input.clone()],
            &so,
            axis_i64,
            start,
            end,
        )?);

        // Transpose (if needed): move dim to last position.
        let prev_out = if need_permute {
            let tn = format!("{name}_w{w}_tr");
            let to = format!("{tn}_out");
            let mut tr_attrs = HashMap::new();
            tr_attrs.insert("perm".to_string(), AttributeValue::Ints(perm_i64.clone()));
            specs.push(simple_spec(
                &tn,
                LayerType::Transpose,
                vec![so],
                &to,
                tr_attrs,
            ));
            to
        } else {
            so
        };

        // Reshape: insert a size-1 window dim at position `dim`.
        let rn = format!("{name}_w{w}_rs");
        let ro = format!("{rn}_out");
        specs.push(reshape_to_output(&rn, prev_out, &ro, &window_shape)?);
        window_outs.push(ro);
    }

    // Concat all windows, then reshape to final output.
    if n_windows == 1 {
        specs.push(reshape_to_output(
            name,
            window_outs.remove(0),
            output_tensor,
            output_shape,
        )?);
    } else {
        // Concat produces [..., n_windows, ..., size] but may not exactly
        // match output_shape if the permute reordered axes. Final reshape
        // ensures correct shape.
        let cn = format!("{name}_cat");
        let co = format!("{cn}_out");
        specs.push(concat_spec(&cn, window_outs, &co, cat_onnx_axis));
        specs.push(reshape_to_output(name, co, output_tensor, output_shape)?);
    }

    Ok(NodeOutput { specs })
}

/// SliceSet → up to 3 × Slice + Concat.
///
/// `SliceSet { dim, start }` writes `src` (`input[1]`) into `self`
/// (`input[0]`) at `[start..start+src_len]` along `dim`; the output has the
/// shape of `self`. Decomposes into before-slice / src / after-slice
/// concatenated along `dim`. Mirrors NN's `translate_slice_set` (#3094).
#[allow(clippy::too_many_arguments)]
fn translate_slice_set(
    name: &str,
    dim: usize,
    axis: i64,
    start: usize,
    input_tensors: &[String],
    output_tensor: &str,
    output_shape: &[usize],
    input_shape: &[i64],
    src_shape: &[i64],
) -> Result<NodeOutput> {
    if input_tensors.len() < 2 {
        return Err(NyError::InternalError(
            "SliceSet requires 2 inputs (self, src)".to_string(),
        ));
    }
    let self_input = input_tensors[0].clone();
    let src_input = input_tensors[1].clone();

    let rank = input_shape.len();
    if dim >= rank {
        return Err(NyError::UnsupportedOp(format!(
            "SliceSet: dim {dim} >= rank {rank}"
        )));
    }
    let dim_size = checked_i64_to_usize(input_shape[dim], &format!("SliceSet self dim {dim}"))?;

    // Determine src length along dim from src_shape.
    if dim >= src_shape.len() {
        return Err(NyError::UnsupportedOp(format!(
            "SliceSet: dim {dim} >= src rank {}",
            src_shape.len()
        )));
    }
    let src_len = checked_i64_to_usize(src_shape[dim], &format!("SliceSet src dim {dim}"))?;
    let end = start.checked_add(src_len).ok_or_else(|| {
        NyError::InternalError(format!(
            "SliceSet: start ({start}) + src_len ({src_len}) overflows"
        ))
    })?;
    if end > dim_size {
        return Err(NyError::UnsupportedOp(format!(
            "SliceSet: start ({start}) + src_len ({src_len}) = {end} > dim_size ({dim_size})"
        )));
    }

    let axis_i64 = axis;
    let mut specs = Vec::new();
    let mut concat_inputs = Vec::new();

    // Before part: self[0..start] along dim.
    if start > 0 {
        let bn = format!("{name}_before");
        let bo = format!("{bn}_out");
        specs.push(slice_spec(
            &bn,
            vec![self_input.clone()],
            &bo,
            axis_i64,
            0,
            start,
        )?);
        concat_inputs.push(bo);
    }

    // Middle part: src replaces self[start..end].
    concat_inputs.push(src_input);

    // After part: self[end..dim_size] along dim.
    if end < dim_size {
        let an = format!("{name}_after");
        let ao = format!("{an}_out");
        specs.push(slice_spec(
            &an,
            vec![self_input],
            &ao,
            axis_i64,
            end,
            dim_size,
        )?);
        concat_inputs.push(ao);
    }

    // If only one piece (full replacement), just reshape to output.
    if concat_inputs.len() == 1 {
        specs.push(reshape_to_output(
            name,
            concat_inputs.remove(0),
            output_tensor,
            output_shape,
        )?);
    } else {
        specs.push(concat_spec(name, concat_inputs, output_tensor, axis_i64));
    }

    Ok(NodeOutput { specs })
}

// ---------------------------------------------------------------------------
// Constant / mask / gather emissions (mirror NN's dispatch_extended)
// ---------------------------------------------------------------------------

/// Translate `TraceOp::Arange` to a constant weight tensor.
///
/// Arange is data-independent: `[start, start+step, start+2*step, ...]` is
/// fully determined at trace time. Emit as a constant weight so downstream
/// ops treat it as a fixed tensor, not a variable input. Mirrors NN's
/// `translate_arange` (#2271).
fn translate_arange(
    name: &str,
    start: f64,
    end: f64,
    step: f64,
    output_shape: &[usize],
    output_tensor: &str,
    ctx: &mut Ctx,
) -> Result<NodeOutput> {
    if !start.is_finite() || !end.is_finite() || !step.is_finite() {
        return Err(NyError::ModelLoad(format!(
            "Arange start/end/step must be finite, got start={start}, end={end}, step={step}"
        )));
    }
    if step == 0.0 {
        return Err(NyError::UnsupportedOp(
            "Arange with step=0 is undefined".to_string(),
        ));
    }
    if (step > 0.0 && start > end) || (step < 0.0 && start < end) {
        return Err(NyError::ModelLoad(format!(
            "Arange step {step} points away from end {end} starting at {start}"
        )));
    }

    let ratio = (end - start) / step;
    let count_f64 = ratio.ceil().max(0.0);
    if !count_f64.is_finite() {
        return Err(NyError::ModelLoad(format!(
            "Arange element count is non-finite for start={start}, end={end}, step={step}"
        )));
    }
    if count_f64 > MAX_ARANGE_ELEMENTS as f64 {
        return Err(NyError::ModelLoad(format!(
            "Arange requires {count_f64} elements, exceeding the {MAX_ARANGE_ELEMENTS}-element limit"
        )));
    }
    let count = count_f64 as usize;

    if output_shape.len() != 1 {
        return Err(NyError::ModelLoad(format!(
            "Arange must declare a rank-1 output shape, got {output_shape:?}"
        )));
    }
    let declared_elements = checked_shape_product(output_shape).ok_or_else(|| {
        NyError::ModelLoad(format!(
            "Arange output shape {output_shape:?} has a dimension product that overflows usize"
        ))
    })?;
    if declared_elements != count {
        return Err(NyError::ModelLoad(format!(
            "Arange declared output shape {output_shape:?} has {declared_elements} elements but parameters produce {count}"
        )));
    }

    let mut data = Vec::new();
    data.try_reserve_exact(count).map_err(|error| {
        NyError::ModelLoad(format!(
            "Arange allocation failed for {count} elements: {error}"
        ))
    })?;
    for index in 0..count {
        let val = start + (index as f64) * step;
        let val_f32 = checked_f64_to_f32(val, "Arange element")?;
        data.push(val_f32);
    }

    let arr = ArrayD::from_shape_vec(IxDyn(output_shape), data)
        .map_err(|e| NyError::InternalError(format!("{name} Arange shape mismatch: {e}")))?;
    ctx.insert_weight(output_tensor, arr)?;
    Ok(NodeOutput::none())
}

/// Translate `TraceOp::Triu` / `TraceOp::Tril` to NY triangular mask layers.
///
/// The graph-build backend's `convert_triangular` generates a binary mask from
/// the input shape and diagonal offset, producing a `MulConstantLayer`. The
/// LayerSpec just needs the `diagonal` attribute. Mirrors NN's
/// `translate_triangular` (#2271).
fn translate_triangular(
    name: &str,
    diagonal: i64,
    lower: bool,
    input_tensors: &[String],
    output_tensor: &str,
) -> Result<NodeOutput> {
    let layer_type = if lower {
        LayerType::Tril
    } else {
        LayerType::Triu
    };
    let mut attrs = HashMap::new();
    attrs.insert("diagonal".to_string(), AttributeValue::Int(diagonal));
    Ok(NodeOutput::one(simple_spec(
        name,
        layer_type,
        input_tensors.to_vec(),
        output_tensor,
        attrs,
    )))
}

/// Translate `TraceOp::IndexSelect` / `TraceOp::Gather` to `LayerType::Gather`
/// with a TRAILING-RELATIVE (negative) `axis` attribute w.r.t. the DATA
/// tensor's rank (ONNX Gather axes index the data input).
///
/// Mirrors NN's `IndexSelect | Gather` arm. NN emits unconditionally and lets
/// the graph build reject a missing index tensor; the bridge refuses < 2-input
/// nodes here (fail-closed — a 1-input gather has no index tensor and cannot
/// be lowered, so translation must not succeed).
///
/// Axis-audit note (consolidation pass): the historic pretend-batched `+1`
/// encoding was WRONG under ny-build's recorded-rank convention for every
/// exercised configuration — the data input is always a recorded tensor, so
/// `dim + 1` either fails the range check (`dim == rank - 1`) or silently
/// selects dimension `dim + 1` (interior dims; e.g. an embedding-table
/// lookup `dim=0` on a rank-2 table would gather along the embedding axis).
/// It was latent only because no suite drives a 2-input Gather to
/// propagation. Trailing-relative negative resolves to `dim` in every
/// ny-build conversion regime; the `axis == 0 && constant-data` embedding
/// special case in convert_gather is unaffected (it triggers only on
/// ONNX-front-end positive-0 emissions).
fn translate_gather(
    op_desc: &str,
    name: &str,
    dim: usize,
    input_tensors: &[String],
    output_tensor: &str,
    ctx: &Ctx,
) -> Result<NodeOutput> {
    if input_tensors.len() < 2 {
        return Err(NyError::UnsupportedOp(format!(
            "{op_desc} requires 2 inputs (data, indices), got {} — index tensor missing",
            input_tensors.len()
        )));
    }
    let data_rank = lookup_tensor_shape(ctx, &input_tensors[0], op_desc)?.len();
    let axis = super::trailing_axis(dim, data_rank, &format!("{op_desc} axis"))?;
    let mut attrs = HashMap::new();
    attrs.insert("axis".to_string(), AttributeValue::Int(axis));
    Ok(NodeOutput::one(simple_spec(
        name,
        LayerType::Gather,
        input_tensors.to_vec(),
        output_tensor,
        attrs,
    )))
}

// ---------------------------------------------------------------------------
// Local spec helpers (mirror NN's decompose_scan/decompose module-privates;
// not in the bridge's shared mod.rs — dedupe later if another family needs
// them)
// ---------------------------------------------------------------------------

/// Convert `i64` to `usize`, rejecting negatives/overflow. NN keeps this in
/// `graph_tensor`; the bridge has no shared equivalent yet.
fn checked_i64_to_usize(val: i64, context: &str) -> Result<usize> {
    usize::try_from(val)
        .map_err(|_| NyError::InternalError(format!("{context}: dimension {val} is negative")))
}

fn slice_spec(
    name: &str,
    inputs: Vec<String>,
    output: &str,
    axis: i64,
    start: usize,
    end: usize,
) -> Result<LayerSpec> {
    let mut attrs = HashMap::new();
    attrs.insert("axis".to_string(), AttributeValue::Int(axis));
    attrs.insert(
        "start".to_string(),
        AttributeValue::Int(dim_as_i64(start, "slice start")?),
    );
    attrs.insert(
        "end".to_string(),
        AttributeValue::Int(dim_as_i64(end, "slice end")?),
    );
    Ok(simple_spec(name, LayerType::Slice, inputs, output, attrs))
}

fn concat_spec(name: &str, inputs: Vec<String>, output: &str, axis: i64) -> LayerSpec {
    let mut attrs = HashMap::new();
    attrs.insert("axis".to_string(), AttributeValue::Int(axis));
    simple_spec(name, LayerType::Concat, inputs, output, attrs)
}

fn reshape_to_output(
    name: &str,
    input: String,
    output: &str,
    shape: &[usize],
) -> Result<LayerSpec> {
    let mut attrs = HashMap::new();
    attrs.insert(
        "shape".to_string(),
        AttributeValue::Ints(shape_to_i64(shape, name)?),
    );
    Ok(simple_spec(
        name,
        LayerType::Reshape,
        vec![input],
        output,
        attrs,
    ))
}

/// Look up a tensor's recorded shape, mirroring NN's
/// `ctx.tensor_shapes.get(...)` lookups with the same error contexts.
fn lookup_tensor_shape(ctx: &Ctx, tensor: &str, context: &str) -> Result<Vec<i64>> {
    ctx.tensor_shapes.get(tensor).cloned().ok_or_else(|| {
        NyError::InternalError(format!("{context} input shape not found for {tensor}"))
    })
}

// ---------------------------------------------------------------------------
// Comparisons
// ---------------------------------------------------------------------------

/// `CompareOp` -> the `compare_op` attribute string ny-build's converter
/// expects. Exhaustive: the schema enum is NY-owned, so a new variant is a
/// compile error here — never a silently substituted operator.
fn compare_op_attr(op: CompareOp) -> &'static str {
    match op {
        CompareOp::Gt => "Gt",
        CompareOp::Ge => "Ge",
        CompareOp::Lt => "Lt",
        CompareOp::Le => "Le",
        CompareOp::Eq => "Eq",
        CompareOp::Ne => "Ne",
    }
}

/// Translate `TraceOp::Compare { op, value }`: a 2-input Compare LayerSpec
/// with the scalar threshold as a constant weight and the `compare_op`
/// attribute (mirrors NN's `translate_compare_scalar`).
fn translate_compare_scalar(
    name: &str,
    op: CompareOp,
    value: f64,
    input_tensors: &[String],
    output_tensor: &str,
    ctx: &mut Ctx,
) -> Result<NodeOutput> {
    let data_input = first_input(input_tensors, "Compare")?;
    let threshold = checked_f64_to_f32(value, "Compare threshold")?;
    let const_name = format!("{name}_threshold");
    insert_scalar_constant(ctx, &const_name, threshold)?;
    let mut attrs = HashMap::new();
    attrs.insert(
        "compare_op".to_string(),
        AttributeValue::String(compare_op_attr(op).to_string()),
    );
    Ok(NodeOutput::one(simple_spec(
        name,
        LayerType::Compare,
        vec![data_input, const_name],
        output_tensor,
        attrs,
    )))
}

/// Translate `TraceOp::CompareTensor { op }`: element-wise comparison of two
/// activation inputs (mirrors NN's `translate_compare_tensor`).
fn translate_compare_tensor(
    name: &str,
    op: CompareOp,
    input_tensors: &[String],
    output_tensor: &str,
) -> Result<NodeOutput> {
    if input_tensors.len() < 2 {
        return Err(NyError::UnsupportedOp(
            "CompareTensor requires two inputs".to_string(),
        ));
    }
    let mut attrs = HashMap::new();
    attrs.insert(
        "compare_op".to_string(),
        AttributeValue::String(compare_op_attr(op).to_string()),
    );
    Ok(NodeOutput::one(simple_spec(
        name,
        LayerType::CompareTensor,
        input_tensors.to_vec(),
        output_tensor,
        attrs,
    )))
}

#[cfg(test)]
mod tests {
    use ny_build::{AttributeValue, GraphModel, LayerSpec};
    use ny_core::{LayerType, NyError};

    use crate::schema::{
        CompareOp, ComputationGraph, DType, NodeId, TraceNode, TraceOp, WeightPayload,
    };
    use crate::translate::{translate, translate_multi_input};

    fn node(id: u64, name: &str, op: TraceOp, inputs: &[u64], shape: &[usize]) -> TraceNode {
        TraceNode::new(
            NodeId(id),
            name,
            op,
            inputs.iter().map(|&i| NodeId(i)).collect(),
            shape.to_vec(),
            DType::F32,
        )
    }

    fn count(model: &GraphModel, lt: &LayerType) -> usize {
        model
            .network
            .layers
            .iter()
            .filter(|l| &l.layer_type == lt)
            .count()
    }

    fn layer<'a>(model: &'a GraphModel, name: &str) -> &'a LayerSpec {
        model
            .network
            .layers
            .iter()
            .find(|l| l.name == name)
            .unwrap_or_else(|| panic!("layer {name} not found"))
    }

    fn assert_builds(model: &GraphModel, what: &str) {
        model
            .build_graph_network(ny_build::GraphNetworkOptions::default())
            .unwrap_or_else(|e| panic!("{what}: GraphModel must build a graph network: {e:?}"));
    }

    /// SwiGlu decomposes into SiLU(gate) + Mul(silu_out, up) — NN #3557.
    #[test]
    fn swiglu_decomposes_to_silu_mul() {
        // gate = relu(x), up = x — both derive from the single input.
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[4]),
            node(1, "gate", TraceOp::Relu, &[0], &[4]),
            node(2, "sg", TraceOp::SwiGlu, &[1, 0], &[4]),
        ]);
        let model = translate(&graph).expect("SwiGlu translates");
        assert_eq!(count(&model, &LayerType::SiLU), 1, "one SiLU");
        let silu = layer(&model, "layer0_trace_2_silu");
        assert_eq!(silu.inputs, vec!["layer0_trace_1_out".to_string()]);
        let mul = layer(&model, "layer0_trace_2");
        assert_eq!(mul.layer_type, LayerType::Mul);
        assert_eq!(
            mul.inputs,
            vec![
                "layer0_trace_2_silu_out".to_string(),
                "layer0_trace_0_out".to_string()
            ]
        );
        assert_builds(&model, "SwiGlu");
    }

    /// SwiGlu with a single input is refused (needs gate + up).
    #[test]
    fn swiglu_single_input_refused() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[4]),
            node(1, "sg", TraceOp::SwiGlu, &[0], &[4]),
        ]);
        let err = translate(&graph).expect_err("1-input SwiGlu refused");
        assert!(
            matches!(err, NyError::UnsupportedOp(ref m) if m.contains("SwiGlu requires 2 inputs")),
            "got {err:?}"
        );
    }

    /// Powf special-case exponents map to NN's exact primitives.
    #[test]
    fn powf_special_cases_match_nn() {
        let build = |exponent: f64| {
            let graph = ComputationGraph::from_nodes(vec![
                node(0, "x", TraceOp::Input, &[], &[4]),
                node(1, "p", TraceOp::Powf { exponent }, &[0], &[4]),
            ]);
            translate(&graph).expect("Powf translates")
        };

        // x^0 → Clip[1,1].
        let m = build(0.0);
        let clip = layer(&m, "layer0_trace_1");
        assert_eq!(clip.layer_type, LayerType::Clip);
        assert_eq!(
            clip.attributes.get("min"),
            Some(&AttributeValue::Float(1.0))
        );
        assert_eq!(
            clip.attributes.get("max"),
            Some(&AttributeValue::Float(1.0))
        );

        // x^1 → identity (Add + 0), no dtype-cast Clip.
        let m = build(1.0);
        assert_eq!(layer(&m, "layer0_trace_1").layer_type, LayerType::Add);
        assert_eq!(count(&m, &LayerType::Clip), 0);

        // x^2 → Pow with power=2 attr and the _pow2 constant.
        let m = build(2.0);
        let pow = layer(&m, "layer0_trace_1");
        assert_eq!(pow.layer_type, LayerType::Pow);
        assert_eq!(
            pow.attributes.get("power"),
            Some(&AttributeValue::Float(2.0))
        );
        assert!(m.weights.contains_key("layer0_trace_1_pow2"));

        // x^0.5 → Sqrt.
        let m = build(0.5);
        assert_eq!(layer(&m, "layer0_trace_1").layer_type, LayerType::Sqrt);

        // x^-1 → Reciprocal.
        let m = build(-1.0);
        assert_eq!(
            layer(&m, "layer0_trace_1").layer_type,
            LayerType::Reciprocal
        );

        // x^-0.5 → Sqrt + Reciprocal.
        let m = build(-0.5);
        assert_eq!(layer(&m, "layer0_trace_1_sqrt").layer_type, LayerType::Sqrt);
        assert_eq!(
            layer(&m, "layer0_trace_1").layer_type,
            LayerType::Reciprocal
        );
        assert_builds(&m, "Powf(-0.5)");
    }

    #[test]
    fn powf_general_exponents_fail_closed() {
        for exponent in [3.0, 1.5, -3.0, f64::NAN] {
            let graph = ComputationGraph::from_nodes(vec![
                node(0, "x", TraceOp::Input, &[], &[4]),
                node(1, "p", TraceOp::Powf { exponent }, &[0], &[4]),
            ]);
            let err = translate(&graph).expect_err("general Powf must fail closed");
            assert!(
                matches!(err, NyError::UnsupportedOp(ref message)
                    if message.contains("Powf exponent")
                        && message.contains("no sound lowering")),
                "{exponent:?}: got {err:?}"
            );
        }
    }

    #[test]
    fn fract_refuses_unsound_floor_lowering_for_negative_inputs() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[1]),
            node(1, "f", TraceOp::Fract, &[0], &[1]),
        ]);
        let err = translate(&graph).expect_err("Fract must fail closed");
        assert!(
            matches!(err, NyError::UnsupportedOp(ref message)
                if message.contains("Fract") && message.contains("truncation")),
            "got {err:?}"
        );
    }

    /// Atan2 → binary LayerType::Atan2 spec (inputs y, x) — NN binary arm.
    #[test]
    fn atan2_maps_to_atan2_layer() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[4]),
            node(1, "y", TraceOp::Relu, &[0], &[4]),
            node(2, "a", TraceOp::Atan2, &[1, 0], &[4]),
        ]);
        let model = translate(&graph).expect("Atan2 translates");
        let a2 = layer(&model, "layer0_trace_2");
        assert_eq!(a2.layer_type, LayerType::Atan2);
        assert_eq!(
            a2.inputs,
            vec![
                "layer0_trace_1_out".to_string(),
                "layer0_trace_0_out".to_string()
            ]
        );
        assert_builds(&model, "Atan2");
    }

    /// Cumsum([3], dim 0) → 3 Slice + 2 Add + Concat on trailing axis -1
    /// (dim 0 of rank 1).
    #[test]
    fn cumsum_decomposes_to_slice_add_concat() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[3]),
            node(1, "c", TraceOp::Cumsum { dim: 0 }, &[0], &[3]),
        ]);
        let model = translate(&graph).expect("Cumsum translates");
        assert_eq!(count(&model, &LayerType::Slice), 3);
        // 1 input-identity Add + 2 accumulation Adds.
        assert_eq!(count(&model, &LayerType::Add), 3);
        let s0 = layer(&model, "layer0_trace_1_n0");
        assert_eq!(s0.attributes.get("axis"), Some(&AttributeValue::Int(-1)));
        assert_eq!(s0.attributes.get("start"), Some(&AttributeValue::Int(0)));
        assert_eq!(s0.attributes.get("end"), Some(&AttributeValue::Int(1)));
        let a2 = layer(&model, "layer0_trace_1_a2");
        assert_eq!(
            a2.inputs,
            vec![
                "layer0_trace_1_a1_out".to_string(),
                "layer0_trace_1_n2_out".to_string()
            ]
        );
        let cat = layer(&model, "layer0_trace_1");
        assert_eq!(cat.layer_type, LayerType::Concat);
        assert_eq!(cat.attributes.get("axis"), Some(&AttributeValue::Int(-1)));
        assert_builds(&model, "Cumsum");
    }

    #[test]
    fn cumsum_large_dim_is_refused_instead_of_substituting_identity() {
        let n = super::MAX_DECOMPOSE_DIM + 1;
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[n]),
            node(1, "c", TraceOp::Cumsum { dim: 0 }, &[0], &[n]),
        ]);
        let err = translate(&graph).expect_err("large Cumsum must fail closed");
        assert!(
            matches!(err, NyError::UnsupportedOp(ref message)
                if message.contains("Cumsum") && message.contains("decomposition limit")),
            "got {err:?}"
        );
    }

    /// Flip([3], dim 0) → 3 Slices in reverse order + Concat.
    #[test]
    fn flip_decomposes_to_reversed_slices() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[3]),
            node(1, "f", TraceOp::Flip { dim: 0 }, &[0], &[3]),
        ]);
        let model = translate(&graph).expect("Flip translates");
        assert_eq!(count(&model, &LayerType::Slice), 3);
        let cat = layer(&model, "layer0_trace_1");
        assert_eq!(cat.layer_type, LayerType::Concat);
        // Reverse order: s2, s1, s0.
        assert_eq!(
            cat.inputs,
            vec![
                "layer0_trace_1_s2_out".to_string(),
                "layer0_trace_1_s1_out".to_string(),
                "layer0_trace_1_s0_out".to_string()
            ]
        );
        let s2 = layer(&model, "layer0_trace_1_s2");
        assert_eq!(s2.attributes.get("start"), Some(&AttributeValue::Int(2)));
        assert_eq!(s2.attributes.get("end"), Some(&AttributeValue::Int(3)));
        assert_builds(&model, "Flip");
    }

    /// Shape ratios cannot recover omitted per-element repeat counts.
    #[test]
    fn repeat_interleave_refuses_missing_per_element_counts() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[2]),
            node(1, "r", TraceOp::RepeatInterleave { dim: 0 }, &[0], &[4]),
        ]);
        let err = translate(&graph).expect_err("RepeatInterleave must fail closed");
        assert!(
            matches!(err, NyError::UnsupportedOp(ref message)
                if message.contains("RepeatInterleave")
                    && message.contains("repeat counts")),
            "got {err:?}"
        );
    }

    /// Arange is emitted as a constant weight (no layer), like NN #2271.
    #[test]
    fn arange_becomes_constant_weight() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[3]),
            node(
                1,
                "ar",
                TraceOp::Arange {
                    start: 0.0,
                    end: 3.0,
                    step: 1.0,
                },
                &[],
                &[3],
            ),
            node(2, "sum", TraceOp::Add, &[0, 1], &[3]),
        ]);
        let model = translate(&graph).expect("Arange translates");
        let w = model
            .weights
            .get("layer0_trace_1_out")
            .expect("arange stored as weight");
        assert_eq!(w.as_slice().unwrap(), &[0.0, 1.0, 2.0]);
        // Only the input-identity Add and the consuming Add — no Arange layer.
        assert_eq!(count(&model, &LayerType::Add), 2);
        assert_builds(&model, "Arange");
    }

    /// Arange with step=0 is refused (undefined range).
    #[test]
    fn arange_zero_step_refused() {
        let graph = ComputationGraph::from_nodes(vec![node(
            0,
            "ar",
            TraceOp::Arange {
                start: 0.0,
                end: 3.0,
                step: 0.0,
            },
            &[],
            &[3],
        )]);
        let err = translate(&graph).expect_err("step=0 refused");
        assert!(
            matches!(err, NyError::UnsupportedOp(ref m) if m.contains("Arange with step=0")),
            "got {err:?}"
        );
    }

    #[test]
    fn arange_rejects_non_finite_or_inconsistent_parameters() {
        for (start, end, step, expected) in [
            (f64::NAN, 3.0, 1.0, "must be finite"),
            (0.0, f64::INFINITY, 1.0, "must be finite"),
            (0.0, 3.0, f64::NEG_INFINITY, "must be finite"),
            (0.0, 3.0, -1.0, "points away"),
            (3.0, 0.0, 1.0, "points away"),
        ] {
            let graph = ComputationGraph::from_nodes(vec![node(
                0,
                "ar",
                TraceOp::Arange { start, end, step },
                &[],
                &[3],
            )]);
            let err = translate(&graph).expect_err("invalid Arange must fail closed");
            assert!(
                matches!(err, NyError::ModelLoad(ref message)
                    if message.contains("Arange") && message.contains(expected)),
                "{start:?}, {end:?}, {step:?}: got {err:?}"
            );
        }
    }

    #[test]
    fn arange_validates_declared_shape_and_resource_cap() {
        let shape_mismatch = ComputationGraph::from_nodes(vec![node(
            0,
            "ar",
            TraceOp::Arange {
                start: 0.0,
                end: 3.0,
                step: 1.0,
            },
            &[],
            &[2],
        )]);
        let err = translate(&shape_mismatch).expect_err("shape mismatch must fail closed");
        assert!(
            matches!(err, NyError::ModelLoad(ref message)
                if message.contains("declared output shape")
                    && message.contains("parameters produce 3")),
            "got {err:?}"
        );

        let oversized = super::MAX_ARANGE_ELEMENTS + 1;
        let over_cap = ComputationGraph::from_nodes(vec![node(
            0,
            "ar",
            TraceOp::Arange {
                start: 0.0,
                end: oversized as f64,
                step: 1.0,
            },
            &[],
            &[oversized],
        )]);
        let err = translate(&over_cap).expect_err("oversized Arange must fail before allocation");
        assert!(
            matches!(err, NyError::ModelLoad(ref message)
                if message.contains("exceeding")
                    && message.contains("element limit")),
            "got {err:?}"
        );
    }

    #[test]
    fn arange_negative_step_translates_exactly() {
        let graph = ComputationGraph::from_nodes(vec![node(
            0,
            "ar",
            TraceOp::Arange {
                start: 3.0,
                end: 0.0,
                step: -1.0,
            },
            &[],
            &[3],
        )]);
        let model = translate(&graph).expect("negative-step Arange translates");
        assert_eq!(
            model
                .weights
                .get("layer0_trace_0_out")
                .expect("range weight")
                .as_slice()
                .expect("contiguous"),
            &[3.0, 2.0, 1.0]
        );
    }

    /// Triu / Tril map to the triangular-mask layers with the diagonal attr.
    #[test]
    fn triu_tril_carry_diagonal() {
        for (op, lt, diag) in [
            (TraceOp::Triu { diagonal: 1 }, LayerType::Triu, 1),
            (TraceOp::Tril { diagonal: -1 }, LayerType::Tril, -1),
        ] {
            let graph = ComputationGraph::from_nodes(vec![
                node(0, "x", TraceOp::Input, &[], &[3, 3]),
                node(1, "t", op, &[0], &[3, 3]),
            ]);
            let model = translate(&graph).expect("triangular translates");
            let tri = layer(&model, "layer0_trace_1");
            assert_eq!(tri.layer_type, lt);
            assert_eq!(
                tri.attributes.get("diagonal"),
                Some(&AttributeValue::Int(diag))
            );
            assert_builds(&model, "Triu/Tril");
        }
    }

    /// SliceSet(self=[4], src=[2], dim 0, start 1) → before/after Slices +
    /// Concat(before, src, after) on ONNX axis 1.
    #[test]
    fn slice_set_decomposes_to_slices_concat() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "self", TraceOp::Input, &[], &[4]),
            node(
                1,
                "src",
                TraceOp::ConstantWeight {
                    weight: WeightPayload::f32(vec![9.0, 9.0], vec![2]),
                },
                &[],
                &[2],
            ),
            node(
                2,
                "ss",
                TraceOp::SliceSet { dim: 0, start: 1 },
                &[0, 1],
                &[4],
            ),
        ]);
        let model = translate(&graph).expect("SliceSet translates");
        let before = layer(&model, "layer0_trace_2_before");
        assert_eq!(
            before.attributes.get("start"),
            Some(&AttributeValue::Int(0))
        );
        assert_eq!(before.attributes.get("end"), Some(&AttributeValue::Int(1)));
        let after = layer(&model, "layer0_trace_2_after");
        assert_eq!(after.attributes.get("start"), Some(&AttributeValue::Int(3)));
        assert_eq!(after.attributes.get("end"), Some(&AttributeValue::Int(4)));
        let cat = layer(&model, "layer0_trace_2");
        assert_eq!(cat.layer_type, LayerType::Concat);
        assert_eq!(
            cat.inputs,
            vec![
                "layer0_trace_2_before_out".to_string(),
                "layer0_trace_1_out".to_string(),
                "layer0_trace_2_after_out".to_string()
            ]
        );
        assert_builds(&model, "SliceSet");
    }

    /// SliceSet with two variable inputs works through the multi-input entry
    /// (mirrors NN's test fixture shape).
    #[test]
    fn slice_set_multi_input_builds() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "self", TraceOp::Input, &[], &[2, 6]),
            node(1, "src", TraceOp::Input, &[], &[2, 2]),
            node(
                2,
                "ss",
                TraceOp::SliceSet { dim: 1, start: 2 },
                &[0, 1],
                &[2, 6],
            ),
        ]);
        let translation = translate_multi_input(&graph).expect("multi-input SliceSet translates");
        assert_builds(&translation.model, "SliceSet multi-input");
    }

    /// Unfold([4], dim 0, size 2, step 1) → 3 windows of Slice + Reshape,
    /// Concat, final Reshape to [3, 2].
    #[test]
    fn unfold_decomposes_to_windows() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[4]),
            node(
                1,
                "u",
                TraceOp::Unfold {
                    dim: 0,
                    size: 2,
                    step: 1,
                },
                &[0],
                &[3, 2],
            ),
        ]);
        let model = translate(&graph).expect("Unfold translates");
        assert_eq!(count(&model, &LayerType::Slice), 3, "one Slice per window");
        // No permute needed for the last axis: 3 window Reshapes + final.
        assert_eq!(count(&model, &LayerType::Transpose), 0);
        assert_eq!(count(&model, &LayerType::Reshape), 4);
        let w1 = layer(&model, "layer0_trace_1_w1_sl");
        assert_eq!(w1.attributes.get("start"), Some(&AttributeValue::Int(1)));
        assert_eq!(w1.attributes.get("end"), Some(&AttributeValue::Int(3)));
        let cat = layer(&model, "layer0_trace_1_cat");
        // Trailing-relative vs the rank-2 window tensors: 0 - (1 + 1) = -2.
        assert_eq!(cat.attributes.get("axis"), Some(&AttributeValue::Int(-2)));
        let final_rs = layer(&model, "layer0_trace_1");
        assert_eq!(
            final_rs.attributes.get("shape"),
            Some(&AttributeValue::Ints(vec![3, 2]))
        );
        assert_builds(&model, "Unfold");
    }

    /// IndexSelect / Gather map to LayerType::Gather with a trailing-relative
    /// axis vs the data tensor's rank.
    #[test]
    fn gather_and_index_select_map_to_gather_layer() {
        for op in [TraceOp::IndexSelect { dim: 0 }, TraceOp::Gather { dim: 0 }] {
            let graph = ComputationGraph::from_nodes(vec![
                node(0, "x", TraceOp::Input, &[], &[4]),
                node(
                    1,
                    "idx",
                    TraceOp::ConstantWeight {
                        weight: WeightPayload::f32(vec![0.0, 2.0], vec![2]),
                    },
                    &[],
                    &[2],
                ),
                node(2, "g", op, &[0, 1], &[2]),
            ]);
            let model = translate(&graph).expect("gather translates");
            let g = layer(&model, "layer0_trace_2");
            assert_eq!(g.layer_type, LayerType::Gather);
            // Trailing-relative: trace dim 0 of the rank-1 data → -1.
            assert_eq!(g.attributes.get("axis"), Some(&AttributeValue::Int(-1)));
            assert_eq!(
                g.inputs,
                vec![
                    "layer0_trace_0_out".to_string(),
                    "layer0_trace_1_out".to_string()
                ]
            );
            assert_builds(&model, "Gather/IndexSelect");
        }
    }

    /// A 1-input Gather/IndexSelect has no index tensor: refused fail-closed
    /// (NN defers to a graph-build failure; translation must not succeed).
    #[test]
    fn gather_without_indices_refused() {
        for (op, opname) in [
            (TraceOp::IndexSelect { dim: 0 }, "IndexSelect"),
            (TraceOp::Gather { dim: 0 }, "Gather"),
        ] {
            let graph = ComputationGraph::from_nodes(vec![
                node(0, "x", TraceOp::Input, &[], &[4]),
                node(1, "g", op, &[0], &[4]),
            ]);
            let err = translate(&graph).expect_err("1-input gather refused");
            assert!(
                matches!(err, NyError::UnsupportedOp(ref m) if m.contains(opname)),
                "error must name {opname}, got {err:?}"
            );
        }
    }

    /// The deliberately-refused misc ops keep the exact fail-closed
    /// UnsupportedOp refusal (coverage-taxonomy Unsupported / vacuous-only
    /// NN lowerings — see module docs).
    #[test]
    fn refused_misc_ops_stay_refused() {
        let cases: Vec<(TraceOp, &str)> = vec![
            (
                TraceOp::Roll {
                    shifts: vec![1],
                    dims: vec![0],
                },
                "Roll",
            ),
            (TraceOp::WhereCond, "WhereCond"),
            (TraceOp::ScatterAdd { dim: 0 }, "ScatterAdd"),
            (TraceOp::IndexAdd { dim: 0 }, "IndexAdd"),
            (TraceOp::IndexPut { dim: 0 }, "IndexPut"),
            (
                TraceOp::MoeGating {
                    num_experts: 4,
                    top_k: 2,
                },
                "MoeGating",
            ),
        ];
        for (op, opname) in cases {
            let graph = ComputationGraph::from_nodes(vec![
                node(0, "x", TraceOp::Input, &[], &[4]),
                node(1, "r", op, &[0], &[4]),
            ]);
            let err = translate(&graph).expect_err("refused op must error");
            match err {
                NyError::UnsupportedOp(m) => assert_eq!(
                    m,
                    format!("{opname} not supported in NY trace translation"),
                    "refusal message must keep the catch-all shape"
                ),
                other => panic!("expected UnsupportedOp for {opname}, got {other:?}"),
            }
        }
    }
    /// Compare lowers to a 2-input Compare layer: data + scalar-threshold
    /// constant, with the `compare_op` attribute (NN #2271 parity).
    #[test]
    fn compare_scalar_lowers_to_compare_layer() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[4]),
            node(
                1,
                "c",
                TraceOp::Compare {
                    op: CompareOp::Gt,
                    value: 0.5,
                },
                &[0],
                &[4],
            ),
        ]);
        let model = translate(&graph).expect("Compare translates");
        let cmp = layer(&model, "layer0_trace_1");
        assert_eq!(cmp.layer_type, LayerType::Compare);
        assert_eq!(
            cmp.inputs,
            vec![
                "layer0_trace_0_out".to_string(),
                "layer0_trace_1_threshold".to_string()
            ]
        );
        assert_eq!(
            cmp.attributes.get("compare_op"),
            Some(&AttributeValue::String("Gt".to_string()))
        );
        assert_builds(&model, "Compare");
    }

    /// CompareTensor lowers to a 2-activation-input CompareTensor layer with
    /// the `compare_op` attribute (NN #2271 parity).
    #[test]
    fn compare_tensor_lowers_to_compare_tensor_layer() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[4]),
            node(1, "y", TraceOp::Relu, &[0], &[4]),
            node(
                2,
                "c",
                TraceOp::CompareTensor { op: CompareOp::Le },
                &[0, 1],
                &[4],
            ),
        ]);
        let model = translate(&graph).expect("CompareTensor translates");
        let cmp = layer(&model, "layer0_trace_2");
        assert_eq!(cmp.layer_type, LayerType::CompareTensor);
        assert_eq!(
            cmp.attributes.get("compare_op"),
            Some(&AttributeValue::String("Le".to_string()))
        );
        assert_builds(&model, "CompareTensor");
    }
}
