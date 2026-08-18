// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Padding / resampling family: `ReflectionPad1d`, `ReflectionPad2d`,
//! `ConstantPadNd`, `PixelShuffle`, `PixelUnshuffle`, `Upsample1d`,
//! `Upsample2d`, `ResizeBilinear`, `GridSample`.
//!
//! Ported from NN's trace-path ground truth:
//! `trace_to_graph_layerspec_pad.rs` (ReflectionPad1d, ConstantPadNd),
//! `trace_to_graph_layerspec_decompose.rs` (PixelShuffle, PixelUnshuffle,
//! Upsample2d), `trace_to_graph_layerspec_shape.rs`
//! (Upsample1d), and the `translate_conservative_passthrough` arm of
//! `trace_to_graph_layerspec_dispatch_extended.rs` (GridSample).
//!
//! ## Decompositions (faithful to NN's emission)
//!
//! - `ReflectionPad1d` → one single-element `Slice` per reflected position +
//!   `Concat` along the last axis (ONNX-adjusted).
//! - `ConstantPadNd` → a `LayerType::Pad` (mode `constant`) with ONNX
//!   `pads = [before_0.., after_0..]` at trace rank. A private identity alias
//!   carries the unbatched input-shape compatibility metadata without
//!   overwriting a source tensor that may fan out to other consumers.
//! - `PixelShuffle` / `PixelUnshuffle` → `Reshape` + `Transpose` + `Reshape`
//!   with perms `[0,1,4,2,5,3]` / `[0,1,3,5,2,4]`.
//! - `Upsample1d` → `Reshape` + `Tile(axis=-1)` + `Reshape` (nearest only).
//! - `Upsample2d` (nearest only) → per-axis `Reshape` + `Tile` + `Reshape`,
//!   H pass then W pass (6 specs).
//! ## Still refused (fail-closed)
//!
//! `ReflectionPad2d` has **no** translator arm in NN: it falls into NN's
//! catch-all, which emits an opaque-skip layer with vacuous `[-inf, +inf]`
//! bounds. The bridge's sound-by-construction contract forbids vacuous layers
//! (see `mod.rs`), so `ReflectionPad2d` keeps the explicit `UnsupportedOp`
//! refusal here until a real Slice+Concat lowering is ported. `ResizeBilinear`
//! and `GridSample` are also refused: their former Tile/identity surrogates
//! preserved per-element input intervals, not the global convex hull required
//! for resampling, and could under-approximate reachable outputs.

use std::collections::HashMap;

use ny_build::{AttributeValue, LayerSpec};
use ny_core::{LayerType, NyError, Result};

use crate::schema::{TraceNode, TraceOp, TraceUpsampleMode};

use super::{
    checked_f64_to_f32, dim_as_i64, first_input, op_name, ops_core, shape_to_i64, simple_spec, Ctx,
    NodeOutput,
};

/// Maximum total pad size for decomposition (limits graph explosion).
///
/// Mirrors NN's `MAX_PAD_SIZE` in `trace_to_graph_layerspec_pad.rs`.
const MAX_PAD_SIZE: usize = 256;

/// Translate a pad/resample-family op node.
///
/// Dispatches to the faithful per-op port; `ReflectionPad2d` (and any
/// non-family op mis-routed here) refuses with the same
/// [`NyError::UnsupportedOp`] error shape the pre-split catch-all produced.
pub(super) fn translate_pad_resample(
    node: &TraceNode,
    name: &str,
    input_tensors: &[String],
    output_tensor: &str,
    _node_names: &HashMap<u64, String>,
    ctx: &mut Ctx,
) -> Result<NodeOutput> {
    let output_shape = &node.output_shape;
    match &node.op {
        TraceOp::ReflectionPad1d {
            pad_left,
            pad_right,
        } => translate_reflection_pad_1d(
            name,
            *pad_left,
            *pad_right,
            input_tensors,
            output_tensor,
            output_shape,
            ctx,
        ),
        TraceOp::ConstantPadNd { padding, value } => translate_constant_pad_nd(
            name,
            padding,
            *value,
            input_tensors,
            output_tensor,
            output_shape,
            ctx,
        ),
        TraceOp::PixelShuffle { upscale_factor } => translate_pixel_shuffle(
            name,
            *upscale_factor,
            input_tensors,
            output_tensor,
            output_shape,
        ),
        TraceOp::PixelUnshuffle { downscale_factor } => translate_pixel_unshuffle(
            name,
            *downscale_factor,
            input_tensors,
            output_tensor,
            output_shape,
        ),
        TraceOp::Upsample1d { factor } => {
            translate_upsample1d(name, *factor, input_tensors, output_tensor, output_shape)
        }
        TraceOp::Upsample2d {
            mode,
            scale_h,
            scale_w,
        } => translate_upsample2d(
            name,
            *mode,
            *scale_h,
            *scale_w,
            input_tensors,
            output_tensor,
            output_shape,
        ),
        TraceOp::ResizeBilinear { .. } => Err(NyError::UnsupportedOp(
            "ResizeBilinear: no sound per-element interval lowering is available".to_string(),
        )),
        TraceOp::GridSample { .. } => Err(NyError::UnsupportedOp(
            "GridSample: no sound per-element interval lowering is available".to_string(),
        )),
        // ReflectionPad2d (and any mis-routed op): fail-closed refusal. See
        // the module docs — NN's only handling is a vacuous opaque-skip
        // catch-all, which the bridge must not reproduce.
        other => Err(NyError::UnsupportedOp(format!(
            "{} not supported in NY trace translation",
            op_name(other)
        ))),
    }
}

// ---------------------------------------------------------------------------
// ReflectionPad1d (port of NN translate_reflection_pad_1d)
// ---------------------------------------------------------------------------

/// Translate `ReflectionPad1d { pad_left, pad_right }` → Slice + Concat.
///
/// Reflection padding reflects values at the boundary (excluding the boundary
/// element itself). For `[a, b, c, d, e]` with pad_left=2, pad_right=1:
/// result = `[c, b, a, b, c, d, e, d]`. Each reflected position becomes a
/// single-element Slice of the original tensor, concatenated with the
/// original along the trailing-relative last axis (`-1`).
fn translate_reflection_pad_1d(
    name: &str,
    pad_left: usize,
    pad_right: usize,
    input_tensors: &[String],
    output_tensor: &str,
    output_shape: &[usize],
    ctx: &mut Ctx,
) -> Result<NodeOutput> {
    let data_input = first_input(input_tensors, "ReflectionPad1d")?;

    if output_shape.is_empty() {
        return Err(NyError::UnsupportedOp(
            "ReflectionPad1d: output shape is empty".to_string(),
        ));
    }

    let total_pad = pad_left.checked_add(pad_right).ok_or_else(|| {
        NyError::InternalError("ReflectionPad1d: pad_left + pad_right overflows usize".to_string())
    })?;
    if total_pad == 0 {
        return ops_core::translate_identity_add_zero(
            name,
            "ReflectionPad1d with zero padding",
            input_tensors,
            output_tensor,
            ctx,
        );
    }
    if total_pad > MAX_PAD_SIZE {
        return Err(NyError::UnsupportedOp(format!(
            "ReflectionPad1d: total pad {total_pad} exceeds limit {MAX_PAD_SIZE}"
        )));
    }

    let last_dim = output_shape.len() - 1;
    let input_len = output_shape[last_dim]
        .checked_sub(total_pad)
        .ok_or_else(|| {
            NyError::InternalError(format!(
                "ReflectionPad1d: output dim {} < pad {}",
                output_shape[last_dim], total_pad
            ))
        })?;
    // Axis-audit note (consolidation pass): trailing-relative encoding of the
    // last data dim — always `-1`, rank-agnostic. The historic pretend-batched
    // `+1` encoding (`last_dim + 1 == rank`) was WRONG under ny-build's
    // recorded-rank convention whenever exercised: the pad's data input is a
    // recorded node output of rank `rank`, so the emitted axis was out of
    // range in every conversion regime (unbatched: runtime Slice/Concat
    // resolution; batched-classified: the recorded-rank range check). Latent
    // only because no suite drives ReflectionPad1d to propagation.
    let axis = super::trailing_axis(last_dim, output_shape.len(), "ReflectionPad1d axis")?;

    let mut specs = Vec::new();
    let mut cat_inputs = Vec::new();

    // Left reflection: elements at indices pad_left, pad_left-1, ..., 1.
    for i in 0..pad_left {
        let idx = pad_left - i;
        let sn = format!("{name}_lp{i}");
        let so = format!("{sn}_out");
        specs.push(single_slice(&sn, &data_input, &so, axis, idx)?);
        cat_inputs.push(so);
    }

    cat_inputs.push(data_input.clone());

    // Right reflection: elements at indices input_len-2, input_len-3, ...
    // (NN computes `input_len - 2 - i` unchecked; checked here so a
    // degenerate trace errors instead of panicking — same emission otherwise.)
    for i in 0..pad_right {
        let idx = input_len.checked_sub(2 + i).ok_or_else(|| {
            NyError::InternalError(format!(
                "ReflectionPad1d: pad_right {pad_right} too large for input length {input_len}"
            ))
        })?;
        let sn = format!("{name}_rp{i}");
        let so = format!("{sn}_out");
        specs.push(single_slice(&sn, &data_input, &so, axis, idx)?);
        cat_inputs.push(so);
    }

    let mut attrs = HashMap::new();
    attrs.insert("axis".to_string(), AttributeValue::Int(axis));
    specs.push(simple_spec(
        name,
        LayerType::Concat,
        cat_inputs,
        output_tensor,
        attrs,
    ));

    Ok(NodeOutput { specs })
}

// ---------------------------------------------------------------------------
// ConstantPadNd (port of NN translate_constant_pad_nd)
// ---------------------------------------------------------------------------

/// Translate `ConstantPadNd { padding, value }` → single `LayerType::Pad`.
///
/// Padding is applied innermost-dim first: `padding = [left_last, right_last,
/// left_2nd_last, right_2nd_last, ...]` (PyTorch convention). Emits one NY
/// `Pad` LayerSpec (mode `constant`) with ONNX
/// `pads = [before_0.., after_0..]` at the traced tensor rank and the f32
/// fill value.
///
/// Trace shapes are already data-layout shapes, so the emitted spec carries an
/// internal certificate telling ny-build not to discard its leading pad pair
/// as an ONNX batch axis. Raw ONNX cannot supply this private attribute.
fn translate_constant_pad_nd(
    name: &str,
    padding: &[usize],
    value: f64,
    input_tensors: &[String],
    output_tensor: &str,
    output_shape: &[usize],
    ctx: &mut Ctx,
) -> Result<NodeOutput> {
    let data_input = first_input(input_tensors, "ConstantPadNd")?;

    if !padding.len().is_multiple_of(2) {
        return Err(NyError::ModelLoad(format!(
            "ConstantPadNd: padding length {} is not even",
            padding.len()
        )));
    }
    if output_shape.is_empty() {
        return Err(NyError::UnsupportedOp(
            "ConstantPadNd: output shape is empty".to_string(),
        ));
    }

    let val_f32 = checked_f64_to_f32(value, &format!("{name} pad value"))?;
    let num_dim_pairs = padding.len() / 2;
    let rank = output_shape.len();

    if num_dim_pairs > rank {
        return Err(NyError::ModelLoad(format!(
            "ConstantPadNd: {num_dim_pairs} dim pairs but rank is {rank}"
        )));
    }

    // Per-dim pad cap guards against pathological traces (NN applies its
    // MAX_PAD_SIZE intent per-axis for the single Pad emission).
    for pair_idx in 0..num_dim_pairs {
        let pl = padding[pair_idx * 2];
        let pr = padding[pair_idx * 2 + 1];
        let pair_total = pl.checked_add(pr).ok_or_else(|| {
            NyError::InternalError(format!(
                "ConstantPadNd: pair {pair_idx} total pad overflows usize"
            ))
        })?;
        if pair_total > MAX_PAD_SIZE {
            return Err(NyError::UnsupportedOp(format!(
                "ConstantPadNd: pair {pair_idx} total pad {pair_total} exceeds limit {MAX_PAD_SIZE}"
            )));
        }
    }

    // Build ONNX pads = [before_0..before_{rank-1}, after_0..after_{rank-1}]
    // at the trace rank. Dims without a PyTorch pair get (0, 0). PyTorch
    // `padding[2*pair_idx]/[2*pair_idx + 1]` maps to `dim = rank - 1 - pair_idx`.
    let mut before = vec![0_i64; rank];
    let mut after = vec![0_i64; rank];
    let mut total_pad: usize = 0;
    for pair_idx in 0..num_dim_pairs {
        let dim = rank - 1 - pair_idx;
        let pl = padding[pair_idx * 2];
        let pr = padding[pair_idx * 2 + 1];
        before[dim] = dim_as_i64(pl, "ConstantPadNd before")?;
        after[dim] = dim_as_i64(pr, "ConstantPadNd after")?;
        total_pad = total_pad
            .checked_add(pl)
            .and_then(|total| total.checked_add(pr))
            .ok_or_else(|| {
                NyError::InternalError(
                    "ConstantPadNd: aggregate padding overflows usize".to_string(),
                )
            })?;
    }

    // A no-op still needs a real producer for this trace node's output tensor.
    if total_pad == 0 {
        return ops_core::translate_identity_add_zero(
            name,
            "ConstantPadNd with zero padding",
            input_tensors,
            output_tensor,
            ctx,
        );
    }

    // ONNX pads layout: all befores then all afters, at trace rank.
    let mut pads: Vec<i64> = Vec::with_capacity(rank * 2);
    pads.extend(before);
    pads.extend(after);

    let mut attrs = HashMap::new();
    attrs.insert(
        "mode".to_string(),
        AttributeValue::String("constant".to_string()),
    );
    attrs.insert("pads".to_string(), AttributeValue::Ints(pads));
    attrs.insert("value".to_string(), AttributeValue::Float(val_f32));
    attrs.insert(
        ny_build::PAD_PRESERVE_ALL_AXES_ATTR.to_string(),
        AttributeValue::Int(1),
    );

    Ok(NodeOutput::one(simple_spec(
        name,
        LayerType::Pad,
        vec![data_input],
        output_tensor,
        attrs,
    )))
}

// ---------------------------------------------------------------------------
// PixelShuffle / PixelUnshuffle (port of NN translate_pixel_(un)shuffle)
// ---------------------------------------------------------------------------

/// PixelShuffle → Reshape + Transpose + Reshape.
///
/// `[B, C*r², H_in, W_in] → [B, C, r, r, H_in, W_in] → [B, C, H_in, r, W_in, r]
/// → [B, C, H, W]`
fn translate_pixel_shuffle(
    name: &str,
    upscale_factor: usize,
    input_tensors: &[String],
    output_tensor: &str,
    output_shape: &[usize],
) -> Result<NodeOutput> {
    let r = upscale_factor;
    if r == 0 {
        return Err(NyError::UnsupportedOp(
            "PixelShuffle: upscale factor is 0".to_string(),
        ));
    }
    if output_shape.len() != 4 {
        return Err(NyError::UnsupportedOp(format!(
            "PixelShuffle: expected rank-4 output, got rank {}",
            output_shape.len()
        )));
    }
    let [b, c, h, w] = [
        output_shape[0],
        output_shape[1],
        output_shape[2],
        output_shape[3],
    ];
    let data_input = first_input(input_tensors, "PixelShuffle")?;

    if !h.is_multiple_of(r) || !w.is_multiple_of(r) {
        return Err(NyError::ModelLoad(format!(
            "PixelShuffle: output spatial shape [{h}, {w}] is not divisible by factor {r}"
        )));
    }
    let h_in = h / r;
    let w_in = w / r;

    // Step 1: Reshape → [B, C, r, r, H_in, W_in]
    let rs1 = format!("{name}_rs1");
    let rs1_out = format!("{rs1}_out");
    let spec1 = reshape_spec(&rs1, vec![data_input], &rs1_out, &[b, c, r, r, h_in, w_in])?;

    // Step 2: Transpose [0, 1, 4, 2, 5, 3]
    let tr = format!("{name}_tr");
    let tr_out = format!("{tr}_out");
    let spec2 = transpose_spec(&tr, vec![rs1_out], &tr_out, &[0, 1, 4, 2, 5, 3]);

    // Step 3: Reshape → [B, C, H, W]
    let spec3 = reshape_spec(name, vec![tr_out], output_tensor, output_shape)?;

    Ok(NodeOutput {
        specs: vec![spec1, spec2, spec3],
    })
}

/// PixelUnshuffle → Reshape + Transpose + Reshape.
///
/// `[B, C, H*r, W*r] → [B, C, H, r, W, r] → [B, C, r, r, H, W] → [B, C*r², H, W]`
fn translate_pixel_unshuffle(
    name: &str,
    downscale_factor: usize,
    input_tensors: &[String],
    output_tensor: &str,
    output_shape: &[usize],
) -> Result<NodeOutput> {
    let r = downscale_factor;
    if r == 0 {
        return Err(NyError::UnsupportedOp(
            "PixelUnshuffle: downscale factor is 0".to_string(),
        ));
    }
    if output_shape.len() != 4 {
        return Err(NyError::UnsupportedOp(format!(
            "PixelUnshuffle: expected rank-4 output, got rank {}",
            output_shape.len()
        )));
    }
    let [b, c_out, h, w] = [
        output_shape[0],
        output_shape[1],
        output_shape[2],
        output_shape[3],
    ];
    let r_sq = r.checked_mul(r).ok_or_else(|| {
        NyError::UnsupportedOp(format!("PixelUnshuffle: factor {r} squared overflows"))
    })?;
    if !c_out.is_multiple_of(r_sq) {
        return Err(NyError::ModelLoad(format!(
            "PixelUnshuffle: output channels {c_out} are not divisible by factor squared {r_sq}"
        )));
    }
    let c = c_out / r_sq;
    let data_input = first_input(input_tensors, "PixelUnshuffle")?;

    // Step 1: Reshape → [B, C, H, r, W, r]
    let rs1 = format!("{name}_rs1");
    let rs1_out = format!("{rs1}_out");
    let spec1 = reshape_spec(&rs1, vec![data_input], &rs1_out, &[b, c, h, r, w, r])?;

    // Step 2: Transpose [0, 1, 3, 5, 2, 4]
    let tr = format!("{name}_tr");
    let tr_out = format!("{tr}_out");
    let spec2 = transpose_spec(&tr, vec![rs1_out], &tr_out, &[0, 1, 3, 5, 2, 4]);

    // Step 3: Reshape → [B, C*r², H, W]
    let spec3 = reshape_spec(name, vec![tr_out], output_tensor, output_shape)?;

    Ok(NodeOutput {
        specs: vec![spec1, spec2, spec3],
    })
}

// ---------------------------------------------------------------------------
// Upsample1d (port of NN translate_upsample1d)
// ---------------------------------------------------------------------------

/// Translate `Upsample1d { factor }` to Reshape → Tile → Reshape.
///
/// Nearest-neighbor 1D upsampling by `factor` along the last axis:
/// `[..., T]` → `[..., T, 1]` → `[..., T, factor]` → `[..., T*factor]`.
fn translate_upsample1d(
    name: &str,
    factor: usize,
    input_tensors: &[String],
    output_tensor: &str,
    output_shape: &[usize],
) -> Result<NodeOutput> {
    let data_input = first_input(input_tensors, "Upsample1d")?;

    let rank = output_shape.len();
    if rank == 0 {
        return Err(NyError::UnsupportedOp(
            "Upsample1d: output rank is 0".to_string(),
        ));
    }
    if factor == 0 {
        return Err(NyError::UnsupportedOp(
            "Upsample1d: factor is 0".to_string(),
        ));
    }

    let out_last = output_shape[rank - 1];
    if !out_last.is_multiple_of(factor) {
        return Err(NyError::UnsupportedOp(format!(
            "Upsample1d: output last dim {out_last} not divisible by factor {factor}"
        )));
    }
    let t = out_last / factor;

    // Step 1: Reshape [..., T] -> [..., T, 1]
    let reshape1_name = format!("{name}_unsq");
    let reshape1_out = format!("{reshape1_name}_out");
    let mut unsq_shape: Vec<usize> = output_shape[..rank - 1].to_vec();
    unsq_shape.push(t);
    unsq_shape.push(1);
    let spec1 = reshape_spec(&reshape1_name, vec![data_input], &reshape1_out, &unsq_shape)?;

    // Step 2: Tile [..., T, 1] -> [..., T, factor] along last axis.
    // Tile axis is passed directly (no ONNX batch-dim adjustment).
    let tile_name = format!("{name}_tile");
    let tile_out = format!("{tile_name}_out");
    let spec2 = tile_spec(&tile_name, vec![reshape1_out], &tile_out, -1, factor)?;

    // Step 3: Reshape [..., T, factor] -> [..., T*factor]
    let spec3 = reshape_spec(name, vec![tile_out], output_tensor, output_shape)?;

    Ok(NodeOutput {
        specs: vec![spec1, spec2, spec3],
    })
}

// ---------------------------------------------------------------------------
// Upsample2d (port of NN translate_upsample2d)
// ---------------------------------------------------------------------------

/// Upsample2d (nearest) → 2× (Reshape + Tile + Reshape): H pass then W pass.
fn translate_upsample2d(
    name: &str,
    mode: TraceUpsampleMode,
    scale_h: f64,
    scale_w: f64,
    input_tensors: &[String],
    output_tensor: &str,
    output_shape: &[usize],
) -> Result<NodeOutput> {
    if mode != TraceUpsampleMode::Nearest {
        // Same message shape as NN (which matches on the lowercase mode name).
        let mode_str = match mode {
            TraceUpsampleMode::Nearest => "nearest",
            TraceUpsampleMode::Bilinear => "bilinear",
            TraceUpsampleMode::Bicubic => "bicubic",
        };
        return Err(NyError::UnsupportedOp(format!(
            "Upsample2d mode '{mode_str}' not supported; only 'nearest'"
        )));
    }
    let sh = checked_f64_to_usize(scale_h, "Upsample2d scale_h")?;
    let sw = checked_f64_to_usize(scale_w, "Upsample2d scale_w")?;
    if sh == 0 || sw == 0 {
        return Err(NyError::UnsupportedOp(
            "Upsample2d: scale factor is 0".to_string(),
        ));
    }
    let data_input = first_input(input_tensors, "Upsample2d")?;
    let rank = output_shape.len();
    if rank < 2 {
        return Err(NyError::UnsupportedOp(format!(
            "Upsample2d: output rank {rank} < 2"
        )));
    }
    let h_out = output_shape[rank - 2];
    let w_out = output_shape[rank - 1];
    if !h_out.is_multiple_of(sh) || !w_out.is_multiple_of(sw) {
        return Err(NyError::ModelLoad(format!(
            "Upsample2d: output spatial shape [{h_out}, {w_out}] is not divisible by \
             scale [{sh}, {sw}]"
        )));
    }
    let h_in = h_out / sh;
    let w_in = w_out / sw;
    let prefix = &output_shape[..rank - 2];

    let mut specs = Vec::new();

    // Step 1: Upsample H — Reshape → Tile → Reshape
    let mut shape_h_unsq: Vec<usize> = prefix.to_vec();
    shape_h_unsq.extend_from_slice(&[h_in, 1, w_in]);
    let rs1 = format!("{name}_h_unsq");
    let rs1_out = format!("{rs1}_out");
    specs.push(reshape_spec(
        &rs1,
        vec![data_input],
        &rs1_out,
        &shape_h_unsq,
    )?);

    let t1 = format!("{name}_h_tile");
    let t1_out = format!("{t1}_out");
    let tile_axis_h = dim_as_i64(shape_h_unsq.len() - 2, "Upsample2d h tile")?;
    specs.push(tile_spec(&t1, vec![rs1_out], &t1_out, tile_axis_h, sh)?);

    let mut shape_h_merge: Vec<usize> = prefix.to_vec();
    shape_h_merge.extend_from_slice(&[h_out, w_in]);
    let rs2 = format!("{name}_h_merge");
    let rs2_out = format!("{rs2}_out");
    specs.push(reshape_spec(&rs2, vec![t1_out], &rs2_out, &shape_h_merge)?);

    // Step 2: Upsample W — Reshape → Tile → Reshape
    let mut shape_w_unsq: Vec<usize> = prefix.to_vec();
    shape_w_unsq.extend_from_slice(&[h_out, w_in, 1]);
    let rs3 = format!("{name}_w_unsq");
    let rs3_out = format!("{rs3}_out");
    specs.push(reshape_spec(&rs3, vec![rs2_out], &rs3_out, &shape_w_unsq)?);

    let t2 = format!("{name}_w_tile");
    let t2_out = format!("{t2}_out");
    let tile_axis_w = dim_as_i64(shape_w_unsq.len() - 1, "Upsample2d w tile")?;
    specs.push(tile_spec(&t2, vec![rs3_out], &t2_out, tile_axis_w, sw)?);

    specs.push(reshape_spec(
        name,
        vec![t2_out],
        output_tensor,
        output_shape,
    )?);

    Ok(NodeOutput { specs })
}

// ---------------------------------------------------------------------------
// Local spec helpers (mirroring NN's decompose/pad helpers; candidates for
// dedupe into mod.rs once all family ports land)
// ---------------------------------------------------------------------------

/// Emit a single-element Slice LayerSpec at position `idx` along `axis`.
fn single_slice(name: &str, input: &str, output: &str, axis: i64, idx: usize) -> Result<LayerSpec> {
    let mut attrs = HashMap::new();
    attrs.insert("axis".to_string(), AttributeValue::Int(axis));
    attrs.insert(
        "start".to_string(),
        AttributeValue::Int(dim_as_i64(idx, "pad slice start")?),
    );
    attrs.insert(
        "end".to_string(),
        AttributeValue::Int(dim_as_i64(idx + 1, "pad slice end")?),
    );
    Ok(simple_spec(
        name,
        LayerType::Slice,
        vec![input.to_string()],
        output,
        attrs,
    ))
}

/// Build a Reshape LayerSpec with a `shape` attribute.
fn reshape_spec(
    name: &str,
    inputs: Vec<String>,
    output: &str,
    shape: &[usize],
) -> Result<LayerSpec> {
    let mut attrs = HashMap::new();
    attrs.insert(
        "shape".to_string(),
        AttributeValue::Ints(shape_to_i64(shape, name)?),
    );
    Ok(simple_spec(name, LayerType::Reshape, inputs, output, attrs))
}

/// Build a Transpose LayerSpec with a `perm` attribute.
fn transpose_spec(name: &str, inputs: Vec<String>, output: &str, perm: &[i64]) -> LayerSpec {
    let mut attrs = HashMap::new();
    attrs.insert("perm".to_string(), AttributeValue::Ints(perm.to_vec()));
    simple_spec(name, LayerType::Transpose, inputs, output, attrs)
}

/// Build a Tile LayerSpec with `axis` + `reps` attributes.
fn tile_spec(
    name: &str,
    inputs: Vec<String>,
    output: &str,
    axis: i64,
    reps: usize,
) -> Result<LayerSpec> {
    let mut attrs = HashMap::new();
    attrs.insert("axis".to_string(), AttributeValue::Int(axis));
    attrs.insert(
        "reps".to_string(),
        AttributeValue::Int(dim_as_i64(reps, "tile reps")?),
    );
    Ok(simple_spec(name, LayerType::Tile, inputs, output, attrs))
}

/// Convert an `f64` scale factor to `usize`, rejecting NaN/negative/
/// non-integral values (which would silently truncate).
///
/// Mirrors NN's `checked_f64_to_usize` in `graph_tensor.rs` (copied here —
/// no equivalent shared helper in `mod.rs`; dedupe later).
fn checked_f64_to_usize(val: f64, context: &str) -> Result<usize> {
    if !val.is_finite() {
        return Err(NyError::UnsupportedOp(format!(
            "{context}: f64 value {val} is non-finite"
        )));
    }
    if val < 0.0 {
        return Err(NyError::UnsupportedOp(format!(
            "{context}: f64 value {val} is negative"
        )));
    }
    let rounded = val.round();
    if (rounded - val).abs() > 1e-6 {
        return Err(NyError::UnsupportedOp(format!(
            "{context}: f64 value {val} is not integral"
        )));
    }
    // Safe: rounded is finite, non-negative, and integral. For practical
    // tensor shapes (< 2^53), f64 → usize cast is exact.
    Ok(rounded as usize)
}

#[cfg(test)]
mod tests {
    use ny_build::{AttributeValue, GraphModel, LayerSpec};
    use ny_core::{LayerType, NyError};

    use super::super::translate;
    use crate::schema::{
        ComputationGraph, DType, GridSamplePaddingMode, NodeId, TraceNode, TraceOp,
        TraceUpsampleMode, WeightPayload,
    };

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

    fn find<'a>(model: &'a GraphModel, name: &str) -> &'a LayerSpec {
        model
            .network
            .layers
            .iter()
            .find(|l| l.name == name)
            .unwrap_or_else(|| panic!("layer '{name}' present"))
    }

    fn build_ok(model: &GraphModel, what: &str) {
        model
            .build_graph_network(ny_build::GraphNetworkOptions::default())
            .unwrap_or_else(|e| panic!("{what} GraphModel builds a graph network: {e:?}"));
    }

    /// ReflectionPad1d [a,b,c,d,e] pad(2,1) → slices at idx 2,1 (left), 3
    /// (right) + Concat, all on trailing-relative axis -1 (the last data
    /// dim, rank-agnostic; the historic pretend-batched `+1` encoding was
    /// out of range under ny-build's recorded-rank convention).
    #[test]
    fn reflection_pad1d_decomposes_to_slices_and_concat() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[5]),
            node(
                1,
                "pad",
                TraceOp::ReflectionPad1d {
                    pad_left: 2,
                    pad_right: 1,
                },
                &[0],
                &[8],
            ),
        ]);
        let model = translate(&graph).expect("reflection pad translates");

        assert_eq!(count(&model, &LayerType::Slice), 3, "3 reflected positions");
        assert_eq!(count(&model, &LayerType::Concat), 1, "one Concat");

        // Left reflections: indices pad_left - i = 2, 1.
        let lp0 = find(&model, "layer0_trace_1_lp0");
        assert_eq!(lp0.attributes.get("axis"), Some(&AttributeValue::Int(-1)));
        assert_eq!(lp0.attributes.get("start"), Some(&AttributeValue::Int(2)));
        assert_eq!(lp0.attributes.get("end"), Some(&AttributeValue::Int(3)));
        let lp1 = find(&model, "layer0_trace_1_lp1");
        assert_eq!(lp1.attributes.get("start"), Some(&AttributeValue::Int(1)));

        // Right reflection: index input_len - 2 - 0 = 3.
        let rp0 = find(&model, "layer0_trace_1_rp0");
        assert_eq!(rp0.attributes.get("start"), Some(&AttributeValue::Int(3)));
        assert_eq!(rp0.attributes.get("end"), Some(&AttributeValue::Int(4)));

        // Concat order: [lp0, lp1, original, rp0] on the same ONNX axis.
        let cat = find(&model, "layer0_trace_1");
        assert_eq!(cat.layer_type, LayerType::Concat);
        assert_eq!(cat.attributes.get("axis"), Some(&AttributeValue::Int(-1)));
        assert_eq!(
            cat.inputs,
            vec![
                "layer0_trace_1_lp0_out".to_string(),
                "layer0_trace_1_lp1_out".to_string(),
                "layer0_trace_0_out".to_string(),
                "layer0_trace_1_rp0_out".to_string(),
            ]
        );
        assert_eq!(cat.outputs, vec!["layer0_trace_1_out".to_string()]);

        build_ok(&model, "reflection pad1d");
    }

    /// ReflectionPad1d refuses a pad total beyond MAX_PAD_SIZE.
    #[test]
    fn reflection_pad1d_refuses_oversized_pad() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[500]),
            node(
                1,
                "pad",
                TraceOp::ReflectionPad1d {
                    pad_left: 200,
                    pad_right: 100,
                },
                &[0],
                &[800],
            ),
        ]);
        let err = translate(&graph).expect_err("oversized pad refused");
        assert!(
            matches!(err, NyError::UnsupportedOp(ref m) if m.contains("exceeds limit")),
            "got {err:?}"
        );
    }

    #[test]
    fn reflection_pad1d_rejects_pad_sum_overflow() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[1]),
            node(
                1,
                "pad",
                TraceOp::ReflectionPad1d {
                    pad_left: usize::MAX,
                    pad_right: 1,
                },
                &[0],
                &[1],
            ),
        ]);
        let err = translate(&graph).expect_err("overflowing pad sum refused");
        assert!(
            matches!(err, NyError::InternalError(ref m) if m.contains("overflows usize")),
            "got {err:?}"
        );
    }

    #[test]
    fn zero_reflection_pad1d_emits_identity_output() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[5]),
            node(
                1,
                "pad",
                TraceOp::ReflectionPad1d {
                    pad_left: 0,
                    pad_right: 0,
                },
                &[0],
                &[5],
            ),
        ]);
        let model = translate(&graph).expect("zero reflection pad translates");
        let pad = find(&model, "layer0_trace_1");
        assert_eq!(pad.layer_type, LayerType::Add);
        assert_eq!(pad.outputs, vec!["layer0_trace_1_out".to_string()]);
        build_ok(&model, "zero reflection pad1d");
    }

    /// ConstantPadNd emits a Pad layer with exact attributes and an internal
    /// certificate preserving every trace-native axis.
    #[test]
    fn constant_pad_nd_emits_pad_layer() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[2, 3]),
            node(
                1,
                "pad",
                TraceOp::ConstantPadNd {
                    padding: vec![1, 1],
                    value: 0.5,
                },
                &[0],
                &[2, 5],
            ),
        ]);
        let model = translate(&graph).expect("constant pad translates");

        assert_eq!(count(&model, &LayerType::Pad), 1, "one Pad layer");
        let pad = find(&model, "layer0_trace_1");
        assert_eq!(
            pad.attributes.get("mode"),
            Some(&AttributeValue::String("constant".to_string()))
        );
        // rank 2: before=[0,1], after=[0,1] → pads=[0,1,0,1].
        assert_eq!(
            pad.attributes.get("pads"),
            Some(&AttributeValue::Ints(vec![0, 1, 0, 1]))
        );
        assert_eq!(
            pad.attributes.get("value"),
            Some(&AttributeValue::Float(0.5))
        );
        assert_eq!(
            pad.attributes.get(ny_build::PAD_PRESERVE_ALL_AXES_ATTR),
            Some(&AttributeValue::Int(1))
        );
        assert_eq!(pad.inputs, vec!["layer0_trace_0_out".to_string()]);
        assert_eq!(
            model.tensor_shapes.get("layer0_trace_0_out"),
            Some(&vec![2, 3]),
            "source shape remains intact"
        );

        build_ok(&model, "constant pad nd");
    }

    #[test]
    fn zero_constant_pad_nd_emits_identity_output() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[3]),
            node(
                1,
                "pad",
                TraceOp::ConstantPadNd {
                    padding: vec![0, 0],
                    value: 7.0,
                },
                &[0],
                &[3],
            ),
        ]);
        let model = translate(&graph).expect("zero constant pad translates");
        let pad = find(&model, "layer0_trace_1");
        assert_eq!(pad.layer_type, LayerType::Add);
        assert_eq!(pad.outputs, vec!["layer0_trace_1_out".to_string()]);
        assert_eq!(count(&model, &LayerType::Pad), 0);
        build_ok(&model, "zero constant pad nd");
    }

    #[test]
    fn constant_pad_nd_preserves_fanout_source_shape() {
        let mut graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[2, 3]),
            node(
                1,
                "pad",
                TraceOp::ConstantPadNd {
                    padding: vec![1, 1],
                    value: 0.0,
                },
                &[0],
                &[2, 5],
            ),
            node(2, "relu", TraceOp::Relu, &[0], &[2, 3]),
        ]);
        graph.output_nodes = vec![NodeId(1), NodeId(2)];

        let model = translate(&graph).expect("fan-out constant pad translates");
        assert_eq!(
            model.tensor_shapes.get("layer0_trace_0_out"),
            Some(&vec![2, 3]),
            "shared source keeps its traced shape"
        );
        assert_eq!(
            find(&model, "layer0_trace_1").inputs,
            vec!["layer0_trace_0_out".to_string()]
        );
        assert!(!model
            .tensor_shapes
            .contains_key("layer0_trace_1_pad_input_out"));
        assert_eq!(
            find(&model, "layer0_trace_2").inputs,
            vec!["layer0_trace_0_out".to_string()]
        );
        build_ok(&model, "constant pad nd fan-out");
    }

    /// ConstantPadNd with an odd padding vector is a hard error.
    #[test]
    fn constant_pad_nd_rejects_odd_padding() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[4]),
            node(
                1,
                "pad",
                TraceOp::ConstantPadNd {
                    padding: vec![1, 1, 2],
                    value: 0.0,
                },
                &[0],
                &[6],
            ),
        ]);
        let err = translate(&graph).expect_err("odd padding refused");
        assert!(
            matches!(err, NyError::ModelLoad(ref m) if m.contains("not even")),
            "got {err:?}"
        );
    }

    #[test]
    fn constant_pad_nd_rejects_pair_sum_overflow() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[1]),
            node(
                1,
                "pad",
                TraceOp::ConstantPadNd {
                    padding: vec![usize::MAX, 1],
                    value: 0.0,
                },
                &[0],
                &[1],
            ),
        ]);
        let err = translate(&graph).expect_err("overflowing pad pair refused");
        assert!(
            matches!(err, NyError::InternalError(ref m) if m.contains("overflows usize")),
            "got {err:?}"
        );
    }

    /// PixelShuffle [1,4,2,2] r=2 → Reshape[1,1,2,2,2,2] + Transpose
    /// [0,1,4,2,5,3] + Reshape[1,1,4,4] (NN's exact 3-spec decomposition).
    #[test]
    fn pixel_shuffle_decomposes_reshape_transpose_reshape() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[1, 4, 2, 2]),
            node(
                1,
                "ps",
                TraceOp::PixelShuffle { upscale_factor: 2 },
                &[0],
                &[1, 1, 4, 4],
            ),
        ]);
        let model = translate(&graph).expect("pixel shuffle translates");

        assert_eq!(count(&model, &LayerType::Reshape), 2);
        assert_eq!(count(&model, &LayerType::Transpose), 1);

        let rs1 = find(&model, "layer0_trace_1_rs1");
        assert_eq!(
            rs1.attributes.get("shape"),
            Some(&AttributeValue::Ints(vec![1, 1, 2, 2, 2, 2])),
            "[B, C, r, r, H_in, W_in]"
        );
        let tr = find(&model, "layer0_trace_1_tr");
        assert_eq!(
            tr.attributes.get("perm"),
            Some(&AttributeValue::Ints(vec![0, 1, 4, 2, 5, 3]))
        );
        let rs2 = find(&model, "layer0_trace_1");
        assert_eq!(
            rs2.attributes.get("shape"),
            Some(&AttributeValue::Ints(vec![1, 1, 4, 4]))
        );
        assert_eq!(rs2.outputs, vec!["layer0_trace_1_out".to_string()]);

        build_ok(&model, "pixel shuffle");
    }

    /// PixelUnshuffle [1,1,4,4] r=2 → Reshape[1,1,2,2,2,2] + Transpose
    /// [0,1,3,5,2,4] + Reshape[1,4,2,2].
    #[test]
    fn pixel_unshuffle_decomposes_reshape_transpose_reshape() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[1, 1, 4, 4]),
            node(
                1,
                "pus",
                TraceOp::PixelUnshuffle {
                    downscale_factor: 2,
                },
                &[0],
                &[1, 4, 2, 2],
            ),
        ]);
        let model = translate(&graph).expect("pixel unshuffle translates");

        let rs1 = find(&model, "layer0_trace_1_rs1");
        assert_eq!(
            rs1.attributes.get("shape"),
            Some(&AttributeValue::Ints(vec![1, 1, 2, 2, 2, 2])),
            "[B, C, H, r, W, r]"
        );
        let tr = find(&model, "layer0_trace_1_tr");
        assert_eq!(
            tr.attributes.get("perm"),
            Some(&AttributeValue::Ints(vec![0, 1, 3, 5, 2, 4]))
        );
        let rs2 = find(&model, "layer0_trace_1");
        assert_eq!(
            rs2.attributes.get("shape"),
            Some(&AttributeValue::Ints(vec![1, 4, 2, 2]))
        );

        build_ok(&model, "pixel unshuffle");
    }

    /// PixelShuffle refuses non-rank-4 output.
    #[test]
    fn pixel_shuffle_refuses_non_rank4() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[4, 2, 2]),
            node(
                1,
                "ps",
                TraceOp::PixelShuffle { upscale_factor: 2 },
                &[0],
                &[1, 4, 4],
            ),
        ]);
        let err = translate(&graph).expect_err("rank-3 refused");
        assert!(
            matches!(err, NyError::UnsupportedOp(ref m) if m.contains("rank-4")),
            "got {err:?}"
        );
    }

    /// Upsample1d [2,3] factor 2 → Reshape[2,3,1] + Tile(axis=-1, reps=2) +
    /// Reshape[2,6].
    #[test]
    fn upsample1d_decomposes_reshape_tile_reshape() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[2, 3]),
            node(1, "up", TraceOp::Upsample1d { factor: 2 }, &[0], &[2, 6]),
        ]);
        let model = translate(&graph).expect("upsample1d translates");

        assert_eq!(count(&model, &LayerType::Reshape), 2);
        assert_eq!(count(&model, &LayerType::Tile), 1);

        let unsq = find(&model, "layer0_trace_1_unsq");
        assert_eq!(
            unsq.attributes.get("shape"),
            Some(&AttributeValue::Ints(vec![2, 3, 1]))
        );
        let tile = find(&model, "layer0_trace_1_tile");
        assert_eq!(tile.attributes.get("axis"), Some(&AttributeValue::Int(-1)));
        assert_eq!(tile.attributes.get("reps"), Some(&AttributeValue::Int(2)));
        let merge = find(&model, "layer0_trace_1");
        assert_eq!(
            merge.attributes.get("shape"),
            Some(&AttributeValue::Ints(vec![2, 6]))
        );

        build_ok(&model, "upsample1d");
    }

    /// Upsample1d refuses an indivisible output dim.
    #[test]
    fn upsample1d_refuses_indivisible_factor() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[2, 3]),
            node(1, "up", TraceOp::Upsample1d { factor: 4 }, &[0], &[2, 7]),
        ]);
        let err = translate(&graph).expect_err("indivisible refused");
        assert!(
            matches!(err, NyError::UnsupportedOp(ref m) if m.contains("not divisible")),
            "got {err:?}"
        );
    }

    /// Upsample2d nearest [1,1,2,2]×(2,2) → NN's exact 6-spec H-pass/W-pass
    /// decomposition with the right shapes and tile axes.
    #[test]
    fn upsample2d_decomposes_h_then_w_pass() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[1, 1, 2, 2]),
            node(
                1,
                "up",
                TraceOp::Upsample2d {
                    mode: TraceUpsampleMode::Nearest,
                    scale_h: 2.0,
                    scale_w: 2.0,
                },
                &[0],
                &[1, 1, 4, 4],
            ),
        ]);
        let model = translate(&graph).expect("upsample2d translates");

        assert_eq!(count(&model, &LayerType::Reshape), 4, "4 Reshapes");
        assert_eq!(count(&model, &LayerType::Tile), 2, "H tile + W tile");

        // H pass: [1,1,2,1,2] → tile axis 3 (len-2) → merge [1,1,4,2].
        let h_unsq = find(&model, "layer0_trace_1_h_unsq");
        assert_eq!(
            h_unsq.attributes.get("shape"),
            Some(&AttributeValue::Ints(vec![1, 1, 2, 1, 2]))
        );
        let h_tile = find(&model, "layer0_trace_1_h_tile");
        assert_eq!(h_tile.attributes.get("axis"), Some(&AttributeValue::Int(3)));
        assert_eq!(h_tile.attributes.get("reps"), Some(&AttributeValue::Int(2)));
        let h_merge = find(&model, "layer0_trace_1_h_merge");
        assert_eq!(
            h_merge.attributes.get("shape"),
            Some(&AttributeValue::Ints(vec![1, 1, 4, 2]))
        );

        // W pass: [1,1,4,2,1] → tile axis 4 (len-1) → final [1,1,4,4].
        let w_unsq = find(&model, "layer0_trace_1_w_unsq");
        assert_eq!(
            w_unsq.attributes.get("shape"),
            Some(&AttributeValue::Ints(vec![1, 1, 4, 2, 1]))
        );
        let w_tile = find(&model, "layer0_trace_1_w_tile");
        assert_eq!(w_tile.attributes.get("axis"), Some(&AttributeValue::Int(4)));
        assert_eq!(w_tile.attributes.get("reps"), Some(&AttributeValue::Int(2)));
        let fin = find(&model, "layer0_trace_1");
        assert_eq!(
            fin.attributes.get("shape"),
            Some(&AttributeValue::Ints(vec![1, 1, 4, 4]))
        );

        build_ok(&model, "upsample2d");
    }

    /// Upsample2d refuses non-nearest modes (NN only supports nearest).
    #[test]
    fn upsample2d_refuses_bilinear_mode() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[1, 1, 2, 2]),
            node(
                1,
                "up",
                TraceOp::Upsample2d {
                    mode: TraceUpsampleMode::Bilinear,
                    scale_h: 2.0,
                    scale_w: 2.0,
                },
                &[0],
                &[1, 1, 4, 4],
            ),
        ]);
        let err = translate(&graph).expect_err("bilinear refused");
        assert!(
            matches!(err, NyError::UnsupportedOp(ref m)
                if m.contains("bilinear") && m.contains("only 'nearest'")),
            "got {err:?}"
        );
    }

    /// A Tile/Slice surrogate only preserves each input element's interval; it
    /// does not bound interpolated values by the global input convex hull.
    #[test]
    fn resize_bilinear_is_refused_without_a_sound_lowering() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[1, 2, 2]),
            node(
                1,
                "rb",
                TraceOp::ResizeBilinear {
                    target_h: 4,
                    target_w: 4,
                },
                &[0],
                &[1, 4, 4],
            ),
        ]);
        let err = translate(&graph).expect_err("ResizeBilinear must fail closed");
        assert!(
            matches!(err, NyError::UnsupportedOp(ref m)
                if m.contains("ResizeBilinear") && m.contains("sound")),
            "got {err:?}"
        );
    }

    /// GridSample remaps coordinates and zero padding can introduce values not
    /// present at the corresponding input element, so identity is unsound.
    #[test]
    fn grid_sample_is_refused_without_a_sound_lowering() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[1, 1, 4, 4]),
            node(
                1,
                "grid",
                TraceOp::ConstantWeight {
                    weight: WeightPayload::f32(vec![0.0; 32], vec![1, 4, 4, 2]),
                },
                &[],
                &[1, 4, 4, 2],
            ),
            node(
                2,
                "gs",
                TraceOp::GridSample {
                    padding_mode: GridSamplePaddingMode::Zeros,
                    align_corners: false,
                },
                &[0, 1],
                &[1, 1, 4, 4],
            ),
        ]);
        let err = translate(&graph).expect_err("GridSample must fail closed");
        assert!(
            matches!(err, NyError::UnsupportedOp(ref m)
                if m.contains("GridSample") && m.contains("sound")),
            "got {err:?}"
        );
    }

    /// ReflectionPad2d stays refused: NN's only handling is a vacuous
    /// opaque-skip catch-all, which the bridge must not reproduce.
    #[test]
    fn reflection_pad2d_stays_refused() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[1, 4, 4]),
            node(
                1,
                "pad",
                TraceOp::ReflectionPad2d {
                    pad_left: 1,
                    pad_right: 1,
                    pad_top: 1,
                    pad_bottom: 1,
                },
                &[0],
                &[1, 6, 6],
            ),
        ]);
        let err = translate(&graph).expect_err("ReflectionPad2d refused");
        assert!(
            matches!(err, NyError::UnsupportedOp(ref m) if m.contains("ReflectionPad2d")),
            "got {err:?}"
        );
    }
}
