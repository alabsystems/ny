// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Padding / resampling family: `ReflectionPad1d`, `ReflectionPad2d`,
//! `ConstantPadNd`, `PixelShuffle`, `PixelUnshuffle`, `Upsample1d`,
//! `Upsample2d`, `ResizeBilinear`, `GridSample`.
//!
//! Ported from NN's trace-path ground truth:
//! `trace_to_graph_layerspec_pad.rs` (ConstantPadNd),
//! `trace_to_graph_layerspec_decompose.rs` (PixelShuffle, PixelUnshuffle,
//! Upsample2d), `trace_to_graph_layerspec_shape.rs`
//! (Upsample1d), and the `translate_conservative_passthrough` arm of
//! `trace_to_graph_layerspec_dispatch_extended.rs` (GridSample).
//!
//! ## Lowerings
//!
//! - `ReflectionPad1d` / `ReflectionPad2d` → a single native `LayerType::Pad`
//!   (mode `reflect`) with ONNX `pads = [before_0.., after_0..]` at trace
//!   rank. ny-propagate's `PadLayer` `PadMode::Reflect` performs the exact
//!   PyTorch index mirroring (boundary element excluded) with exact IBP and a
//!   certified duplicate-fan-in CROWN backward, and fails closed on any
//!   geometry it cannot mirror (`pad >= dim`, `dim < 2`). This replaces the
//!   historic per-position `Slice` + `Concat` port of ReflectionPad1d, whose
//!   axis-audit note admitted it was never exercised under propagation; the
//!   native layer is exercised and exact. ReflectionPad2d has no NN arm at
//!   all (NN's catch-all emits a vacuous opaque-skip the bridge forbids), so
//!   the native Pad is its first sound lowering.
//! - `ConstantPadNd` → a `LayerType::Pad` (mode `constant`) with ONNX
//!   `pads = [before_0.., after_0..]` at trace rank. A private identity alias
//!   carries the unbatched input-shape compatibility metadata without
//!   overwriting a source tensor that may fan out to other consumers.
//! - `PixelShuffle` / `PixelUnshuffle` → `Reshape` + `Transpose` + `Reshape`
//!   with perms `[0,1,4,2,5,3]` / `[0,1,3,5,2,4]`.
//! - `Upsample1d` → `Reshape` + `Tile(axis=-1)` + `Reshape` (nearest only).
//! - `Upsample2d` (nearest) → per-axis `Reshape` + `Tile` + `Reshape`,
//!   H pass then W pass (6 specs); (bilinear) → the same SEPARATED constant
//!   `Linear` resampler `ResizeBilinear` uses, under the coordinate mapping
//!   `align_corners` names — see [`translate_upsample2d_bilinear`].
//! - `ResizeBilinear` → a SEPARATED pair of constant `Linear` resampling
//!   matrices (W pass, then a `Transpose`d H pass) — see
//!   [`translate_resize_bilinear`].
//!
//! ## The bilinear family: which conventions ride the wire
//!
//! Interpolating resamplers are the bridge's sharpest silent-unsoundness
//! hazard. Frameworks — and nn itself — disagree on `align_corners`
//! (corner-vs-center endpoint mapping), half-pixel sample centers, grid
//! normalization (`[-1, 1]` for GridSample) and out-of-range handling. A
//! lowering built on the wrong convention reads the *wrong source cells*, so
//! every emitted interval is an enclosure of the wrong value: a confidently
//! non-enclosing bound whose failure is silent (all shapes still match). The
//! decision rule is therefore **lower only what the serialized op pins
//! unambiguously, refuse the rest**:
//!
//! | wire op | convention flags on the wire? | verdict |
//! | --- | --- | --- |
//! | `ResizeBilinear { target_h, target_w }` | not carried, and nn has exactly ONE kernel (`DynTensor::resize_bilinear` → `resize_bilinear_coord`, half-pixel + clamp, no `align_corners` parameter anywhere on the path) | **lowered** |
//! | `Upsample2d { mode: Bilinear, .., align_corners }` | CARRIED (it was not, and that is why this row used to read "refused"): nn has two mappings and the flag says which one ran | **lowered**, under the named mapping |
//! | `Upsample2d { mode: Bicubic, .. }` | flag carried, but nn has NO bicubic kernel on any device path | **refused** — a missing kernel, not a missing flag |
//! | `GridSample { padding_mode, align_corners }` | both carried | conventions fine; **refused** twice over — structurally when the sampling grid is a runtime tensor operand, and ARITHMETICALLY even when it is a translate-time constant (see [`grid_sample_refusal`]) |
//!
//! `ResizeBilinear` was lowerable first *because* nn's op has no
//! `align_corners` knob at all, so the absent flag could not hide a second
//! meaning; bilinear `Upsample2d` joined it when nn started serializing the
//! flag. The two are NOT the same arithmetic even so — the half-pixel branch
//! of `bilinear_coord` multiplies then divides where `resize_bilinear_coord`
//! consumes a hoisted `scale`, so [`BilinearCoordMap`] keeps them as separate
//! coordinate expressions rather than aliasing one to the other.
//!
//! And `GridSample` is the row that shows the bar is about ARITHMETIC, not just
//! about which flags ride the wire. Its flags are all present; what stops it is
//! that grid sampling is not separable, so a resampling matrix would have to
//! pre-multiply `fl(cy * cx)` into one coefficient where nn's kernel evaluates
//! `fl(fl(v * cy) * cx)` — up to 2 ULP apart, which is wider than the tightest
//! sound interval a `Linear` layer can give for a single product. That is a
//! silent NON-ENCLOSURE, the same class of failure as reading the wrong source
//! cells, so the constant-grid case is refused alongside the runtime one.
//!
//! A second, independent hazard the old Tile/identity surrogates fell into:
//! they preserved per-element input intervals rather than the convex hull an
//! interpolated value needs, and could under-approximate reachable outputs.
//! The `Linear` lowering does not have that failure mode — it is the exact
//! affine map, propagated by `LinearLayer`'s directed-rounding interval
//! arithmetic.

use std::collections::HashMap;

use ndarray::{Array2, IxDyn};
use ny_build::{AttributeValue, DataType, LayerSpec, WeightRef};
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

/// Cap on ONE bilinear resampling matrix, in f32 entries
/// (`out_size * in_size`).
///
/// The separated lowering emits at most one matrix per spatial axis, so the
/// worst case is `2 * MAX_RESAMPLE_AXIS_MATRIX_ELEMS` f32 of frozen weight.
/// 4 Mi entries = 16 MiB per axis; a 2048→4096 resize fits, 4096→8192 does
/// not and is refused rather than silently materialized.
const MAX_RESAMPLE_AXIS_MATRIX_ELEMS: usize = 1 << 22;

/// Translate a pad/resample-family op node.
///
/// Dispatches to the per-op lowering; the bilinear/resampling family (and any
/// non-family op mis-routed here) refuses with an explicit
/// [`NyError::UnsupportedOp`] — see the module docs for the convention-risk
/// rationale.
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
        } => translate_reflection_pad_native(
            "ReflectionPad1d",
            name,
            &[ReflectAxis {
                offset_from_end: 1,
                pad_before: *pad_left,
                pad_after: *pad_right,
                label_before: "pad_left",
                label_after: "pad_right",
            }],
            input_tensors,
            output_tensor,
            output_shape,
            ctx,
        ),
        TraceOp::ReflectionPad2d {
            pad_left,
            pad_right,
            pad_top,
            pad_bottom,
        } => translate_reflection_pad_native(
            "ReflectionPad2d",
            name,
            &[
                ReflectAxis {
                    offset_from_end: 1,
                    pad_before: *pad_left,
                    pad_after: *pad_right,
                    label_before: "pad_left",
                    label_after: "pad_right",
                },
                ReflectAxis {
                    offset_from_end: 2,
                    pad_before: *pad_top,
                    pad_after: *pad_bottom,
                    label_before: "pad_top",
                    label_after: "pad_bottom",
                },
            ],
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
            align_corners,
        } => translate_upsample2d(
            name,
            *mode,
            *scale_h,
            *scale_w,
            *align_corners,
            input_tensors,
            output_tensor,
            output_shape,
            ctx,
        ),
        TraceOp::ResizeBilinear { target_h, target_w } => translate_resize_bilinear(
            name,
            *target_h,
            *target_w,
            input_tensors,
            output_tensor,
            output_shape,
            ctx,
        ),
        // GridSample: refused, and the reason DEPENDS ON THE GRID OPERAND, so
        // the arm resolves which of the two refusals applies rather than
        // asserting the runtime one unconditionally (see
        // [`grid_sample_refusal`]). The op's CONVENTIONS are not at issue
        // either way: `padding_mode` and `align_corners` are both serialized,
        // so the ambiguity that refuses `Upsample2d` bilinear does not apply.
        TraceOp::GridSample { .. } => Err(grid_sample_refusal(input_tensors, ctx)),
        // Any mis-routed op: fail-closed refusal with the pre-split
        // catch-all's error shape.
        other => Err(NyError::UnsupportedOp(format!(
            "{} not supported in NY trace translation",
            op_name(other)
        ))),
    }
}

// ---------------------------------------------------------------------------
// GridSample — refused, with the reason resolved from the grid operand
// ---------------------------------------------------------------------------

/// Build the `GridSample` refusal, naming the obstruction that actually
/// applies to THIS node.
///
/// # The runtime-grid case (structural)
///
/// GridSample's source coordinates come from the `grid` TENSOR OPERAND, so the
/// output-to-input map is a function of a runtime value. The bridge admits
/// value-dependent operands only as translate-time constants (the
/// `Gather`/`IndexSelect` index-slot rule in
/// `mod.rs::is_value_independent_integer_leaf`); a runtime grid has no map to
/// freeze into a layer, and a layer emitted without one would bound the wrong
/// source cells.
///
/// # The constant-grid case (arithmetic) — the carve-out that does NOT exist
///
/// A file comment used to describe a constant-grid carve-out as merely
/// unimplemented, and handed down a recipe for it: freeze the grid, replicate
/// nn's `unnormalize` / `bilinear_sample` / `safe_pixel`, "and emit the
/// two-stage form (x-interpolate, then scale-and-add along y) with a
/// matrix-size cap". That recipe is wrong, and the comment is gone. The
/// carve-out is refused on ARITHMETIC grounds, held to the same bar
/// [`bilinear_axis_matrix`] meets — nn's **bit-identical f32 coefficients**:
///
/// nn's kernel (nn-core `ops_ext/spatial_grid_sample.rs::bilinear_sample`)
/// evaluates, in f32, left to right:
///
/// ```text
/// v00*(1-wy)*(1-wx) + v01*(1-wy)*wx + v10*wy*(1-wx) + v11*wy*wx
/// ```
///
/// Each corner is scaled by its **y** weight FIRST and by its **x** weight
/// SECOND: `fl(fl(v * cy) * cx)`.
///
/// 1. **A single matrix cannot reproduce that.** Unlike a resize, grid sampling
///    is not separable — `ix` and `iy` vary independently per output cell — so
///    the per-axis factorization that lets [`translate_resize_bilinear`] write
///    `1.0f32 - w` and `w` into a matrix verbatim is unavailable, and a matrix
///    entry must carry the PRODUCT `fl(cy * cx)`. Then the layer bounds
///    `v * fl(cy*cx)` while nn computed `fl(fl(v*cy) * cx)`, and those differ
///    by up to **2 ULP** — while the tightest sound interval a `Linear` layer
///    can produce for one product (the exact product, rounded outward) spans
///    only 1 ULP either side. nn's own value can therefore fall OUTSIDE the
///    emitted bounds. `grid_sample_single_matrix_lowering_would_not_enclose_nn`
///    pins a concrete witness; this is a silent non-enclosure, not a looseness.
/// 2. **The two-stage repair does not repair it.** Interpolating along x and
///    then scaling by y re-associates every term (`fl(fl(v*(1-wx)) + ...)` then
///    `* cy`), which is a DIFFERENT rounding structure from nn's y-then-x
///    order, and it disagrees with the kernel on MORE random inputs than the
///    single matrix does, not fewer (measured: ~58% vs ~40% of samples differ
///    at the bit level). Reproducing nn's association operation-for-operation
///    would need a gather-to-corners tensor plus two elementwise constant
///    multiplies plus a fixed-order 4-term sum — a far larger mechanism than
///    "two matrices", and one no observed trace needs.
/// 3. **Nothing calls for it.** A constant grid means a fixed warp; the real
///    GridSample users — deformable attention above all — compute the grid from
///    features, which is case 1. So the carve-out would add soundness surface
///    for a case that does not occur.
///
/// If a constant-grid trace ever does show up, the honest options are the
/// gather-plus-two-multiplies emission above, or a matrix lowering whose rows
/// are WIDENED outward by the maximum association gap — not the deleted recipe.
fn grid_sample_refusal(input_tensors: &[String], ctx: &Ctx) -> NyError {
    let grid_is_constant = input_tensors
        .get(1)
        .is_some_and(|grid| ctx.constant_tensors.contains(grid) && ctx.weights.contains_key(grid));

    if grid_is_constant {
        return NyError::UnsupportedOp(
            "GridSample with a translate-time-constant grid is refused on ARITHMETIC grounds, \
             not structural ones: grid sampling is not separable (ix and iy vary independently \
             per output cell), so a constant resampling matrix would have to carry the product \
             fl(cy*cx) as one coefficient, while nn's kernel evaluates fl(fl(v*cy)*cx) — up to \
             2 ULP away, where the tightest sound interval a Linear layer can give for one \
             product spans 1 ULP either side, so nn's own value can fall outside the emitted \
             bounds. Reproducing nn's association needs a gather-to-corners tensor plus two \
             elementwise constant multiplies plus a fixed-order 4-term sum; the x-then-y \
             two-stage form is NOT that repair (it re-associates every term and disagrees with \
             the kernel on more inputs, not fewer). ResizeBilinear is lowered only because its \
             per-axis factorization lets the matrix carry nn's coefficients verbatim"
                .to_string(),
        );
    }

    NyError::UnsupportedOp(
        "GridSample: no sound lowering is implemented — its sampling grid is a runtime \
         tensor operand, so the output-to-input coordinate map is value-dependent and \
         cannot be frozen into a layer at translate time (the bridge admits \
         value-dependent operands only as translate-time constants, per the \
         Gather/IndexSelect index-slot rule); emitting a layer without that map would \
         bound the wrong source cells. The op's conventions themselves are \
         unambiguous: padding_mode and align_corners both ride the wire. A CONSTANT grid \
         is refused too, for a separate arithmetic reason — see grid_sample_refusal"
            .to_string(),
    )
}

// ---------------------------------------------------------------------------
// ReflectionPad1d / ReflectionPad2d → native LayerType::Pad (mode reflect)
// ---------------------------------------------------------------------------

/// One reflected axis of a reflection-pad op, addressed from the end of the
/// trace shape so the lowering is rank-agnostic (mirrors the trailing-relative
/// axis convention used for every other emitted axis).
struct ReflectAxis {
    /// Offset from the end of the shape: `1` = last dim (W), `2` = H.
    offset_from_end: usize,
    /// Elements mirrored before the axis start (`pad_left` / `pad_top`).
    pad_before: usize,
    /// Elements mirrored after the axis end (`pad_right` / `pad_bottom`).
    pad_after: usize,
    /// Trace-parameter names for error messages.
    label_before: &'static str,
    label_after: &'static str,
}

/// Translate `ReflectionPad1d` / `ReflectionPad2d` → one native
/// `LayerType::Pad` (mode `reflect`) with ONNX
/// `pads = [before_0.., after_0..]` at trace rank.
///
/// Reflection padding mirrors interior values at the boundary, excluding the
/// boundary element itself: `[a, b, c, d, e]` with pad_left=2, pad_right=1 →
/// `[c, b, a, b, c, d, e, d]`. ny-propagate's `PadLayer` `PadMode::Reflect`
/// implements exactly that index map (`reflect_index`), so IBP is an exact
/// index copy of each bound and the CROWN backward scatter carries a
/// certified duplicate-fan-in coefficient error. The graph-build convert arm
/// (`ny-build` `convert/pad.rs`) maps mode `"reflect"` to `PadMode::Reflect`
/// verbatim; the emitted spec carries the same internal
/// [`ny_build::PAD_PRESERVE_ALL_AXES_ATTR`] certificate as `ConstantPadNd`,
/// because trace shapes are already data-layout shapes and the leading pad
/// pair must not be discarded as an ONNX batch axis.
///
/// Fail-closed geometry checks, per reflected axis:
/// - pad sum overflow / `output dim < total pad` are internal errors (the
///   trace is inconsistent with itself);
/// - `total pad > MAX_PAD_SIZE` keeps the family's decomposition cap;
/// - `pad >= input length` is refused: reflection has no source element to
///   mirror (PyTorch rejects this too), and `PadLayer` would refuse the same
///   geometry at build time — refusing here names the trace parameter.
///
/// A zero-total pad still needs a real producer for the node's output tensor
/// and lowers to the identity `Add(x, 0)`, like every other no-op arm.
fn translate_reflection_pad_native(
    op: &str,
    name: &str,
    axes: &[ReflectAxis],
    input_tensors: &[String],
    output_tensor: &str,
    output_shape: &[usize],
    ctx: &mut Ctx,
) -> Result<NodeOutput> {
    let data_input = first_input(input_tensors, op)?;

    let rank = output_shape.len();
    let needed_rank = axes
        .iter()
        .map(|axis| axis.offset_from_end)
        .max()
        .unwrap_or(0);
    if rank < needed_rank {
        return Err(NyError::UnsupportedOp(format!(
            "{op}: output rank {rank} is less than the {needed_rank} required for its \
             reflected axes"
        )));
    }

    let mut before = vec![0_i64; rank];
    let mut after = vec![0_i64; rank];
    let mut grand_total: usize = 0;
    for axis in axes {
        let ReflectAxis {
            offset_from_end,
            pad_before,
            pad_after,
            label_before,
            label_after,
        } = *axis;
        let pair_total = pad_before.checked_add(pad_after).ok_or_else(|| {
            NyError::InternalError(format!(
                "{op}: {label_before} + {label_after} overflows usize"
            ))
        })?;
        if pair_total > MAX_PAD_SIZE {
            return Err(NyError::UnsupportedOp(format!(
                "{op}: total pad {pair_total} on axis -{offset_from_end} exceeds limit \
                 {MAX_PAD_SIZE}"
            )));
        }
        let dim_index = rank - offset_from_end;
        let out_dim = output_shape[dim_index];
        let input_len = out_dim.checked_sub(pair_total).ok_or_else(|| {
            NyError::InternalError(format!("{op}: output dim {out_dim} < pad {pair_total}"))
        })?;
        // Reflection mirrors interior elements only (boundary excluded), so
        // each side needs `pad` strictly inside the input extent. PadLayer
        // enforces the same bound at build time; refusing here names the
        // offending trace parameter instead of a converted-layer axis.
        for (pad, label) in [(pad_before, label_before), (pad_after, label_after)] {
            if pad > 0 && pad >= input_len {
                return Err(NyError::UnsupportedOp(format!(
                    "{op}: {label} {pad} must be smaller than the input length {input_len} \
                     on axis -{offset_from_end} (reflection mirrors interior elements, \
                     excluding the boundary)"
                )));
            }
        }
        before[dim_index] = dim_as_i64(pad_before, &format!("{op} before"))?;
        after[dim_index] = dim_as_i64(pad_after, &format!("{op} after"))?;
        grand_total = grand_total.checked_add(pair_total).ok_or_else(|| {
            NyError::InternalError(format!("{op}: aggregate padding overflows usize"))
        })?;
    }

    // A no-op still needs a real producer for this trace node's output tensor.
    if grand_total == 0 {
        return ops_core::translate_identity_add_zero(
            name,
            &format!("{op} with zero padding"),
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
        AttributeValue::String("reflect".to_string()),
    );
    attrs.insert("pads".to_string(), AttributeValue::Ints(pads));
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

/// Upsample2d: nearest → 2× (Reshape + Tile + Reshape); bilinear → the
/// separated constant-`Linear` resampler [`ResizeBilinear`](translate_resize_bilinear)
/// uses, under the coordinate mapping `align_corners` names.
///
/// # Why bilinear lowers now
///
/// It used to be refused, and the refusal was correct at the time: nn recorded
/// `Upsample2d { mode: Bilinear, scale_h, scale_w }` from BOTH
/// `DynTensor::upsample_bilinear_2d(.., align_corners)` and
/// `upsample_bilinear_2d_to_size(.., align_corners)` without serializing the
/// flag, and the two flag values are two different source-coordinate maps —
///
/// ```text
/// align_corners = true : src = dst * (in - 1) / (out - 1)
/// align_corners = false: src = (dst + 0.5) * in / out - 0.5
/// ```
///
/// — with identical wire bytes and identical output shapes. `align_corners` is
/// now on the wire (nn-core `TraceOp::Upsample2d`), recorded from the same
/// variable the compute body branches on, so the convention is PINNED per node
/// and each one lowers under the mapping it actually ran. Both flag values are
/// lowered: each is a constant affine map with convex per-row weights, so the
/// resampler encloses either one; nothing about the `true` branch is less
/// determined than the `false` branch once the flag is known.
///
/// # Why the resampler is reused rather than a new kernel written
///
/// `bilinear_2d_core` computes the same 4-tap sum `resize_bilinear_compute`
/// does, over source cells picked by the same `floor`/`min` clamp with the same
/// f32 weight cast — only the coordinate EXPRESSION differs. Separability and
/// the exact-merge invariant therefore carry over verbatim; see
/// [`bilinear_axis_matrix`] for both arguments and
/// [`emit_separable_bilinear_resample`] for the emission. What does NOT carry
/// over is the arithmetic: the half-pixel branch here multiplies then divides
/// where `ResizeBilinear` consumes a hoisted `scale`, which is why
/// [`BilinearCoordMap`] keeps them as separate variants rather than aliasing
/// one to the other (see [`upsample_bilinear_coord`]).
///
/// # Geometry
///
/// The coordinate map is a function of `in_size`/`out_size` only — nn passes
/// `bilinear_coord` the extents, never the scale — so `scale_h`/`scale_w`
/// matter solely as the thing that produced `out_h`/`out_w`, and the recorded
/// output shape already carries those. Input extents come from the recorded
/// producer shape, exactly as `ResizeBilinear` reads them. The nearest path
/// keeps deriving its input extents by dividing the output shape by the integer
/// scale, because that path has no recorded-shape dependency to begin with.
///
/// `Bicubic` stays refused: nn has no bicubic kernel at all (the mode enum
/// carries the variant; nothing records it), so there is no ground truth to
/// replicate — a missing kernel, not a missing flag.
#[allow(clippy::too_many_arguments)]
fn translate_upsample2d(
    name: &str,
    mode: TraceUpsampleMode,
    scale_h: f64,
    scale_w: f64,
    align_corners: bool,
    input_tensors: &[String],
    output_tensor: &str,
    output_shape: &[usize],
    ctx: &mut Ctx,
) -> Result<NodeOutput> {
    match mode {
        TraceUpsampleMode::Nearest => {}
        TraceUpsampleMode::Bilinear => {
            return translate_upsample2d_bilinear(
                name,
                align_corners,
                input_tensors,
                output_tensor,
                output_shape,
                ctx,
            );
        }
        TraceUpsampleMode::Bicubic => {
            return Err(NyError::UnsupportedOp(
                "Upsample2d mode 'bicubic' not supported — nn has no bicubic kernel on any \
                 device path (the mode enum carries the variant, nothing records it), so there \
                 is no ground-truth semantics to lower. Unlike the bilinear modes, this is a \
                 MISSING KERNEL rather than a missing wire field: align_corners is now \
                 serialized, but there is no nn arithmetic for it to disambiguate"
                    .to_string(),
            ));
        }
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

/// Translate `Upsample2d { mode: Bilinear, .. }` via the separated constant
/// `Linear` resampler, under the mapping `align_corners` selects.
///
/// ## Soundness argument
///
/// Identical in structure to `ResizeBilinear`'s, and for the same reasons:
///
/// * **The geometry is trace-time known.** `bilinear_2d_core` derives every
///   source coordinate from `in_h`/`in_w`/`out_h`/`out_w` alone — no tensor
///   VALUE participates — so the op is a CONSTANT linear map. Output extents
///   come from the recorded output shape and input extents from the recorded
///   producer shape, both fixed at translate time.
/// * **The emitted coefficients are nn's, not a re-derivation.**
///   [`upsample_bilinear_coord`] is `bilinear_coord` transcribed
///   operation-for-operation (including which of the two half-pixel
///   associations it uses), and [`bilinear_axis_matrix`] replicates the
///   `floor`/`min` clamp and the `as f32` weight cast. `1 - w` and `w` are
///   written into the matrix verbatim, never pre-multiplied into a `cy*cx`
///   product that would round — which is exactly what makes the separated pair
///   model the fused 4-tap sum rather than approximate it.
/// * **Each matrix is row-stochastic with non-negative entries**, so the
///   composite map is a convex combination of input cells and its exact range
///   lies inside the input hull.
/// * **Float rounding is the IBP tier's job, not this lowering's.** The emitted
///   nodes are ordinary `LayerType::Linear`, so `propagate_ibp` accumulates
///   round-to-nearest and `propagate_ibp_sound` applies the certified per-node
///   widening; this lowering neither strengthens nor weakens that contract.
///
/// The SCOPE caveat on [`resize_bilinear_coord`] applies here unchanged: what
/// is certified is nn's f64 coordinate arithmetic. `bilinear_2d_core` is the
/// only kernel on this op (its GPU path round-trips to CPU explicitly), so
/// there is no second backend to diverge today.
///
/// ## Fail-closed refusals
///
/// - rank < 2 (nn's `upsample_bilinear_2d` refuses it too);
/// - a recorded input shape whose rank or batch/channel prefix contradicts the
///   output shape;
/// - zero spatial extent on either side;
/// - a resampling matrix beyond [`MAX_RESAMPLE_AXIS_MATRIX_ELEMS`].
fn translate_upsample2d_bilinear(
    name: &str,
    align_corners: bool,
    input_tensors: &[String],
    output_tensor: &str,
    output_shape: &[usize],
    ctx: &mut Ctx,
) -> Result<NodeOutput> {
    let data_input = first_input(input_tensors, "Upsample2d")?;
    let rank = output_shape.len();
    if rank < 2 {
        return Err(NyError::UnsupportedOp(format!(
            "Upsample2d (bilinear): output rank {rank} < 2 — nn's upsample_bilinear_2d refuses \
             a rank below 2, so the trace disagrees with the op it records"
        )));
    }

    let in_shape = lookup_shape(ctx, &data_input, "Upsample2d (bilinear)")?;
    if in_shape.len() != rank {
        return Err(NyError::InternalError(format!(
            "Upsample2d (bilinear): input rank {} does not match output rank {rank}",
            in_shape.len()
        )));
    }
    if in_shape[..rank - 2] != output_shape[..rank - 2] {
        return Err(NyError::InternalError(format!(
            "Upsample2d (bilinear): batch/channel prefix {:?} does not survive to the output {:?}",
            &in_shape[..rank - 2],
            &output_shape[..rank - 2]
        )));
    }
    let (in_h, in_w) = (in_shape[rank - 2], in_shape[rank - 1]);
    let (out_h, out_w) = (output_shape[rank - 2], output_shape[rank - 1]);
    if in_h == 0 || in_w == 0 || out_h == 0 || out_w == 0 {
        return Err(NyError::UnsupportedOp(format!(
            "Upsample2d (bilinear): spatial extents must be > 0, got in [{in_h}, {in_w}] \
             out [{out_h}, {out_w}]"
        )));
    }

    let map = if align_corners {
        BilinearCoordMap::UpsampleAlignCorners
    } else {
        BilinearCoordMap::UpsampleHalfPixel
    };
    emit_separable_bilinear_resample(
        name,
        "Upsample2d (bilinear)",
        map,
        &data_input,
        in_shape,
        out_h,
        out_w,
        input_tensors,
        output_tensor,
        ctx,
    )
}

// ---------------------------------------------------------------------------
// ResizeBilinear → separated constant Linear resampling matrices
// ---------------------------------------------------------------------------

/// nn-core's bilinear-resize source-coordinate map, replicated verbatim.
///
/// GROUND TRUTH: `nn-core/src/dyn_tensor/ops_ext/spatial_upsample.rs`
/// `resize_bilinear_coord`:
///
/// ```text
/// let src = (dst as f64 + 0.5) * scale - 0.5;
/// src.clamp(0.0, (in_size - 1) as f64)
/// ```
///
/// with `scale = in_size as f64 / out_size as f64` hoisted out of the loop by
/// the caller (`resize_bilinear_compute` computes `scale_y` / `scale_x` once).
/// This is the half-pixel-center ("`align_corners = false`") convention, and
/// it is the ONLY convention on nn's `ResizeBilinear` path: `resize_bilinear`
/// takes no `align_corners` argument, and the Metal kernel
/// (`nn-metal/src/dyn_tensor_metal_resize_bilinear.rs`) documents and
/// implements the same mapping.
///
/// The arithmetic here is bit-for-bit nn's — same `f64` operations in the same
/// order, same `clamp` — so the emitted matrix selects exactly nn's source
/// cells with exactly nn's interpolation weights.
///
/// SCOPE (do not overstate this): what is certified is nn's `f64` coordinate
/// arithmetic, which is what both nn backends on this path implement (the CPU
/// `resize_bilinear_coord` and the Metal kernel, which documents the same
/// mapping). A hypothetical backend computing the coordinate in `f32` is NOT
/// covered: a <= 1 ULP perturbation of `src` straddling an integer boundary
/// changes `floor(src)`, hence the selected cell PAIR — and while the
/// interpolated VALUE stays continuous across that switch (the far cell takes
/// weight 0 at the boundary), the emitted matrix is a FIXED constant map, so it
/// would then model a slightly different function than the backend ran. Should
/// such a backend appear on this wire op, this lowering must either widen each
/// row by the max |f64 weight - f32 weight| over the row or refuse.
fn resize_bilinear_coord(dst: usize, scale: f64, in_size: usize) -> f64 {
    let src = (dst as f64 + 0.5) * scale - 0.5;
    src.clamp(0.0, (in_size - 1) as f64)
}

/// nn-core's `Upsample2d` source-coordinate map, replicated verbatim.
///
/// GROUND TRUTH: `nn-core/src/dyn_tensor/ops_ext/spatial_upsample.rs`
/// `bilinear_coord`, the function `bilinear_2d_core` calls per output row and
/// column (it is handed `in_size` and `out_size`, NOT a scale factor — the
/// traced `scale_h`/`scale_w` only ever determine `out_h`/`out_w`):
///
/// ```text
/// let src = if align_corners && out_size > 1 {
///     dst as f64 * (in_size as f64 - 1.0) / (out_size as f64 - 1.0)
/// } else if align_corners {
///     0.0
/// } else {
///     (dst as f64 + 0.5) * (in_size as f64) / (out_size as f64) - 0.5
/// };
/// src.clamp(0.0, (in_size - 1) as f64)
/// ```
///
/// # This is NOT [`resize_bilinear_coord`] with a different name
///
/// The `align_corners = false` branch is the same half-pixel CONVENTION as
/// `ResizeBilinear`'s, but not the same f64 EXPRESSION: this one multiplies
/// then divides (`(dst + 0.5) * in / out`), while `resize_bilinear_coord`
/// consumes a `scale = in / out` its caller hoisted out of the loop
/// (`(dst + 0.5) * scale`). `fl(fl(a * b) / c)` and `fl(a * fl(b / c))` differ
/// in general, and a sub-ULP difference in `src` that straddles an integer
/// changes `floor(src)` — hence the selected source cell PAIR. Re-using the
/// hoisted-scale form here would model a function nn did not run at some
/// geometries, so the two expressions are kept apart deliberately.
fn upsample_bilinear_coord(
    dst: usize,
    in_size: usize,
    out_size: usize,
    align_corners: bool,
) -> f64 {
    let src = if align_corners && out_size > 1 {
        dst as f64 * (in_size as f64 - 1.0) / (out_size as f64 - 1.0)
    } else if align_corners {
        // out_size == 1: map to center (0.0 for a single output pixel).
        0.0
    } else {
        (dst as f64 + 0.5) * (in_size as f64) / (out_size as f64) - 0.5
    };
    src.clamp(0.0, (in_size - 1) as f64)
}

/// Which nn source-coordinate mapping a resampling matrix is built from.
///
/// The three variants are three DISTINCT f64 expressions, not three spellings
/// of one — see [`upsample_bilinear_coord`] for why the two half-pixel forms
/// must not be collapsed. Each is transcribed operation-for-operation from the
/// nn function named in its doc.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BilinearCoordMap {
    /// nn `resize_bilinear_coord`, with `scale = in / out` hoisted by the
    /// caller exactly as `resize_bilinear_compute` hoists `scale_y`/`scale_x`.
    ResizeHoistedScale,
    /// nn `bilinear_coord(.., align_corners = false)`.
    UpsampleHalfPixel,
    /// nn `bilinear_coord(.., align_corners = true)`.
    UpsampleAlignCorners,
}

impl BilinearCoordMap {
    /// The clamped source coordinate this mapping assigns to output cell `dst`.
    fn coord(self, dst: usize, in_size: usize, out_size: usize) -> f64 {
        match self {
            Self::ResizeHoistedScale => {
                // Hoisted exactly as nn hoists `scale_y` / `scale_x`. Recomputed
                // per call here rather than per axis; `in_size` and `out_size`
                // are loop-invariant, so this is the same f64 value the hoist
                // produces, not a re-association.
                let scale = in_size as f64 / out_size as f64;
                resize_bilinear_coord(dst, scale, in_size)
            }
            Self::UpsampleHalfPixel => upsample_bilinear_coord(dst, in_size, out_size, false),
            Self::UpsampleAlignCorners => upsample_bilinear_coord(dst, in_size, out_size, true),
        }
    }
}

/// Build the `[out_size, in_size]` bilinear resampling matrix for ONE spatial
/// axis.
///
/// # Why one matrix per axis is EXACTLY nn's function
///
/// nn's inner loop computes, for output cell `(oh, ow)`:
///
/// ```text
/// v00*(1-wy)*(1-wx) + v01*(1-wy)*wx + v10*wy*(1-wx) + v11*wy*wx
/// ```
///
/// Factor the row sum: `(1-wy) * [v00*(1-wx) + v01*wx] + wy * [v10*(1-wx) +
/// v11*wx]`. Bilinear interpolation is *separable*: interpolate along W for
/// every source row, then along H. In exact real arithmetic the separated and
/// fused forms are the same number, so the emitted pair of `Linear` layers
/// realizes nn's function with nn's **bit-identical f32 coefficients** —
/// `1.0f32 - w` and `w` are written into the matrix verbatim, never
/// pre-multiplied into a `cy*cx` product that would round.
///
/// # The duplicate-index merge is exact
///
/// When `i0 == i1` (the clamped last cell, or `in_size == 1`) the two
/// contributions land in the same matrix entry, `(1 - w) + w`. That merge is
/// exact *because `w` is provably 0 there*: `src` is clamped into
/// `[0, in_size - 1]`, so `floor(src) == in_size - 1` forces
/// `src == in_size - 1` exactly, hence `w = src - i0 = 0`. The entry is then
/// `1.0f32 + 0.0f32 = 1.0f32`, with no rounding. The invariant is not
/// assumed — it is CHECKED below and refuses fail-closed if it ever breaks,
/// because a merged entry with `w != 0` would be the one place a coefficient
/// could round away from nn's.
fn bilinear_axis_matrix(
    in_size: usize,
    out_size: usize,
    axis: &str,
    map: BilinearCoordMap,
    op: &str,
) -> Result<Array2<f32>> {
    if in_size == 0 || out_size == 0 {
        return Err(NyError::UnsupportedOp(format!(
            "{op}: {axis} extent must be non-zero (in {in_size}, out {out_size})"
        )));
    }
    let elems = out_size.checked_mul(in_size).ok_or_else(|| {
        NyError::InternalError(format!(
            "{op}: {axis} matrix {out_size}x{in_size} overflows usize"
        ))
    })?;
    if elems > MAX_RESAMPLE_AXIS_MATRIX_ELEMS {
        return Err(NyError::UnsupportedOp(format!(
            "{op}: {axis} resampling matrix {out_size}x{in_size} = {elems} entries \
             exceeds limit {MAX_RESAMPLE_AXIS_MATRIX_ELEMS}"
        )));
    }

    let mut matrix = Array2::<f32>::zeros((out_size, in_size));
    for dst in 0..out_size {
        let src = map.coord(dst, in_size, out_size);
        // nn: `let y0 = src_y.floor() as usize; let y1 = (y0 + 1).min(in_h - 1);
        //      let y0 = y0.min(in_h - 1);`
        //
        // `bilinear_2d_core` clamps y0 BEFORE deriving y1
        // (`let y0 = src_y.floor().min(in_h - 1); let y1 = (y0 + 1).min(in_h - 1);`).
        // The two orderings agree everywhere: `src` is clamped into
        // `[0, in_size - 1]`, so `floor(src) <= in_size - 1` and the `.min` on
        // `i0` is a no-op — which is why one builder serves both kernels.
        let i0_raw = src.floor() as usize;
        let i1 = (i0_raw + 1).min(in_size - 1);
        let i0 = i0_raw.min(in_size - 1);
        // nn: `let wy = (src_y - y0 as f64) as f32;`
        let weight = (src - i0 as f64) as f32;
        if i0 == i1 && weight != 0.0 {
            return Err(NyError::InternalError(format!(
                "{op}: {axis} cell {dst} merges source index {i0} with a non-zero \
                 interpolation weight {weight} — the exact-merge invariant nn's clamp \
                 guarantees is broken, refusing rather than rounding the coefficient"
            )));
        }
        matrix[[dst, i0]] += 1.0_f32 - weight;
        matrix[[dst, i1]] += weight;
    }
    Ok(matrix)
}

/// Emit one `Linear` (ONNX Gemm, `transB = 1`) applying `matrix` to the last
/// axis of `input`, registering the matrix as a frozen weight.
fn resample_linear_spec(
    name: &str,
    matrix: Array2<f32>,
    input: &str,
    output: &str,
    ctx: &mut Ctx,
) -> Result<LayerSpec> {
    let (out_features, in_features) = matrix.dim();
    let weight_name = format!("{name}_weight");
    let shape = vec![out_features, in_features];
    let array = matrix
        .into_shape_with_order(IxDyn(&shape))
        .map_err(|e| NyError::InternalError(format!("{name}: resampling weight reshape: {e}")))?;
    ctx.insert_weight(&weight_name, array)?;

    let mut attrs = HashMap::new();
    attrs.insert("transB".to_string(), AttributeValue::Int(1));
    Ok(LayerSpec {
        name: name.to_string(),
        layer_type: LayerType::Linear,
        inputs: vec![input.to_string(), weight_name.clone()],
        outputs: vec![output.to_string()],
        weights: Some(WeightRef {
            name: weight_name,
            shape,
            original_dtype: DataType::Float32,
        }),
        attributes: attrs,
    })
}

/// Emit a `Transpose` swapping the last two axes of a rank-`rank` tensor.
fn swap_last_two_spec(name: &str, rank: usize, input: &str, output: &str) -> Result<LayerSpec> {
    if rank < 2 {
        return Err(NyError::InternalError(format!(
            "{name}: rank {rank} has no two trailing axes to swap"
        )));
    }
    let mut perm: Vec<i64> = (0..rank)
        .map(|axis| dim_as_i64(axis, "Transpose perm"))
        .collect::<Result<_>>()?;
    perm.swap(rank - 2, rank - 1);
    Ok(transpose_spec(name, vec![input.to_string()], output, &perm))
}

/// Look up a previously translated tensor's recorded shape as `usize` dims.
fn lookup_shape(ctx: &Ctx, tensor: &str, context: &str) -> Result<Vec<usize>> {
    let shape = ctx.tensor_shapes.get(tensor).ok_or_else(|| {
        NyError::InternalError(format!("{context}: input shape not found for {tensor}"))
    })?;
    shape
        .iter()
        .map(|&dim| {
            usize::try_from(dim).map_err(|_| {
                NyError::InternalError(format!("{context}: negative dimension {dim} in {tensor}"))
            })
        })
        .collect()
}

/// Translate `ResizeBilinear { target_h, target_w }` into a separated pair of
/// constant `Linear` resampling matrices.
///
/// ```text
/// [.., H_in, W_in]  --Linear(A_x: [W_out, W_in])-->  [.., H_in, W_out]
///                   --Transpose(swap -1,-2)------->  [.., W_out, H_in]
///                   --Linear(A_y: [H_out, H_in])-->  [.., W_out, H_out]
///                   --Transpose(swap -1,-2)------->  [.., H_out, W_out]
/// ```
///
/// An axis whose extent is unchanged contributes an exact identity matrix, so
/// its pass is SKIPPED entirely (`A = I` is provable from the coordinate map:
/// `in == out` ⇒ `scale = 1` ⇒ `src = dst` ⇒ `w = 0`). When both axes are
/// unchanged the node lowers to the family's `Add(x, 0)` identity — nn's own
/// `resize_bilinear` short-circuits that case before recording, so it should
/// never appear on the wire, but a no-op node still needs a real producer.
///
/// ## Soundness argument
///
/// The geometry is entirely trace-time-known (`target_h` / `target_w` are
/// serialized fields and the input extents come from the recorded shape), so
/// the op is a CONSTANT linear map. Each emitted matrix is row-substochastic
/// with non-negative entries summing to 1 (the two interpolation weights of a
/// convex combination), so the composite map is a convex combination of input
/// cells — its exact range is inside the input hull.
///
/// The float-rounding enclosure comes from ny's IBP tier contract, NOT from
/// anything special to this lowering: the emitted node is a `LayerType::Linear`
/// like any other, so plain `propagate_ibp` accumulates round-to-nearest (as it
/// does for every Linear/Conv in the graph) while `propagate_ibp_sound` applies
/// the certified per-node widening — for Linear, `in_features + 2` ULPs by
/// Higham Thm 3.1 over the two matmuls, the combine and the bias (see
/// `network/graph_ibp.rs`). Callers needing a float-certified answer use the
/// sound entry; this lowering neither strengthens nor weakens that contract.
///
/// What makes this SOUND rather than merely plausible is that the source-cell
/// selection is nn's, not a re-derivation: [`resize_bilinear_coord`] and
/// [`bilinear_axis_matrix`] replicate nn's f64 coordinate arithmetic,
/// `floor`/`min` clamping and f32 weight cast operation-for-operation. The
/// tests below pin that against nn's kernel semantics over dense random
/// inputs at integer, non-integer, downscaling and degenerate scale factors.
///
/// ## Fail-closed refusals
///
/// - rank outside `3..=4` — nn's `resize_bilinear` accepts only those, so any
///   other rank means the trace disagrees with the op it claims to record;
/// - zero spatial extent (nn refuses the same);
/// - a recorded input shape that contradicts `output_shape` on the batch /
///   channel prefix or the target extents;
/// - a resampling matrix beyond [`MAX_RESAMPLE_AXIS_MATRIX_ELEMS`].
fn translate_resize_bilinear(
    name: &str,
    target_h: usize,
    target_w: usize,
    input_tensors: &[String],
    output_tensor: &str,
    output_shape: &[usize],
    ctx: &mut Ctx,
) -> Result<NodeOutput> {
    let data_input = first_input(input_tensors, "ResizeBilinear")?;

    // nn's `DynTensor::resize_bilinear` accepts rank 3 ([C, H, W]) or rank 4
    // ([N, C, H, W]) only; anything else never produced this op.
    let rank = output_shape.len();
    if !(3..=4).contains(&rank) {
        return Err(NyError::UnsupportedOp(format!(
            "ResizeBilinear: expected a rank-3 or rank-4 output, got rank {rank} — nn's \
             resize_bilinear refuses every other rank, so the trace disagrees with the op \
             it records"
        )));
    }
    if target_h == 0 || target_w == 0 {
        return Err(NyError::UnsupportedOp(format!(
            "ResizeBilinear: target dimensions must be > 0, got [{target_h}, {target_w}]"
        )));
    }

    let in_shape = lookup_shape(ctx, &data_input, "ResizeBilinear")?;
    if in_shape.len() != rank {
        return Err(NyError::InternalError(format!(
            "ResizeBilinear: input rank {} does not match output rank {rank}",
            in_shape.len()
        )));
    }
    let (in_h, in_w) = (in_shape[rank - 2], in_shape[rank - 1]);
    if in_h == 0 || in_w == 0 {
        return Err(NyError::UnsupportedOp(format!(
            "ResizeBilinear: input spatial dimensions must be > 0, got [{in_h}, {in_w}]"
        )));
    }
    if output_shape[rank - 2] != target_h || output_shape[rank - 1] != target_w {
        return Err(NyError::InternalError(format!(
            "ResizeBilinear: recorded output spatial shape [{}, {}] contradicts the op's \
             target [{target_h}, {target_w}]",
            output_shape[rank - 2],
            output_shape[rank - 1]
        )));
    }
    if in_shape[..rank - 2] != output_shape[..rank - 2] {
        return Err(NyError::InternalError(format!(
            "ResizeBilinear: batch/channel prefix {:?} does not survive to the output {:?}",
            &in_shape[..rank - 2],
            &output_shape[..rank - 2]
        )));
    }

    emit_separable_bilinear_resample(
        name,
        "ResizeBilinear",
        BilinearCoordMap::ResizeHoistedScale,
        &data_input,
        in_shape,
        target_h,
        target_w,
        input_tensors,
        output_tensor,
        ctx,
    )
}

/// Emit the separated `Linear` resampling pair for one bilinear-family node.
///
/// Shared by `ResizeBilinear` and by the interpolating `Upsample2d` modes; the
/// ONLY thing that differs between them is `map`, the source-coordinate
/// expression (see [`BilinearCoordMap`]). The emission geometry — W pass,
/// transpose, H pass, transpose back, identity-axis skipping — is the same
/// because the separability argument is the same.
///
/// An axis whose extent is unchanged contributes an exact identity matrix, so
/// its pass is SKIPPED entirely (`A = I` is provable from every mapping here:
/// `in == out` ⇒ `src = dst` ⇒ `w = 0`). When both axes are unchanged the node
/// lowers to the family's `Add(x, 0)` identity — a no-op node still needs a
/// real producer.
#[allow(clippy::too_many_arguments)]
fn emit_separable_bilinear_resample(
    name: &str,
    op: &str,
    map: BilinearCoordMap,
    data_input: &str,
    in_shape: Vec<usize>,
    target_h: usize,
    target_w: usize,
    input_tensors: &[String],
    output_tensor: &str,
    ctx: &mut Ctx,
) -> Result<NodeOutput> {
    let rank = in_shape.len();
    let (in_h, in_w) = (in_shape[rank - 2], in_shape[rank - 1]);

    // Identity on an axis ⇒ exact identity matrix ⇒ skip that pass.
    let resample_w = in_w != target_w;
    let resample_h = in_h != target_h;
    if !resample_w && !resample_h {
        return ops_core::translate_identity_add_zero(
            name,
            &format!("{op} with unchanged spatial extents"),
            input_tensors,
            output_tensor,
            ctx,
        );
    }

    let mut specs = Vec::with_capacity(4);
    let mut current = data_input.to_string();
    let mut current_shape = in_shape;

    // W pass: Linear over the LAST axis.
    if resample_w {
        let matrix = bilinear_axis_matrix(in_w, target_w, "width", map, op)?;
        // The H pass (if any) needs its own tensor; otherwise this Linear IS
        // the node's output.
        let (spec_name, out_name) = if resample_h {
            (format!("{name}_w"), format!("{name}_w_out"))
        } else {
            (name.to_string(), output_tensor.to_string())
        };
        specs.push(resample_linear_spec(
            &spec_name, matrix, &current, &out_name, ctx,
        )?);
        current_shape[rank - 1] = target_w;
        ctx.tensor_shapes
            .entry(out_name.clone())
            .or_insert(shape_to_i64(&current_shape, &format!("{op} W pass"))?);
        current = out_name;
    }

    // H pass: transpose the H axis into last position, apply, transpose back.
    if resample_h {
        let t_in_name = format!("{name}_ht");
        let t_in_out = format!("{t_in_name}_out");
        specs.push(swap_last_two_spec(&t_in_name, rank, &current, &t_in_out)?);
        current_shape.swap(rank - 2, rank - 1);
        ctx.tensor_shapes
            .entry(t_in_out.clone())
            .or_insert(shape_to_i64(&current_shape, &format!("{op} H transpose"))?);

        let matrix = bilinear_axis_matrix(in_h, target_h, "height", map, op)?;
        let lin_name = format!("{name}_h");
        let lin_out = format!("{lin_name}_out");
        specs.push(resample_linear_spec(
            &lin_name, matrix, &t_in_out, &lin_out, ctx,
        )?);
        current_shape[rank - 1] = target_h;
        ctx.tensor_shapes
            .entry(lin_out.clone())
            .or_insert(shape_to_i64(&current_shape, &format!("{op} H pass"))?);

        specs.push(swap_last_two_spec(name, rank, &lin_out, output_tensor)?);
    }

    Ok(NodeOutput { specs })
}

// ---------------------------------------------------------------------------
// Local spec helpers (mirroring NN's decompose/pad helpers; candidates for
// dedupe into mod.rs once all family ports land)
// ---------------------------------------------------------------------------

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

    /// ReflectionPad1d [a,b,c,d,e] pad(2,1) → a single native Pad (mode
    /// reflect) at trace rank with the preserve-all-axes certificate — no
    /// Slice/Concat decomposition remains.
    #[test]
    fn reflection_pad1d_lowers_to_native_reflect_pad() {
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

        assert_eq!(count(&model, &LayerType::Pad), 1, "one native Pad");
        assert_eq!(
            count(&model, &LayerType::Slice),
            0,
            "no Slice decomposition"
        );
        assert_eq!(count(&model, &LayerType::Concat), 0, "no Concat");

        let pad = find(&model, "layer0_trace_1");
        assert_eq!(pad.layer_type, LayerType::Pad);
        assert_eq!(
            pad.attributes.get("mode"),
            Some(&AttributeValue::String("reflect".to_string()))
        );
        // rank 1: before=[2], after=[1] → pads=[2, 1].
        assert_eq!(
            pad.attributes.get("pads"),
            Some(&AttributeValue::Ints(vec![2, 1]))
        );
        assert_eq!(
            pad.attributes.get(ny_build::PAD_PRESERVE_ALL_AXES_ATTR),
            Some(&AttributeValue::Int(1))
        );
        assert_eq!(
            pad.attributes.get("value"),
            None,
            "reflect mode carries no constant fill value"
        );
        assert_eq!(pad.inputs, vec!["layer0_trace_0_out".to_string()]);
        assert_eq!(pad.outputs, vec!["layer0_trace_1_out".to_string()]);

        build_ok(&model, "reflection pad1d");
    }

    /// ReflectionPad2d lowers to one native Pad with per-axis pads: W (last
    /// axis) gets (left, right), H (second-to-last) gets (top, bottom), and
    /// every leading axis stays (0, 0) under the preserve-all-axes cert.
    #[test]
    fn reflection_pad2d_lowers_to_native_reflect_pad() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[1, 3, 4]),
            node(
                1,
                "pad",
                TraceOp::ReflectionPad2d {
                    pad_left: 2,
                    pad_right: 1,
                    pad_top: 1,
                    pad_bottom: 2,
                },
                &[0],
                &[1, 6, 7],
            ),
        ]);
        let model = translate(&graph).expect("reflection pad 2d translates");

        assert_eq!(count(&model, &LayerType::Pad), 1, "one native Pad");
        let pad = find(&model, "layer0_trace_1");
        assert_eq!(
            pad.attributes.get("mode"),
            Some(&AttributeValue::String("reflect".to_string()))
        );
        // rank 3: before=[0, top=1, left=2], after=[0, bottom=2, right=1].
        assert_eq!(
            pad.attributes.get("pads"),
            Some(&AttributeValue::Ints(vec![0, 1, 2, 0, 2, 1]))
        );
        assert_eq!(
            pad.attributes.get(ny_build::PAD_PRESERVE_ALL_AXES_ATTR),
            Some(&AttributeValue::Int(1))
        );
        assert_eq!(pad.inputs, vec!["layer0_trace_0_out".to_string()]);
        assert_eq!(pad.outputs, vec!["layer0_trace_1_out".to_string()]);

        build_ok(&model, "reflection pad2d");
    }

    /// End-to-end IBP through the built graph network: asymmetric heavy-left
    /// pad (3, 1) on `[1..5]` must reproduce the hand-mirrored PyTorch
    /// reflection `[d,c,b,a,b,c,d,e,d]` exactly on BOTH bounds (reflect-pad
    /// IBP is a pure index copy; the input's `Add(x, 0)` identity is exact).
    #[test]
    fn reflection_pad1d_ibp_matches_hand_mirrored_values() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[5]),
            node(
                1,
                "pad",
                TraceOp::ReflectionPad1d {
                    pad_left: 3,
                    pad_right: 1,
                },
                &[0],
                &[9],
            ),
        ]);
        let model = translate(&graph).expect("reflection pad translates");
        let network = model
            .build_graph_network(ny_build::GraphNetworkOptions::default())
            .expect("reflection pad1d builds");

        let lower = ndarray::arr1(&[1.0_f32, 2.0, 3.0, 4.0, 5.0]).into_dyn();
        let upper = ndarray::arr1(&[1.5_f32, 2.5, 3.5, 4.5, 5.5]).into_dyn();
        let bounds = ny_tensor::BoundedTensor::new(lower, upper).expect("bounds");
        let out = network.propagate_ibp(&bounds).expect("reflection pad IBP");

        assert_eq!(out.shape(), &[9]);
        // pad_left=3 mirrors indices 3,2,1 → [4,3,2]; pad_right=1 mirrors
        // index len-2 = 3 → [4].
        assert_eq!(
            out.lower().as_slice().unwrap(),
            &[4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 4.0]
        );
        assert_eq!(
            out.upper().as_slice().unwrap(),
            &[4.5, 3.5, 2.5, 1.5, 2.5, 3.5, 4.5, 5.5, 4.5]
        );
    }

    /// Rank-2 input, pad only on the last axis: the leading axis passes
    /// through untouched (its (0, 0) pair survives via the preserve cert) and
    /// each row mirrors independently.
    #[test]
    fn reflection_pad1d_ibp_leaves_leading_axis_untouched() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[2, 4]),
            node(
                1,
                "pad",
                TraceOp::ReflectionPad1d {
                    pad_left: 2,
                    pad_right: 1,
                },
                &[0],
                &[2, 7],
            ),
        ]);
        let model = translate(&graph).expect("rank-2 reflection pad translates");
        let network = model
            .build_graph_network(ny_build::GraphNetworkOptions::default())
            .expect("rank-2 reflection pad builds");

        let lower = ndarray::arr2(&[[1.0_f32, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]).into_dyn();
        let upper = lower.mapv(|v| v + 0.5);
        let bounds = ny_tensor::BoundedTensor::new(lower, upper).expect("bounds");
        let out = network.propagate_ibp(&bounds).expect("rank-2 reflect IBP");

        assert_eq!(out.shape(), &[2, 7]);
        // Per row [a,b,c,d] pad(2,1) → [c,b,a,b,c,d,c].
        assert_eq!(
            out.lower().as_slice().unwrap(),
            &[3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 7.0, 6.0, 5.0, 6.0, 7.0, 8.0, 7.0]
        );
        assert_eq!(
            out.upper().as_slice().unwrap(),
            &[3.5, 2.5, 1.5, 2.5, 3.5, 4.5, 3.5, 7.5, 6.5, 5.5, 6.5, 7.5, 8.5, 7.5]
        );
    }

    /// End-to-end IBP for ReflectionPad2d on adversarial asymmetric pads
    /// (left=2, right=1, top=1, bottom=2) against fully hand-mirrored
    /// expected values: rows map to [r1, r0, r1, r2, r1, r0], columns to
    /// [w2, w1, w0, w1, w2, w3, w2].
    #[test]
    fn reflection_pad2d_ibp_matches_hand_mirrored_values() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[1, 3, 4]),
            node(
                1,
                "pad",
                TraceOp::ReflectionPad2d {
                    pad_left: 2,
                    pad_right: 1,
                    pad_top: 1,
                    pad_bottom: 2,
                },
                &[0],
                &[1, 6, 7],
            ),
        ]);
        let model = translate(&graph).expect("reflection pad 2d translates");
        let network = model
            .build_graph_network(ny_build::GraphNetworkOptions::default())
            .expect("reflection pad2d builds");

        // Rows r0=[0..3], r1=[10..13], r2=[20..23]; upper = lower + 0.5.
        let lower = ndarray::ArrayD::from_shape_vec(
            ndarray::IxDyn(&[1, 3, 4]),
            vec![
                0.0_f32, 1.0, 2.0, 3.0, //
                10.0, 11.0, 12.0, 13.0, //
                20.0, 21.0, 22.0, 23.0,
            ],
        )
        .unwrap();
        let upper = lower.mapv(|v| v + 0.5);
        let bounds = ny_tensor::BoundedTensor::new(lower, upper).expect("bounds");
        let out = network
            .propagate_ibp(&bounds)
            .expect("reflection pad2d IBP");

        assert_eq!(out.shape(), &[1, 6, 7]);
        // Column-mirrored rows: rN' = [wN2, wN1, wN0, wN1, wN2, wN3, wN2].
        let r0m = [2.0_f32, 1.0, 0.0, 1.0, 2.0, 3.0, 2.0];
        let r1m = [12.0_f32, 11.0, 10.0, 11.0, 12.0, 13.0, 12.0];
        let r2m = [22.0_f32, 21.0, 20.0, 21.0, 22.0, 23.0, 22.0];
        // Row order: top pad mirrors r1; bottom pad mirrors r1 then r0.
        let expected_lower: Vec<f32> = [r1m, r0m, r1m, r2m, r1m, r0m].concat();
        let expected_upper: Vec<f32> = expected_lower.iter().map(|v| v + 0.5).collect();
        assert_eq!(out.lower().as_slice().unwrap(), expected_lower.as_slice());
        assert_eq!(out.upper().as_slice().unwrap(), expected_upper.as_slice());
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
                    align_corners: false,
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

    /// Independent transcription of nn-core `bilinear_2d_core`
    /// (`nn-core/src/dyn_tensor/ops_ext/spatial_upsample.rs`) over a rank-4
    /// `[1, C, H, W]` f32 buffer, INCLUDING its f32 evaluation order and its
    /// `bilinear_coord` mapping.
    ///
    /// Written straight from the kernel rather than reusing
    /// [`super::bilinear_axis_matrix`] or [`super::upsample_bilinear_coord`],
    /// so agreement below is evidence rather than a tautology. Note the
    /// coordinate expression: `(dst + 0.5) * in / out` — multiply then divide,
    /// NOT the hoisted `scale` [`nn_resize_bilinear`] uses.
    fn nn_upsample_bilinear(
        input: &[f32],
        channels: usize,
        in_h: usize,
        in_w: usize,
        out_h: usize,
        out_w: usize,
        align_corners: bool,
    ) -> Vec<f32> {
        fn coord(dst: usize, in_size: usize, out_size: usize, align_corners: bool) -> f64 {
            let src = if align_corners && out_size > 1 {
                dst as f64 * (in_size as f64 - 1.0) / (out_size as f64 - 1.0)
            } else if align_corners {
                0.0
            } else {
                (dst as f64 + 0.5) * (in_size as f64) / (out_size as f64) - 0.5
            };
            src.clamp(0.0, (in_size - 1) as f64)
        }
        let mut out = Vec::with_capacity(channels * out_h * out_w);
        for batch in input.chunks_exact(in_h * in_w).take(channels) {
            for oh in 0..out_h {
                let src_y = coord(oh, in_h, out_h, align_corners);
                let y0 = (src_y.floor() as usize).min(in_h - 1);
                let y1 = (y0 + 1).min(in_h - 1);
                let wy = (src_y - y0 as f64) as f32;
                for ow in 0..out_w {
                    let src_x = coord(ow, in_w, out_w, align_corners);
                    let x0 = (src_x.floor() as usize).min(in_w - 1);
                    let x1 = (x0 + 1).min(in_w - 1);
                    let wx = (src_x - x0 as f64) as f32;
                    let v00 = batch[y0 * in_w + x0];
                    let v01 = batch[y0 * in_w + x1];
                    let v10 = batch[y1 * in_w + x0];
                    let v11 = batch[y1 * in_w + x1];
                    out.push(
                        v00 * (1.0 - wy) * (1.0 - wx)
                            + v01 * (1.0 - wy) * wx
                            + v10 * wy * (1.0 - wx)
                            + v11 * wy * wx,
                    );
                }
            }
        }
        out
    }

    /// Build + propagate a point box through an `Upsample2d { Bilinear }`
    /// translation, returning `(lower, upper)` of the emitted graph.
    ///
    /// `scale_h`/`scale_w` are recorded as the ratio nn would have produced,
    /// but they never enter the coordinate arithmetic — the geometry comes
    /// from the recorded input and output shapes, exactly as the kernel takes
    /// `in_size`/`out_size`.
    fn upsample_point_propagate(
        in_shape: &[usize],
        out_h: usize,
        out_w: usize,
        align_corners: bool,
        values: &[f32],
    ) -> (Vec<f32>, Vec<f32>) {
        let rank = in_shape.len();
        let mut out_shape = in_shape.to_vec();
        out_shape[rank - 2] = out_h;
        out_shape[rank - 1] = out_w;
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], in_shape),
            node(
                1,
                "up",
                TraceOp::Upsample2d {
                    mode: TraceUpsampleMode::Bilinear,
                    scale_h: out_h as f64 / in_shape[rank - 2] as f64,
                    scale_w: out_w as f64 / in_shape[rank - 1] as f64,
                    align_corners,
                },
                &[0],
                &out_shape,
            ),
        ]);
        let model = translate(&graph).expect("Upsample2d bilinear translates");
        let network = model
            .build_graph_network(ny_build::GraphNetworkOptions::default())
            .expect("Upsample2d bilinear builds");
        let point =
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(in_shape), values.to_vec()).unwrap();
        let bounds = ny_tensor::BoundedTensor::new(point.clone(), point).expect("point bounds");
        let out = network
            .propagate_ibp(&bounds)
            .expect("Upsample2d bilinear IBP");
        assert_eq!(out.shape(), out_shape.as_slice());
        (
            out.lower().iter().copied().collect(),
            out.upper().iter().copied().collect(),
        )
    }

    /// Bilinear Upsample2d LOWERS now — to the separated constant-`Linear`
    /// resampler, not the nearest path's Reshape/Tile decomposition.
    ///
    /// The refusal this replaces was correct while `align_corners` was off the
    /// wire: two different coordinate maps produced identical wire bytes, so
    /// picking either read the wrong source cells for half the traces. The flag
    /// is serialized now, so each node lowers under the mapping it ran.
    #[test]
    fn upsample2d_bilinear_lowers_to_the_separated_resampler() {
        for align_corners in [false, true] {
            let graph = ComputationGraph::from_nodes(vec![
                node(0, "x", TraceOp::Input, &[], &[1, 1, 2, 2]),
                node(
                    1,
                    "up",
                    TraceOp::Upsample2d {
                        mode: TraceUpsampleMode::Bilinear,
                        scale_h: 2.0,
                        scale_w: 2.0,
                        align_corners,
                    },
                    &[0],
                    &[1, 1, 4, 4],
                ),
            ]);
            let model = translate(&graph)
                .unwrap_or_else(|e| panic!("align_corners={align_corners} must lower: {e:?}"));

            // Two Linears (W pass, H pass) + two Transposes; no Tile.
            assert_eq!(
                count(&model, &LayerType::Linear),
                2,
                "align_corners={align_corners}: one resampling Linear per axis"
            );
            assert_eq!(count(&model, &LayerType::Transpose), 2);
            assert_eq!(
                count(&model, &LayerType::Tile),
                0,
                "the nearest path's Tile decomposition must NOT be used"
            );
            build_ok(&model, "upsample2d bilinear");
        }
    }

    /// THE PARITY TEST for the bilinear Upsample2d lowering, at BOTH flag
    /// values.
    ///
    /// For a dense sweep of geometries — integer upscale, non-integer upscale,
    /// downscale, mixed, and the degenerate `in == 1` / `out == 1` axes — nn's
    /// kernel value must lie inside the emitted graph's propagated interval at
    /// a point input, at EVERY output cell. This is what catches a
    /// coordinate-convention error: the wrong mapping shifts sampled cells by
    /// half a pixel or more, which on random data moves the value far outside
    /// a rounding-width interval.
    #[test]
    fn upsample2d_bilinear_encloses_nn_kernel_across_geometries() {
        let geometries = [
            // integer upscale
            (2_usize, 2_usize, 4_usize, 4_usize),
            (3, 3, 6, 6),
            (2, 3, 6, 9),
            (4, 4, 8, 8),
            // non-integer upscale
            (2, 2, 3, 3),
            (3, 4, 7, 5),
            (5, 3, 8, 7),
            (3, 3, 4, 4),
            // downscale
            (4, 4, 2, 2),
            (6, 6, 3, 4),
            (8, 5, 3, 3),
            // mixed / one axis unchanged
            (4, 4, 4, 7),
            (5, 5, 9, 5),
            // degenerate extents (the clamp / out_size == 1 branches)
            (1, 4, 3, 8),
            (4, 1, 8, 3),
            (1, 1, 3, 3),
            (4, 4, 1, 1),
            (3, 3, 1, 5),
        ];
        let channels = 2usize;

        for align_corners in [false, true] {
            for &(in_h, in_w, out_h, out_w) in &geometries {
                for seed in [
                    0x2545_F491_4F6C_DD1Du64,
                    0x9E37_79B9_7F4A_7C15,
                    0x1234_5678_9ABC_DEF1,
                ] {
                    let mut state = seed;
                    let values: Vec<f32> = (0..channels * in_h * in_w)
                        .map(|_| prng(&mut state))
                        .collect();

                    let expected = nn_upsample_bilinear(
                        &values,
                        channels,
                        in_h,
                        in_w,
                        out_h,
                        out_w,
                        align_corners,
                    );
                    let (lower, upper) = upsample_point_propagate(
                        &[1, channels, in_h, in_w],
                        out_h,
                        out_w,
                        align_corners,
                        &values,
                    );
                    assert_eq!(expected.len(), lower.len());

                    for (i, &want) in expected.iter().enumerate() {
                        assert!(
                            lower[i] <= want && want <= upper[i],
                            "align_corners={align_corners} {in_h}x{in_w}->{out_h}x{out_w} \
                             seed={seed:#x} cell {i}: nn value {want} escapes \
                             [{}, {}]",
                            lower[i],
                            upper[i]
                        );
                    }
                }
            }
        }
    }

    /// The parity test above has TEETH: `align_corners` really does select a
    /// different function, so an implementation that ignored the flag (or read
    /// it backwards) would fail it.
    ///
    /// Checked by feeding nn's `align_corners = true` values against the
    /// `false` lowering's interval and requiring a MISS at some cell — if the
    /// two agreed everywhere, the parity sweep would be vacuous.
    #[test]
    fn upsample2d_bilinear_align_corners_is_not_a_cosmetic_flag() {
        let (in_h, in_w, out_h, out_w) = (3usize, 3usize, 5usize, 5usize);
        let channels = 1usize;
        let mut state = 0x51ED_2701_A1B2_C3D4u64;
        let values: Vec<f32> = (0..channels * in_h * in_w)
            .map(|_| prng(&mut state))
            .collect();

        let aligned = nn_upsample_bilinear(&values, channels, in_h, in_w, out_h, out_w, true);
        let half_pixel = nn_upsample_bilinear(&values, channels, in_h, in_w, out_h, out_w, false);
        assert!(
            aligned
                .iter()
                .zip(half_pixel.iter())
                .any(|(a, b)| (a - b).abs() > 1e-3),
            "the two conventions must compute visibly different values, or the \
             parity sweep proves nothing"
        );

        // The half-pixel LOWERING must not enclose the aligned kernel's values.
        let (lower, upper) =
            upsample_point_propagate(&[1, channels, in_h, in_w], out_h, out_w, false, &values);
        assert!(
            aligned
                .iter()
                .enumerate()
                .any(|(i, want)| *want < lower[i] || *want > upper[i]),
            "an align_corners=true kernel value must ESCAPE the align_corners=false \
             lowering's interval — otherwise reading the flag backwards would go \
             undetected"
        );
    }

    /// The emitted coefficients are nn's, bit for bit — not merely close.
    ///
    /// Rebuilds each axis matrix from the independent kernel transcription
    /// (a one-hot input isolates one source cell's contribution, which IS its
    /// matrix entry) and requires exact f32 equality.
    #[test]
    fn upsample2d_bilinear_matrix_entries_are_bit_identical_to_the_nn_kernel() {
        for align_corners in [false, true] {
            let map = if align_corners {
                super::BilinearCoordMap::UpsampleAlignCorners
            } else {
                super::BilinearCoordMap::UpsampleHalfPixel
            };
            for (in_size, out_size) in [
                (2usize, 5usize),
                (5, 2),
                (3, 7),
                (7, 3),
                (4, 4),
                (1, 4),
                (4, 1),
            ] {
                let matrix =
                    super::bilinear_axis_matrix(in_size, out_size, "parity", map, "Upsample2d")
                        .expect("matrix");
                for src in 0..in_size {
                    // One-hot row vector [1, 1, 1, in_size] -> [1, 1, 1, out_size].
                    let mut one_hot = vec![0.0f32; in_size];
                    one_hot[src] = 1.0;
                    let kernel =
                        nn_upsample_bilinear(&one_hot, 1, 1, in_size, 1, out_size, align_corners);
                    for dst in 0..out_size {
                        assert_eq!(
                            matrix[[dst, src]].to_bits(),
                            kernel[dst].to_bits(),
                            "align_corners={align_corners} {in_size}->{out_size} \
                             entry [{dst}, {src}]: {} vs nn {}",
                            matrix[[dst, src]],
                            kernel[dst]
                        );
                    }
                }
            }
        }
    }

    /// Every emitted matrix is row-stochastic with non-negative entries — the
    /// convex-combination property the soundness argument rests on.
    #[test]
    fn upsample2d_bilinear_matrices_are_row_stochastic() {
        for align_corners in [false, true] {
            let map = if align_corners {
                super::BilinearCoordMap::UpsampleAlignCorners
            } else {
                super::BilinearCoordMap::UpsampleHalfPixel
            };
            for (in_size, out_size) in [(2usize, 7usize), (7, 2), (3, 3), (1, 5), (5, 1), (6, 9)] {
                let matrix =
                    super::bilinear_axis_matrix(in_size, out_size, "stochastic", map, "Upsample2d")
                        .expect("matrix");
                for dst in 0..out_size {
                    let mut sum = 0.0f64;
                    for src in 0..in_size {
                        let entry = matrix[[dst, src]];
                        assert!(
                            entry >= 0.0,
                            "align_corners={align_corners}: negative entry {entry} at \
                             [{dst}, {src}]"
                        );
                        sum += f64::from(entry);
                    }
                    assert!(
                        (sum - 1.0).abs() < 1e-6,
                        "align_corners={align_corners} {in_size}->{out_size} row {dst} \
                         sums to {sum}, not 1"
                    );
                }
            }
        }
    }

    /// An axis whose extent is unchanged is an exact identity and its pass is
    /// SKIPPED; when both are unchanged the node lowers to the family's
    /// `Add(x, 0)` identity rather than emitting a no-op Linear pair.
    #[test]
    fn upsample2d_bilinear_identity_axes_are_skipped() {
        // Only the width changes -> one Linear, no transposes.
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[1, 1, 4, 2]),
            node(
                1,
                "up",
                TraceOp::Upsample2d {
                    mode: TraceUpsampleMode::Bilinear,
                    scale_h: 1.0,
                    scale_w: 2.0,
                    align_corners: false,
                },
                &[0],
                &[1, 1, 4, 4],
            ),
        ]);
        let model = translate(&graph).expect("width-only resample lowers");
        assert_eq!(count(&model, &LayerType::Linear), 1, "H pass skipped");
        assert_eq!(count(&model, &LayerType::Transpose), 0);

        // Neither axis changes -> identity, no resampling Linear at all.
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[1, 1, 3, 3]),
            node(
                1,
                "up",
                TraceOp::Upsample2d {
                    mode: TraceUpsampleMode::Bilinear,
                    scale_h: 1.0,
                    scale_w: 1.0,
                    align_corners: true,
                },
                &[0],
                &[1, 1, 3, 3],
            ),
        ]);
        let model = translate(&graph).expect("identity resample lowers");
        assert_eq!(count(&model, &LayerType::Linear), 0, "no resampling needed");
        build_ok(&model, "identity bilinear upsample");
    }

    /// nn has no bicubic kernel at all; the refusal must say so rather than
    /// borrow the bilinear flag-ambiguity reason.
    #[test]
    fn upsample2d_refuses_bicubic_for_the_absent_kernel() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[1, 1, 2, 2]),
            node(
                1,
                "up",
                TraceOp::Upsample2d {
                    mode: TraceUpsampleMode::Bicubic,
                    scale_h: 2.0,
                    scale_w: 2.0,
                    align_corners: false,
                },
                &[0],
                &[1, 1, 4, 4],
            ),
        ]);
        let err = translate(&graph).expect_err("bicubic refused");
        assert!(
            matches!(err, NyError::UnsupportedOp(ref m)
                if m.contains("bicubic") && m.contains("no bicubic kernel")),
            "got {err:?}"
        );
    }

    /// A RUNTIME grid is refused for the structural reason. The message must
    /// say that — a stale "coordinate conventions differ across frameworks"
    /// reason would send a future porter looking for the wrong fix, and would
    /// not explain why `ResizeBilinear` next door IS lowered.
    #[test]
    fn grid_sample_is_refused_for_its_runtime_grid_operand() {
        for align_corners in [false, true] {
            for padding_mode in [GridSamplePaddingMode::Zeros, GridSamplePaddingMode::Border] {
                let graph = ComputationGraph::from_nodes(vec![
                    node(0, "x", TraceOp::Input, &[], &[1, 1, 4, 4]),
                    // A computed grid — deformable attention's actual shape.
                    node(1, "grid", TraceOp::Relu, &[0], &[1, 4, 4, 2]),
                    node(
                        2,
                        "gs",
                        TraceOp::GridSample {
                            padding_mode,
                            align_corners,
                        },
                        &[0, 1],
                        &[1, 1, 4, 4],
                    ),
                ]);
                let err = translate(&graph).expect_err("GridSample must fail closed");
                assert!(
                    matches!(err, NyError::UnsupportedOp(ref m)
                        if m.contains("GridSample")
                            && m.contains("runtime tensor operand")
                            && m.contains("value-dependent")
                            && m.contains("Gather/IndexSelect")
                            && m.contains("conventions themselves are")),
                    "refusal must name the runtime-grid structural reason, got {err:?}"
                );
            }
        }
    }

    /// A CONSTANT grid is refused too — and the message must NOT claim the
    /// grid is a runtime operand, because here it demonstrably is not. The old
    /// unconditional message did exactly that, which is what made the deleted
    /// carve-out comment read like a to-do rather than a decision.
    #[test]
    fn grid_sample_with_a_constant_grid_is_refused_on_arithmetic_grounds() {
        for align_corners in [false, true] {
            for padding_mode in [GridSamplePaddingMode::Zeros, GridSamplePaddingMode::Border] {
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
                            padding_mode,
                            align_corners,
                        },
                        &[0, 1],
                        &[1, 1, 4, 4],
                    ),
                ]);
                let err = translate(&graph).expect_err("a constant grid must still fail closed");
                let NyError::UnsupportedOp(message) = &err else {
                    panic!("expected UnsupportedOp, got {err:?}");
                };
                assert!(
                    message.contains("ARITHMETIC grounds")
                        && message.contains("not separable")
                        && message.contains("2 ULP"),
                    "the constant-grid refusal must give the arithmetic reason: {message}"
                );
                assert!(
                    !message.contains("is a runtime"),
                    "and must NOT claim this grid is a runtime operand: {message}"
                );
            }
        }
    }

    /// THE evidence for the constant-grid refusal, as a runnable witness
    /// rather than an assertion in a comment.
    ///
    /// A single constant resampling matrix — the only form a non-separable
    /// warp admits — must fold `cy * cx` into one coefficient. nn's kernel
    /// instead evaluates `fl(fl(v * cy) * cx)`. This test exhibits a concrete
    /// `(v, cy, cx)` where nn's value lies OUTSIDE the tightest interval any
    /// sound `Linear` layer could emit for `v * fl(cy*cx)` — the exact product
    /// rounded outward one step each way. That is a silent non-enclosure, so
    /// the matrix lowering is refused rather than shipped and widened later.
    ///
    /// The fixture is deliberately the degenerate single-corner case (the
    /// other three corner values are 0, contributing exact zeros to nn's
    /// 3-add chain): with no cancellation and no accumulation slack, the
    /// interval is at its tightest and the gap is unambiguous.
    #[test]
    fn grid_sample_single_matrix_lowering_would_not_enclose_nn() {
        // Witness found by search over f32 triples; pinned so the argument is
        // reproducible rather than re-derived.
        let v = f32::from_bits(0xbf9f_5640); // -1.2448196
        let cy = 2.3004788e-1_f32;
        let cx = 5.488435e-1_f32;

        // nn-core `bilinear_sample`: scale by the y weight, THEN the x weight.
        let nn_value = v * cy * cx;
        // A matrix must carry the pre-multiplied coefficient.
        let coefficient = cy * cx;

        // The tightest SOUND interval for one product: the exact real product,
        // rounded outward to f32. Any correct interval layer is at least this
        // wide, and none is narrower.
        let exact = f64::from(v) * f64::from(coefficient);
        let nearest = exact as f32;
        let lower = if f64::from(nearest) > exact {
            next_toward_neg_inf(nearest)
        } else {
            nearest
        };
        let upper = if f64::from(nearest) < exact {
            next_toward_pos_inf(nearest)
        } else {
            nearest
        };
        assert!(lower <= upper, "the outward rounding must not invert");

        assert!(
            nn_value < lower || nn_value > upper,
            "the witness must actually escape: nn={nn_value:e} in [{lower:e}, {upper:e}]"
        );
        // ...and by more than the interval itself spans, which is the point:
        // widening by a fixed ULP is not enough, the gap scales with the value.
        assert_ne!(
            nn_value.to_bits(),
            nearest.to_bits(),
            "nn's association differs from the matrix's at the bit level"
        );
    }

    fn next_toward_neg_inf(x: f32) -> f32 {
        if x > 0.0 {
            f32::from_bits(x.to_bits() - 1)
        } else {
            f32::from_bits(x.to_bits() + 1)
        }
    }

    fn next_toward_pos_inf(x: f32) -> f32 {
        if x >= 0.0 {
            f32::from_bits(x.to_bits() + 1)
        } else {
            f32::from_bits(x.to_bits() - 1)
        }
    }

    // -----------------------------------------------------------------------
    // ResizeBilinear: nn-kernel parity
    // -----------------------------------------------------------------------

    /// Independent transcription of nn-core `resize_bilinear_compute`
    /// (`nn-core/src/dyn_tensor/ops_ext/spatial_upsample.rs`) over a rank-3
    /// `[C, H, W]` f32 buffer, INCLUDING its f32 evaluation order:
    ///
    /// ```text
    /// v00*(1-wy)*(1-wx) + v01*(1-wy)*wx + v10*wy*(1-wx) + v11*wy*wx
    /// ```
    ///
    /// This is the oracle every parity assertion below is measured against.
    /// It is written straight from the kernel rather than reusing
    /// [`super::bilinear_axis_matrix`], so agreement is evidence, not a
    /// tautology.
    fn nn_resize_bilinear(
        input: &[f32],
        channels: usize,
        in_h: usize,
        in_w: usize,
        out_h: usize,
        out_w: usize,
    ) -> Vec<f32> {
        fn coord(dst: usize, scale: f64, in_size: usize) -> f64 {
            let src = (dst as f64 + 0.5) * scale - 0.5;
            src.clamp(0.0, (in_size - 1) as f64)
        }
        let scale_y = in_h as f64 / out_h as f64;
        let scale_x = in_w as f64 / out_w as f64;
        let mut out = Vec::with_capacity(channels * out_h * out_w);
        for batch in input.chunks_exact(in_h * in_w).take(channels) {
            for oh in 0..out_h {
                let src_y = coord(oh, scale_y, in_h);
                let y0 = src_y.floor() as usize;
                let y1 = (y0 + 1).min(in_h - 1);
                let y0 = y0.min(in_h - 1);
                let wy = (src_y - y0 as f64) as f32;
                for ow in 0..out_w {
                    let src_x = coord(ow, scale_x, in_w);
                    let x0 = src_x.floor() as usize;
                    let x1 = (x0 + 1).min(in_w - 1);
                    let x0 = x0.min(in_w - 1);
                    let wx = (src_x - x0 as f64) as f32;
                    let v00 = batch[y0 * in_w + x0];
                    let v01 = batch[y0 * in_w + x1];
                    let v10 = batch[y1 * in_w + x0];
                    let v11 = batch[y1 * in_w + x1];
                    out.push(
                        v00 * (1.0 - wy) * (1.0 - wx)
                            + v01 * (1.0 - wy) * wx
                            + v10 * wy * (1.0 - wx)
                            + v11 * wy * wx,
                    );
                }
            }
        }
        out
    }

    /// Deterministic xorshift PRNG — reproducible adversarial inputs without a
    /// dev-dependency.
    fn prng(state: &mut u64) -> f32 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        // Values in [-8, 8) with plenty of ULP structure (16 mantissa-ish
        // bits, so the interpolation weights land on non-representable
        // products rather than tidy dyadics).
        ((((*state >> 40) & 0xFFFF) as f32) / 4096.0) - 8.0
    }

    /// Build + propagate a point (degenerate) box through a ResizeBilinear
    /// translation, returning `(lower, upper)` of the emitted graph.
    fn resize_point_propagate(
        in_shape: &[usize],
        target_h: usize,
        target_w: usize,
        values: &[f32],
    ) -> (Vec<f32>, Vec<f32>) {
        let mut out_shape = in_shape.to_vec();
        let rank = out_shape.len();
        out_shape[rank - 2] = target_h;
        out_shape[rank - 1] = target_w;
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], in_shape),
            node(
                1,
                "rb",
                TraceOp::ResizeBilinear { target_h, target_w },
                &[0],
                &out_shape,
            ),
        ]);
        let model = translate(&graph).expect("ResizeBilinear translates");
        let network = model
            .build_graph_network(ny_build::GraphNetworkOptions::default())
            .expect("ResizeBilinear builds");
        let point =
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(in_shape), values.to_vec()).unwrap();
        let bounds = ny_tensor::BoundedTensor::new(point.clone(), point).expect("point bounds");
        let out = network.propagate_ibp(&bounds).expect("ResizeBilinear IBP");
        assert_eq!(out.shape(), out_shape.as_slice());
        (
            out.lower().iter().copied().collect(),
            out.upper().iter().copied().collect(),
        )
    }

    /// THE parity test. For a dense sweep of scale factors — integer upscale,
    /// non-integer upscale, downscale, mixed, and the degenerate `in == 1`
    /// axis — nn's kernel value must lie inside the emitted graph's propagated
    /// interval at a point input, at EVERY output cell.
    ///
    /// This is the assertion that catches a coordinate-convention error: an
    /// `align_corners` or asymmetric mapping shifts the sampled cells by half
    /// a pixel or more, which on random data moves the value far outside a
    /// rounding-width interval. It is checked at 21 geometries x 3 seeds.
    #[test]
    fn resize_bilinear_encloses_nn_kernel_across_scale_factors() {
        let geometries = [
            // integer upscale
            (2_usize, 2_usize, 4_usize, 4_usize),
            (3, 3, 6, 6),
            (2, 3, 6, 9),
            (4, 4, 8, 8),
            // non-integer upscale
            (2, 2, 3, 3),
            (3, 4, 7, 5),
            (5, 3, 8, 7),
            (3, 3, 4, 4),
            (4, 5, 6, 7),
            // downscale
            (8, 8, 4, 4),
            (7, 5, 3, 2),
            (6, 6, 4, 4),
            (9, 4, 2, 3),
            // identity on one axis only
            (4, 5, 4, 9),
            (4, 5, 7, 5),
            // degenerate source extents
            (1, 4, 3, 6),
            (4, 1, 6, 3),
            (1, 1, 3, 3),
            // degenerate target extents
            (5, 5, 1, 1),
            (6, 3, 1, 4),
            (3, 6, 4, 1),
        ];
        for (in_h, in_w, out_h, out_w) in geometries {
            for seed in [0x2026_0826_u64, 0xdead_beef, 0x0123_4567_89ab_cdef] {
                let channels = 2;
                let mut state = seed;
                let values: Vec<f32> = (0..channels * in_h * in_w)
                    .map(|_| prng(&mut state))
                    .collect();
                let expected = nn_resize_bilinear(&values, channels, in_h, in_w, out_h, out_w);
                let (lower, upper) =
                    resize_point_propagate(&[channels, in_h, in_w], out_h, out_w, &values);

                assert_eq!(expected.len(), lower.len());
                for (idx, &want) in expected.iter().enumerate() {
                    assert!(
                        lower[idx] <= want && want <= upper[idx],
                        "geometry {in_h}x{in_w}->{out_h}x{out_w} seed {seed:#x} cell {idx}: \
                         nn kernel {want} outside [{}, {}]",
                        lower[idx],
                        upper[idx]
                    );
                }
            }
        }
    }

    /// Same parity, rank 4 (`[N, C, H, W]` — nn's other accepted rank), so
    /// the transpose perm and the batch/channel prefix are exercised too.
    #[test]
    fn resize_bilinear_encloses_nn_kernel_at_rank_4() {
        let (n, c, in_h, in_w, out_h, out_w) =
            (2_usize, 3_usize, 4_usize, 3_usize, 6_usize, 7_usize);
        let mut state = 0xfeed_face_u64;
        let values: Vec<f32> = (0..n * c * in_h * in_w).map(|_| prng(&mut state)).collect();
        // nn's kernel is per-(batch x channel) plane, so N*C planes.
        let expected = nn_resize_bilinear(&values, n * c, in_h, in_w, out_h, out_w);
        let (lower, upper) = resize_point_propagate(&[n, c, in_h, in_w], out_h, out_w, &values);
        assert_eq!(expected.len(), lower.len());
        for (idx, &want) in expected.iter().enumerate() {
            assert!(
                lower[idx] <= want && want <= upper[idx],
                "rank-4 cell {idx}: nn kernel {want} outside [{}, {}]",
                lower[idx],
                upper[idx]
            );
        }
    }

    /// The enclosure above must be TIGHT, not vacuous: a lowering that read
    /// the wrong cells but produced a huge interval would still "enclose".
    /// Pin the interval width to a rounding scale — the propagated width must
    /// stay far below the spread of the data itself.
    #[test]
    fn resize_bilinear_enclosure_is_rounding_tight_not_vacuous() {
        let (in_h, in_w, out_h, out_w) = (5_usize, 7_usize, 9_usize, 4_usize);
        let mut state = 0x5eed_1234_u64;
        let values: Vec<f32> = (0..in_h * in_w).map(|_| prng(&mut state)).collect();
        let (lower, upper) = resize_point_propagate(&[1, in_h, in_w], out_h, out_w, &values);
        for (lo, hi) in lower.iter().zip(&upper) {
            let width = hi - lo;
            assert!(
                width < 1e-3,
                "point-input interval [{lo}, {hi}] (width {width}) is not rounding-tight"
            );
        }
    }

    /// Convention falsification: assert the lowering is NOT `align_corners`
    /// and NOT the naive asymmetric `src = dst * in / out` map. Both are the
    /// classic silent-unsoundness substitutions; on a 2->4 upscale of
    /// `[0, 3]` the three conventions give visibly different values, so a
    /// single cell separates them.
    ///
    /// ```text
    /// half-pixel (nn):     src = (dst+0.5)*0.5 - 0.5 -> [0, 0.25, 0.75, 1] clamped
    /// align_corners:       src = dst*(2-1)/(4-1)     -> [0, 1/3, 2/3, 1]
    /// asymmetric:          src = dst*2/4             -> [0, 0.5, 1, 1.5]->clamp
    /// ```
    #[test]
    fn resize_bilinear_uses_half_pixel_not_align_corners_or_asymmetric() {
        // Row [0, 3] upscaled to 4 columns; the row is duplicated so the H
        // axis is an exact identity and only the W convention is under test.
        let values = vec![0.0_f32, 3.0, 0.0, 3.0];
        let (lower, upper) = resize_point_propagate(&[1, 2, 2], 2, 4, &values);

        // nn half-pixel: src = [0, 0.25, 0.75, 1] (0 and 1 by clamping)
        //   -> [0, 0.75, 2.25, 3]
        let half_pixel = [0.0_f32, 0.75, 2.25, 3.0];
        // align_corners would give src = [0, 1/3, 2/3, 1] -> [0, 1, 2, 3]
        let align_corners = [0.0_f32, 1.0, 2.0, 3.0];
        // asymmetric would give src = [0, 0.5, 1, 1(clamped)] -> [0, 1.5, 3, 3]
        let asymmetric = [0.0_f32, 1.5, 3.0, 3.0];

        for col in 0..4 {
            assert!(
                lower[col] <= half_pixel[col] && half_pixel[col] <= upper[col],
                "col {col}: half-pixel {} outside [{}, {}]",
                half_pixel[col],
                lower[col],
                upper[col]
            );
        }
        // Columns 1 and 2 separate all three conventions.
        for col in [1_usize, 2] {
            assert!(
                align_corners[col] < lower[col] || align_corners[col] > upper[col],
                "col {col}: align_corners value {} is NOT excluded by [{}, {}]",
                align_corners[col],
                lower[col],
                upper[col]
            );
            assert!(
                asymmetric[col] < lower[col] || asymmetric[col] > upper[col],
                "col {col}: asymmetric value {} is NOT excluded by [{}, {}]",
                asymmetric[col],
                lower[col],
                upper[col]
            );
        }
    }

    /// A genuine interval input must still enclose nn's kernel evaluated at
    /// every vertex it contains — spot-checked at the two corners and the
    /// midpoint of a 4-D box. Interpolation is a convex combination, so the
    /// output hull is also inside the input hull.
    #[test]
    fn resize_bilinear_encloses_the_kernel_over_a_whole_box() {
        let (in_h, in_w, out_h, out_w) = (3_usize, 3_usize, 5_usize, 4_usize);
        let mut state = 0x00a1_1ce0_u64;
        let lo_in: Vec<f32> = (0..in_h * in_w).map(|_| prng(&mut state)).collect();
        let hi_in: Vec<f32> = lo_in.iter().map(|v| v + 0.5).collect();

        let out_shape = vec![1, out_h, out_w];
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[1, in_h, in_w]),
            node(
                1,
                "rb",
                TraceOp::ResizeBilinear {
                    target_h: out_h,
                    target_w: out_w,
                },
                &[0],
                &out_shape,
            ),
        ]);
        let model = translate(&graph).expect("ResizeBilinear translates");
        let network = model
            .build_graph_network(ny_build::GraphNetworkOptions::default())
            .expect("builds");
        let shape = ndarray::IxDyn(&[1, in_h, in_w]);
        let bounds = ny_tensor::BoundedTensor::new(
            ndarray::ArrayD::from_shape_vec(shape.clone(), lo_in.clone()).unwrap(),
            ndarray::ArrayD::from_shape_vec(shape, hi_in.clone()).unwrap(),
        )
        .expect("box bounds");
        let out = network.propagate_ibp(&bounds).expect("box IBP");
        let out_lo: Vec<f32> = out.lower().iter().copied().collect();
        let out_hi: Vec<f32> = out.upper().iter().copied().collect();

        let mid: Vec<f32> = lo_in
            .iter()
            .zip(&hi_in)
            .map(|(l, h)| 0.5 * (l + h))
            .collect();
        for sample in [&lo_in, &hi_in, &mid] {
            let want = nn_resize_bilinear(sample, 1, in_h, in_w, out_h, out_w);
            for (idx, &value) in want.iter().enumerate() {
                assert!(
                    out_lo[idx] <= value && value <= out_hi[idx],
                    "vertex sample cell {idx}: {value} outside [{}, {}]",
                    out_lo[idx],
                    out_hi[idx]
                );
            }
        }
        // And the output hull is inside the input hull (convex combination).
        let in_min = lo_in.iter().copied().fold(f32::INFINITY, f32::min);
        let in_max = hi_in.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        for (lo, hi) in out_lo.iter().zip(&out_hi) {
            assert!(
                *lo >= in_min - 1e-4 && *hi <= in_max + 1e-4,
                "output [{lo}, {hi}] escapes the input hull [{in_min}, {in_max}]"
            );
        }
    }

    /// Structure pin: W pass, transpose, H pass, transpose back — with the H
    /// matrix carrying nn's half-pixel weights for a 2->3 upscale.
    #[test]
    fn resize_bilinear_emits_separated_axis_linears() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[1, 2, 2, 2]),
            node(
                1,
                "rb",
                TraceOp::ResizeBilinear {
                    target_h: 3,
                    target_w: 4,
                },
                &[0],
                &[1, 2, 3, 4],
            ),
        ]);
        let model = translate(&graph).expect("translates");
        assert_eq!(count(&model, &LayerType::Linear), 2, "one Linear per axis");
        assert_eq!(count(&model, &LayerType::Transpose), 2, "in and back");

        let w_pass = find(&model, "layer0_trace_1_w");
        assert_eq!(w_pass.layer_type, LayerType::Linear);
        assert_eq!(
            w_pass.attributes.get("transB"),
            Some(&AttributeValue::Int(1))
        );
        let h_pass = find(&model, "layer0_trace_1_h");
        assert_eq!(h_pass.layer_type, LayerType::Linear);

        // Final spec is the transpose back, named for the node.
        let last = find(&model, "layer0_trace_1");
        assert_eq!(last.layer_type, LayerType::Transpose);
        assert_eq!(
            last.attributes.get("perm"),
            Some(&AttributeValue::Ints(vec![0, 1, 3, 2]))
        );
        assert_eq!(last.outputs, vec!["layer0_trace_1_out".to_string()]);

        // H matrix, 2 -> 3: scale = 2/3, src = (d + 0.5)*2/3 - 0.5
        //   d=0 -> -1/6 -> clamped 0        -> [1, 0]
        //   d=1 -> 0.5                      -> [0.5, 0.5]
        //   d=2 -> 7/6 -> clamped 1         -> [0, 1]
        let matrix = super::bilinear_axis_matrix(
            2,
            3,
            "height",
            super::BilinearCoordMap::ResizeHoistedScale,
            "ResizeBilinear",
        )
        .expect("matrix");
        assert_eq!(matrix.dim(), (3, 2));
        assert_eq!(matrix[[0, 0]], 1.0);
        assert_eq!(matrix[[0, 1]], 0.0);
        assert_eq!(matrix[[1, 0]], 0.5);
        assert_eq!(matrix[[1, 1]], 0.5);
        assert_eq!(matrix[[2, 0]], 0.0);
        assert_eq!(matrix[[2, 1]], 1.0);

        build_ok(&model, "resize bilinear");
    }

    /// BIT-TOLERANCE 0 on the thing a convention error corrupts: the source
    /// cells and their interpolation weights.
    ///
    /// For a dense geometry sweep, re-derive nn's per-cell `(i0, i1, w)` from
    /// the kernel's own expression and require the emitted matrix row to hold
    /// EXACTLY those entries, compared on raw f32 bits. Nothing here is
    /// tolerant: a half-pixel-vs-align_corners slip, an off-by-one in the
    /// `floor`/`min` clamp, or a coefficient rounded through a `cy*cx`
    /// product all show up as a bit mismatch.
    #[test]
    fn resize_bilinear_matrix_entries_are_bit_identical_to_the_nn_kernel() {
        for in_size in 1_usize..=17 {
            for out_size in 1_usize..=17 {
                let matrix = super::bilinear_axis_matrix(
                    in_size,
                    out_size,
                    "bitparity",
                    super::BilinearCoordMap::ResizeHoistedScale,
                    "ResizeBilinear",
                )
                .expect("matrix");
                // nn: scale hoisted out of the loop, then per output cell.
                let scale = in_size as f64 / out_size as f64;
                for dst in 0..out_size {
                    let src = ((dst as f64 + 0.5) * scale - 0.5).clamp(0.0, (in_size - 1) as f64);
                    let i0_raw = src.floor() as usize;
                    let i1 = (i0_raw + 1).min(in_size - 1);
                    let i0 = i0_raw.min(in_size - 1);
                    let w = (src - i0 as f64) as f32;

                    let mut expected = vec![0.0_f32; in_size];
                    expected[i0] += 1.0_f32 - w;
                    expected[i1] += w;

                    for (col, want) in expected.iter().enumerate() {
                        let got = matrix[[dst, col]];
                        assert_eq!(
                            got.to_bits(),
                            want.to_bits(),
                            "{in_size}->{out_size} cell {dst} col {col}: emitted {got:?} \
                             (bits {:#x}) != nn {want:?} (bits {:#x})",
                            got.to_bits(),
                            want.to_bits()
                        );
                    }
                }
            }
        }
    }

    /// Where nn's own f32 evaluation is EXACT (dyadic weights, dyadic data),
    /// the emitted graph's interval must pin nn's value to within the
    /// MANDATORY outward rounding the propagator itself performs — one
    /// `next_down_f64`/`next_up_f64` per accumulated term plus one
    /// `next_down_f32`/`next_up_f32` on each pass's f64->f32 narrowing. The
    /// scale of that slack is set by the magnitude of the values being
    /// accumulated, not by the (possibly tiny) output, so the budget is
    /// expressed relative to `max |input|`.
    ///
    /// Geometry 2->4 gives weights `{1, 0.75/0.25, 0.25/0.75, 1}` and the data
    /// is quarter-integers, so every product and sum in nn's kernel is exact
    /// in f32: the ONLY separation left is the propagator's own rounding, and
    /// this test pins that the lowering adds nothing on top of it.
    #[test]
    fn resize_bilinear_is_ulp_exact_where_the_kernel_arithmetic_is_exact() {
        let values: Vec<f32> = vec![
            1.0, 2.25, //
            -3.5, 0.75,
        ];
        let scale = values.iter().fold(0.0_f32, |acc, v| acc.max(v.abs()));
        // Two chained Linear passes; allow a few directed steps each.
        let budget = 8.0 * f32::EPSILON * scale;

        let (lower, upper) = resize_point_propagate(&[1, 2, 2], 4, 4, &values);
        let expected = nn_resize_bilinear(&values, 1, 2, 2, 4, 4);
        for (idx, &want) in expected.iter().enumerate() {
            let (lo, hi) = (lower[idx], upper[idx]);
            assert!(
                lo <= want && want <= hi,
                "cell {idx}: {want} outside [{lo}, {hi}]"
            );
            assert!(
                want - lo <= budget && hi - want <= budget,
                "cell {idx}: [{lo}, {hi}] straddles nn's exact {want} by more than the \
                 propagator's own rounding budget {budget}"
            );
        }
    }

    /// Every emitted axis matrix is row-stochastic with non-negative entries
    /// — the property that makes the map a convex combination of source cells
    /// (hence range-preserving), swept over a wide geometry grid.
    #[test]
    fn resize_bilinear_matrices_are_row_stochastic() {
        for in_size in 1_usize..=12 {
            for out_size in 1_usize..=12 {
                let matrix = super::bilinear_axis_matrix(
                    in_size,
                    out_size,
                    "sweep",
                    super::BilinearCoordMap::ResizeHoistedScale,
                    "ResizeBilinear",
                )
                .expect("matrix");
                assert_eq!(matrix.dim(), (out_size, in_size));
                for row in matrix.rows() {
                    let mut sum = 0.0_f64;
                    let mut nonzeros = 0;
                    for &entry in row {
                        assert!(
                            (0.0..=1.0).contains(&entry),
                            "entry {entry} outside [0, 1] for {in_size}->{out_size}"
                        );
                        if entry != 0.0 {
                            nonzeros += 1;
                        }
                        sum += f64::from(entry);
                    }
                    assert!(
                        (sum - 1.0).abs() <= 1e-6,
                        "row sums to {sum} for {in_size}->{out_size}"
                    );
                    assert!(
                        nonzeros <= 2,
                        "bilinear touches at most 2 source cells per axis, got {nonzeros}"
                    );
                }
            }
        }
    }

    /// An `in == out` axis is an exact identity, so its pass is skipped and
    /// the emitted matrix (when built) is the identity — checked both ways.
    #[test]
    fn resize_bilinear_identity_axis_is_exact_and_skipped() {
        for size in 1_usize..=8 {
            let matrix = super::bilinear_axis_matrix(
                size,
                size,
                "identity",
                super::BilinearCoordMap::ResizeHoistedScale,
                "ResizeBilinear",
            )
            .expect("matrix");
            for r in 0..size {
                for c in 0..size {
                    let want = if r == c { 1.0 } else { 0.0 };
                    assert_eq!(matrix[[r, c]], want, "identity {size} at [{r}, {c}]");
                }
            }
        }

        // Only H changes -> exactly one Linear and two Transposes, no W pass.
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[1, 2, 5]),
            node(
                1,
                "rb",
                TraceOp::ResizeBilinear {
                    target_h: 4,
                    target_w: 5,
                },
                &[0],
                &[1, 4, 5],
            ),
        ]);
        let model = translate(&graph).expect("translates");
        assert_eq!(count(&model, &LayerType::Linear), 1, "H pass only");
        assert_eq!(count(&model, &LayerType::Transpose), 2);
        build_ok(&model, "H-only resize");

        // Only W changes -> a single Linear, no Transpose at all.
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[1, 4, 2]),
            node(
                1,
                "rb",
                TraceOp::ResizeBilinear {
                    target_h: 4,
                    target_w: 5,
                },
                &[0],
                &[1, 4, 5],
            ),
        ]);
        let model = translate(&graph).expect("translates");
        assert_eq!(count(&model, &LayerType::Linear), 1, "W pass only");
        assert_eq!(count(&model, &LayerType::Transpose), 0);
        assert_eq!(find(&model, "layer0_trace_1").layer_type, LayerType::Linear);
        build_ok(&model, "W-only resize");
    }

    /// A wholly unchanged resize is nn's own short-circuit; the bridge still
    /// needs a real producer for the node's output tensor.
    #[test]
    fn resize_bilinear_no_op_emits_identity_output() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[1, 3, 4]),
            node(
                1,
                "rb",
                TraceOp::ResizeBilinear {
                    target_h: 3,
                    target_w: 4,
                },
                &[0],
                &[1, 3, 4],
            ),
        ]);
        let model = translate(&graph).expect("translates");
        let spec = find(&model, "layer0_trace_1");
        assert_eq!(spec.layer_type, LayerType::Add);
        assert_eq!(count(&model, &LayerType::Linear), 0);
        build_ok(&model, "no-op resize");
    }

    /// Adversarial refusals: ranks nn's kernel never produces, zero extents,
    /// a shape that contradicts the recorded target, and a matrix past the
    /// explosion cap. Every one must fail closed.
    #[test]
    fn resize_bilinear_refuses_geometries_nn_never_records() {
        // rank 2 (nn's resize_bilinear demands rank 3 or 4)
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[2, 2]),
            node(
                1,
                "rb",
                TraceOp::ResizeBilinear {
                    target_h: 4,
                    target_w: 4,
                },
                &[0],
                &[4, 4],
            ),
        ]);
        let err = translate(&graph).expect_err("rank-2 refused");
        assert!(
            matches!(err, NyError::UnsupportedOp(ref m)
                if m.contains("rank-3 or rank-4")),
            "got {err:?}"
        );

        // rank 5
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[1, 1, 1, 2, 2]),
            node(
                1,
                "rb",
                TraceOp::ResizeBilinear {
                    target_h: 4,
                    target_w: 4,
                },
                &[0],
                &[1, 1, 1, 4, 4],
            ),
        ]);
        let err = translate(&graph).expect_err("rank-5 refused");
        assert!(
            matches!(err, NyError::UnsupportedOp(ref m) if m.contains("rank-3 or rank-4")),
            "got {err:?}"
        );

        // zero target extent
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[1, 2, 2]),
            node(
                1,
                "rb",
                TraceOp::ResizeBilinear {
                    target_h: 0,
                    target_w: 4,
                },
                &[0],
                &[1, 0, 4],
            ),
        ]);
        let err = translate(&graph).expect_err("zero target refused");
        assert!(
            matches!(err, NyError::UnsupportedOp(ref m) if m.contains("target dimensions")),
            "got {err:?}"
        );

        // recorded output shape contradicts the op's target
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
                &[1, 4, 5],
            ),
        ]);
        let err = translate(&graph).expect_err("shape/target mismatch refused");
        assert!(
            matches!(err, NyError::InternalError(ref m) if m.contains("contradicts")),
            "got {err:?}"
        );

        // batch/channel prefix must survive
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[3, 2, 2]),
            node(
                1,
                "rb",
                TraceOp::ResizeBilinear {
                    target_h: 4,
                    target_w: 4,
                },
                &[0],
                &[5, 4, 4],
            ),
        ]);
        let err = translate(&graph).expect_err("prefix mismatch refused");
        assert!(
            matches!(err, NyError::InternalError(ref m) if m.contains("prefix")),
            "got {err:?}"
        );

        // matrix explosion cap
        let err = super::bilinear_axis_matrix(
            4096,
            8192,
            "width",
            super::BilinearCoordMap::ResizeHoistedScale,
            "ResizeBilinear",
        )
        .expect_err("oversized matrix refused");
        assert!(
            matches!(err, NyError::UnsupportedOp(ref m) if m.contains("exceeds limit")),
            "got {err:?}"
        );
    }

    /// Reflection with `pad >= input length` has no interior element to
    /// mirror: refusal pin for the 1d arm (PyTorch rejects the same config;
    /// PadLayer would refuse the same geometry at build time).
    #[test]
    fn reflection_pad1d_refuses_pad_not_smaller_than_input() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[3]),
            node(
                1,
                "pad",
                TraceOp::ReflectionPad1d {
                    pad_left: 3,
                    pad_right: 0,
                },
                &[0],
                &[6],
            ),
        ]);
        let err = translate(&graph).expect_err("pad >= input length refused");
        assert!(
            matches!(err, NyError::UnsupportedOp(ref m)
                if m.contains("pad_left") && m.contains("smaller than the input length")),
            "got {err:?}"
        );
    }

    /// Same refusal pin on the 2d H axis: pad_top == input height.
    #[test]
    fn reflection_pad2d_refuses_pad_not_smaller_than_input() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[1, 3, 4]),
            node(
                1,
                "pad",
                TraceOp::ReflectionPad2d {
                    pad_left: 0,
                    pad_right: 0,
                    pad_top: 3,
                    pad_bottom: 0,
                },
                &[0],
                &[1, 6, 4],
            ),
        ]);
        let err = translate(&graph).expect_err("pad_top >= input height refused");
        assert!(
            matches!(err, NyError::UnsupportedOp(ref m)
                if m.contains("pad_top") && m.contains("smaller than the input length")),
            "got {err:?}"
        );
    }

    /// ReflectionPad2d keeps the family's per-axis MAX_PAD_SIZE cap.
    #[test]
    fn reflection_pad2d_refuses_oversized_pad() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[1, 3, 500]),
            node(
                1,
                "pad",
                TraceOp::ReflectionPad2d {
                    pad_left: 200,
                    pad_right: 100,
                    pad_top: 1,
                    pad_bottom: 1,
                },
                &[0],
                &[1, 5, 800],
            ),
        ]);
        let err = translate(&graph).expect_err("oversized 2d pad refused");
        assert!(
            matches!(err, NyError::UnsupportedOp(ref m) if m.contains("exceeds limit")),
            "got {err:?}"
        );
    }

    /// ReflectionPad2d needs two trailing spatial axes.
    #[test]
    fn reflection_pad2d_refuses_rank_below_two() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[4]),
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
                &[8],
            ),
        ]);
        let err = translate(&graph).expect_err("rank-1 output refused");
        assert!(
            matches!(err, NyError::UnsupportedOp(ref m)
                if m.contains("ReflectionPad2d") && m.contains("rank")),
            "got {err:?}"
        );
    }

    #[test]
    fn zero_reflection_pad2d_emits_identity_output() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[1, 3, 4]),
            node(
                1,
                "pad",
                TraceOp::ReflectionPad2d {
                    pad_left: 0,
                    pad_right: 0,
                    pad_top: 0,
                    pad_bottom: 0,
                },
                &[0],
                &[1, 3, 4],
            ),
        ]);
        let model = translate(&graph).expect("zero reflection pad2d translates");
        let pad = find(&model, "layer0_trace_1");
        assert_eq!(pad.layer_type, LayerType::Add);
        assert_eq!(pad.outputs, vec!["layer0_trace_1_out".to_string()]);
        assert_eq!(count(&model, &LayerType::Pad), 0);
        build_ok(&model, "zero reflection pad2d");
    }
}
