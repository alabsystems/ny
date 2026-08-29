// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! ONNX `Resize` with `mode="linear"`, lowered to a separated pair of constant
//! `Linear` resampling matrices.
//!
//! # Can the ONNX side REUSE the trace bridge's bilinear resampler?
//!
//! No — and the honest reason is two-fold, one mechanical and one semantic.
//!
//! *Mechanically*, `ny-trace-bridge`'s `translate_resize_bilinear` /
//! `bilinear_axis_matrix` (`translate/ops_pad_resample.rs`) are private items
//! in a crate that DEPENDS on `ny-onnx` (it dev-depends on this crate for its
//! differential-translation oracle). Calling them from here would invert that
//! dependency.
//!
//! *Semantically* — the part that actually matters — the bridge's matrix is
//! hard-wired to nn's ONE kernel: `src = (dst + 0.5) * (in/out) - 0.5` clamped
//! to `[0, in-1]`, i.e. half-pixel centres, because nn's `resize_bilinear`
//! takes no `align_corners` argument and therefore cannot mean anything else.
//! ONNX carries `coordinate_transformation_mode` EXPLICITLY on the wire, and
//! four of its six values denote different source-coordinate maps. Reusing the
//! bridge's fixed map for an `align_corners` or `asymmetric` node would read
//! the WRONG source cells and emit a confidently non-enclosing bound whose
//! failure is silent — every shape still matches. So the arithmetic is
//! re-derived here per authored mode, and the modes whose arithmetic is not
//! replicated operation-for-operation are refused BY NAME.
//!
//! What IS reused is the bridge's *structure* and its soundness argument: the
//! separated `Linear` pair, the row-substochastic convex-combination property,
//! the exact duplicate-index merge invariant (checked, not assumed), and the
//! per-axis matrix size cap.
//!
//! # The lowering
//!
//! ```text
//! [.., H_in, W_in]  --Linear(A_x: [W_out, W_in])-->  [.., H_in, W_out]
//!                   --Transpose(swap -1,-2)------->  [.., W_out, H_in]
//!                   --Linear(A_y: [H_out, H_in])-->  [.., W_out, H_out]
//!                   --Transpose(swap -1,-2)------->  [.., H_out, W_out]
//! ```
//!
//! An axis whose extent is unchanged contributes an exact identity matrix
//! (provable from every admitted coordinate map: `in == out` ⇒ `src == dst` ⇒
//! `w == 0`), so its pass is skipped entirely.
//!
//! # Why the separated pair IS ONNX's function
//!
//! ONNX linear resampling of output cell `(oh, ow)` is
//! `v00*(1-wy)*(1-wx) + v01*(1-wy)*wx + v10*wy*(1-wx) + v11*wy*wx`, which
//! factors as `(1-wy) * [v00*(1-wx) + v01*wx] + wy * [v10*(1-wx) + v11*wx]`.
//! Bilinear interpolation is *separable*: interpolate along W for every source
//! row, then along H. In exact real arithmetic the separated and fused forms
//! are the same number, and the emitted matrices carry `1.0f32 - w` and `w`
//! verbatim — never pre-multiplied into a `wy*wx` product that would round.
//!
//! # Why clamping the coordinate reproduces ONNX's edge handling EXACTLY
//!
//! ONNX evaluates the 2-tap kernel at `i = floor(x)` and `i + 1` with the data
//! extended by EDGE padding, i.e. at `clamp(i, 0, n-1)` and
//! `clamp(i+1, 0, n-1)`, with weights `(1 - r, r)` where `r = x - i`. This
//! module instead clamps the COORDINATE first (`x_c = clamp(x, 0, n-1)`) and
//! interpolates between `i0 = floor(x_c)` and `i1 = min(i0 + 1, n-1)` with
//! `w = x_c - i0`. Over the reals the two agree for every `x`:
//!
//! * `0 <= x <= n-1`. Then `i = floor(x) <= n-1`. If `i < n-1` both taps are
//!   in range and both formulations give `(1-r) D[i] + r D[i+1]`. If
//!   `i == n-1` then `x == n-1` exactly (else `x > n-1`), so `r = 0`, ONNX
//!   gives `D[n-1]`, and here `i0 = i1 = n-1`, `w = 0`, also `D[n-1]`.
//! * `x < 0`. Then `i <= -1`, so `i + 1 <= 0` and BOTH ONNX taps clamp to
//!   index 0: `(1-r) D[0] + r D[0] = D[0]`. Here `x_c = 0`, `w = 0`, `D[0]`.
//! * `x > n-1`. Then `i >= n-1`, so both ONNX taps clamp to `n-1`: `D[n-1]`.
//!   Here `x_c = n-1`, `i0 = i1 = n-1`, `w = 0`, `D[n-1]`.
//!
//! The two out-of-range cases are where the formulations differ in FLOATING
//! POINT and not in the reals: ONNX computes `fl(fl((1-r)*a) + fl(r*a))`,
//! which need not be exactly `a`, while this lowering writes the single
//! coefficient `1.0f32`. That is a coefficient-ASSOCIATION difference of at
//! most a few ULPs, of exactly the kind ny's f32 idealization already assumes
//! for every authored `Gemm`; `propagate_ibp_sound` applies the certified
//! per-node Linear widening (`in_features + 2` ULPs, Higham Thm 3.1) on top.
//! It is NOT a difference in which source cells are read.
//!
//! # Scope note on the coordinate arithmetic (do not overstate this)
//!
//! The source coordinate is computed in `f64` from the authored `f32` scale,
//! and the interpolation weight is cast to `f32` once. A runtime that computes
//! the coordinate in `f32` can, within <= 1 ULP of an integer boundary, select
//! a different cell PAIR. The interpolated VALUE stays continuous across that
//! switch (the far cell takes weight 0 at the boundary), but the emitted
//! matrix is a FIXED constant map, so it would then model a slightly different
//! function. This is the same documented scope limit the trace bridge records
//! for its own `ResizeBilinear` lowering.

use super::lookup_opset_version;
use crate::loader::const_fold::common::read_tensor_i64s;
use crate::onnx_proto::{self, NodeProto};
use crate::{AttributeValue, DataType, LayerSpec, WeightRef, WeightStore};
use ndarray::{ArrayD, IxDyn};
use ny_core::{LayerType, NyError, Result};
use std::collections::HashMap;

/// Cap on ONE resampling matrix, in f32 entries (`out_size * in_size`).
///
/// Matches `ny-trace-bridge`'s `MAX_RESAMPLE_AXIS_MATRIX_ELEMS`: 4 Mi entries
/// = 16 MiB of frozen weight per axis. A 2048 -> 4096 resize fits; 4096 ->
/// 8192 does not and is refused rather than silently materialized.
const MAX_RESAMPLE_AXIS_MATRIX_ELEMS: usize = 1 << 22;

/// The `coordinate_transformation_mode` values whose arithmetic this module
/// replicates operation-for-operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CoordinateMode {
    HalfPixel,
    PytorchHalfPixel,
    AlignCorners,
    Asymmetric,
}

impl CoordinateMode {
    /// ONNX's source coordinate for output index `dst`, BEFORE clamping.
    ///
    /// Each arm is the closed form in the ONNX `Resize` specification, with
    /// `x_scale` the authored scale (output/input) for this axis:
    ///
    /// ```text
    /// half_pixel          x_original = (x_resized + 0.5) / x_scale - 0.5
    /// pytorch_half_pixel  x_original = length_resized > 1
    ///                                    ? (x_resized + 0.5) / x_scale - 0.5
    ///                                    : 0
    /// align_corners       x_original = x_resized * (length_original - 1)
    ///                                    / (length_resized - 1)
    /// asymmetric          x_original = x_resized / x_scale
    /// ```
    fn source_coordinate(self, dst: usize, x_scale: f64, in_len: usize, out_len: usize) -> f64 {
        let dst = dst as f64;
        match self {
            CoordinateMode::HalfPixel => (dst + 0.5) / x_scale - 0.5,
            CoordinateMode::PytorchHalfPixel => {
                if out_len > 1 {
                    (dst + 0.5) / x_scale - 0.5
                } else {
                    0.0
                }
            }
            CoordinateMode::AlignCorners => {
                if out_len == 1 {
                    0.0
                } else {
                    dst * ((in_len as f64) - 1.0) / ((out_len as f64) - 1.0)
                }
            }
            CoordinateMode::Asymmetric => dst / x_scale,
        }
    }
}

fn is_standard_domain(node: &NodeProto) -> bool {
    matches!(node.domain.as_str(), "" | "ai.onnx")
}

fn string_attribute<'a>(node: &'a NodeProto, name: &str) -> Option<&'a [u8]> {
    node.attribute
        .iter()
        .find(|attribute| {
            attribute.name == name && attribute.r#type == onnx_proto::attribute_type::STRING
        })
        .map(onnx_proto::AttributeProto::s_value)
}

fn int_attribute(node: &NodeProto, name: &str) -> Option<i64> {
    node.attribute
        .iter()
        .find(|attribute| {
            attribute.name == name && attribute.r#type == onnx_proto::attribute_type::INT
        })
        .map(onnx_proto::AttributeProto::i_value)
}

/// Build the `[out_len, in_len]` resampling matrix for ONE spatial axis.
///
/// # The duplicate-index merge is exact
///
/// When `i0 == i1` (the clamped last cell, or `in_len == 1`) the two
/// contributions land in the same entry, `(1 - w) + w`. That merge is exact
/// *because `w` is provably 0 there*: `src` is clamped into `[0, in_len - 1]`,
/// so `floor(src) == in_len - 1` forces `src == in_len - 1` exactly, hence
/// `w = src - i0 = 0`. The entry is then `1.0f32 + 0.0f32 = 1.0f32`, with no
/// rounding. The invariant is CHECKED below and refuses fail-closed if it ever
/// breaks, because a merged entry with `w != 0` is the one place a coefficient
/// could round away from ONNX's.
fn linear_axis_matrix(
    node_name: &str,
    axis_label: &str,
    mode: CoordinateMode,
    x_scale: f64,
    in_len: usize,
    out_len: usize,
) -> Result<ArrayD<f32>> {
    if in_len == 0 || out_len == 0 {
        return Err(NyError::UnsupportedOp(format!(
            "standard ONNX Resize node '{node_name}': {axis_label} extent must be non-zero \
             (in {in_len}, out {out_len})"
        )));
    }
    let elements = out_len.checked_mul(in_len).ok_or_else(|| {
        NyError::InternalError(format!(
            "standard ONNX Resize node '{node_name}': {axis_label} matrix {out_len}x{in_len} \
             overflows usize"
        ))
    })?;
    if elements > MAX_RESAMPLE_AXIS_MATRIX_ELEMS {
        return Err(NyError::UnsupportedOp(format!(
            "standard ONNX Resize node '{node_name}': {axis_label} resampling matrix \
             {out_len}x{in_len} = {elements} entries exceeds limit \
             {MAX_RESAMPLE_AXIS_MATRIX_ELEMS}"
        )));
    }

    let mut matrix = ArrayD::<f32>::zeros(IxDyn(&[out_len, in_len]));
    for dst in 0..out_len {
        let raw = mode.source_coordinate(dst, x_scale, in_len, out_len);
        if !raw.is_finite() {
            return Err(NyError::UnsupportedOp(format!(
                "standard ONNX Resize node '{node_name}': {axis_label} source coordinate for \
                 output cell {dst} is not finite (scale {x_scale})"
            )));
        }
        let src = raw.clamp(0.0, (in_len - 1) as f64);
        let i0_raw = src.floor() as usize;
        let i1 = (i0_raw + 1).min(in_len - 1);
        let i0 = i0_raw.min(in_len - 1);
        let weight = (src - i0 as f64) as f32;
        if !(0.0..1.0).contains(&weight) {
            return Err(NyError::InternalError(format!(
                "standard ONNX Resize node '{node_name}': {axis_label} cell {dst} produced \
                 interpolation weight {weight} outside [0, 1); refusing rather than emitting a \
                 non-convex row"
            )));
        }
        if i0 == i1 && weight != 0.0 {
            return Err(NyError::InternalError(format!(
                "standard ONNX Resize node '{node_name}': {axis_label} cell {dst} merges source \
                 index {i0} with a non-zero interpolation weight {weight} — the exact-merge \
                 invariant the coordinate clamp guarantees is broken, refusing rather than \
                 rounding the coefficient"
            )));
        }
        matrix[[dst, i0]] += 1.0_f32 - weight;
        matrix[[dst, i1]] += weight;
    }
    Ok(matrix)
}

/// Register `matrix` as a frozen weight and emit the `Linear` (ONNX Gemm,
/// `transB = 1`) that applies it to the LAST axis of `input`.
fn resample_linear_spec(
    name: &str,
    weight_name: String,
    matrix: ArrayD<f32>,
    input: &str,
    output: &str,
    weights: &mut WeightStore,
) -> Result<LayerSpec> {
    if weights.contains_key(&weight_name) {
        return Err(NyError::ModelLoad(format!(
            "standard ONNX Resize lowering for '{name}' would overwrite the existing tensor \
             '{weight_name}'"
        )));
    }
    let shape = matrix.shape().to_vec();
    weights.insert(weight_name.clone(), matrix);
    let mut attributes = HashMap::new();
    attributes.insert("transB".to_string(), AttributeValue::Int(1));
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
        attributes,
    })
}

/// Emit a `Transpose` swapping the last two axes of a rank-`rank` tensor.
///
/// The perm is authored at ONNX rank; `TransposeLayer::resolve_perm` maps it
/// onto ny's possibly batch-stripped runtime rank through
/// `normalize_transpose_perm_for_rank`, which fails closed when no equivalence
/// can be proven.
fn swap_last_two_spec(name: &str, rank: usize, input: &str, output: &str) -> LayerSpec {
    let mut perm: Vec<i64> = (0..rank as i64).collect();
    perm.swap(rank - 2, rank - 1);
    let mut attributes = HashMap::new();
    attributes.insert("perm".to_string(), AttributeValue::Ints(perm));
    LayerSpec {
        name: name.to_string(),
        layer_type: LayerType::Transpose,
        inputs: vec![input.to_string()],
        outputs: vec![output.to_string()],
        weights: None,
        attributes,
    }
}

/// Per-axis resize geometry: `(output_length, coordinate scale)`.
struct AxisPlan {
    out_len: usize,
    x_scale: f64,
}

/// Resolve the per-axis output lengths and coordinate scales from EXACTLY one
/// of the `scales` / `sizes` operands.
///
/// With `scales`, ONNX defines `output = floor(input * scale)` and the
/// coordinate transform uses the AUTHORED `scale`, not `output / input` — the
/// two differ whenever `input * scale` is not integral. This function requires
/// `input * scale` to be an exact non-negative integer so the two readings
/// cannot disagree, and refuses otherwise rather than picking one.
fn resolve_axis_plans(
    node_name: &str,
    input_shape: &[usize],
    scales: Option<&ArrayD<f32>>,
    sizes: Option<&[i64]>,
) -> Result<Vec<AxisPlan>> {
    let rank = input_shape.len();
    if let Some(scales) = scales {
        if scales.len() != rank {
            return Err(NyError::ModelLoad(format!(
                "standard ONNX Resize node '{node_name}': scales has {} entries for a rank-{rank} \
                 input",
                scales.len()
            )));
        }
        return scales
            .iter()
            .zip(input_shape)
            .enumerate()
            .map(|(axis, (&scale, &in_len))| {
                let x_scale = f64::from(scale);
                if !x_scale.is_finite() || x_scale <= 0.0 {
                    return Err(NyError::UnsupportedOp(format!(
                        "standard ONNX Resize node '{node_name}': axis {axis} scale {scale} is \
                         not a finite positive factor"
                    )));
                }
                let exact = in_len as f64 * x_scale;
                if exact.fract() != 0.0 || exact < 1.0 || exact > usize::MAX as f64 {
                    return Err(NyError::UnsupportedOp(format!(
                        "standard ONNX Resize node '{node_name}': axis {axis} scale {scale} on \
                         extent {in_len} does not give an exact positive integer output extent \
                         ({exact}); ONNX floors the output length but keeps the authored scale in \
                         the coordinate transform, and ny refuses to pick between the two readings"
                    )));
                }
                Ok(AxisPlan {
                    out_len: exact as usize,
                    x_scale,
                })
            })
            .collect();
    }

    let sizes = sizes.expect("caller guarantees exactly one of scales / sizes");
    if sizes.len() != rank {
        return Err(NyError::ModelLoad(format!(
            "standard ONNX Resize node '{node_name}': sizes has {} entries for a rank-{rank} input",
            sizes.len()
        )));
    }
    sizes
        .iter()
        .zip(input_shape)
        .enumerate()
        .map(|(axis, (&size, &in_len))| {
            let out_len = usize::try_from(size)
                .ok()
                .filter(|&len| len > 0)
                .ok_or_else(|| {
                    NyError::UnsupportedOp(format!(
                    "standard ONNX Resize node '{node_name}': axis {axis} size {size} is not a \
                     positive extent"
                ))
                })?;
            Ok(AxisPlan {
                out_len,
                x_scale: out_len as f64 / in_len as f64,
            })
        })
        .collect()
}

/// Canonicalize a standard ONNX `Resize` with `mode="linear"` into the
/// separated constant-`Linear` resampler described in the module docs.
///
/// Returns `Ok(None)` for everything that is not a linear `Resize` — notably
/// the `nearest` mode, which keeps its existing `LayerType::Resize` lowering
/// untouched.
///
/// # What stays refused, and why (each BY NAME)
///
/// * `mode="cubic"`: a 4-tap kernel with a `cubic_coeff_a` parameter and
///   non-convex weights. The matrix would not be row-substochastic and the
///   `A`/`B` coefficients are not replicated here.
/// * `Upsample` with `mode="linear"`: the deprecated opset-9 operator carries
///   no `coordinate_transformation_mode`, so the convention is not on the
///   wire — precisely the `Topk`/`Upsample2d` situation the trace bridge
///   refuses.
/// * opset 10 `Resize`: `coordinate_transformation_mode` does not exist before
///   opset 11, so the same "convention not on the wire" argument applies.
/// * `coordinate_transformation_mode = "tf_crop_and_resize"`: the coordinate
///   map depends on the `roi` operand and an `extrapolation_value` outside the
///   crop, neither of which is modelled here.
/// * `coordinate_transformation_mode = "half_pixel_symmetric"`: its
///   `adjustment`/`offset` arithmetic is not replicated operation-for-operation
///   here, and guessing it would move which source cells are read.
/// * `exclude_outside=1`: zeroes out-of-range taps and RENORMALIZES the
///   remaining coefficients, which is a different map at every border cell.
/// * `antialias=1`: widens the kernel support when downscaling, i.e. more than
///   two taps.
/// * `keep_aspect_ratio_policy` other than `"stretch"`, and any `axes`
///   attribute: both re-interpret which axes the scales/sizes address.
/// * A non-empty `roi`.
/// * Anything but exactly one resized pair of TRAILING axes on a rank-3 or
///   rank-4 input, with every other axis unchanged.
/// * `scales`/`sizes` that are not exact load-time constants, or a dynamic /
///   unknown input shape.
#[allow(clippy::too_many_lines)]
pub(super) fn canonicalize_linear_resize(
    node: &NodeProto,
    node_index: usize,
    weights: &mut WeightStore,
    tensor_shapes: &HashMap<String, Vec<i64>>,
    opset_imports: &HashMap<String, i64>,
) -> Result<Option<Vec<LayerSpec>>> {
    if !is_standard_domain(node) || !matches!(node.op_type.as_str(), "Resize" | "Upsample") {
        return Ok(None);
    }
    let mode = string_attribute(node, "mode").unwrap_or(b"nearest");
    if mode == b"nearest" {
        return Ok(None);
    }
    let name = if node.name.is_empty() {
        format!("resize_{node_index}")
    } else {
        node.name.clone()
    };
    if mode != b"linear" {
        return Err(NyError::UnsupportedOp(format!(
            "standard ONNX {} node '{name}' uses mode='{}'; ny represents mode='nearest' and \
             mode='linear' only — a cubic kernel has four taps and non-convex coefficients that \
             this lowering does not replicate",
            node.op_type,
            String::from_utf8_lossy(mode)
        )));
    }
    if node.op_type == "Upsample" {
        return Err(NyError::UnsupportedOp(format!(
            "standard ONNX Upsample node '{name}' uses mode='linear'; the deprecated opset-9 \
             Upsample carries no coordinate_transformation_mode, so the sampling convention is \
             not on the wire. Re-export as Resize (opset 11+), which pins it explicitly"
        )));
    }
    let opset = lookup_opset_version(opset_imports, &node.domain).ok_or_else(|| {
        NyError::ModelLoad(format!(
            "standard ONNX Resize node '{name}' has no standard-domain opset authority"
        ))
    })?;
    if opset < 11 {
        return Err(NyError::UnsupportedOp(format!(
            "standard ONNX Resize node '{name}' uses mode='linear' at opset {opset}; \
             coordinate_transformation_mode only exists from opset 11, so the sampling convention \
             is not on the wire"
        )));
    }

    let coordinate =
        string_attribute(node, "coordinate_transformation_mode").unwrap_or(b"half_pixel");
    let mode = match coordinate {
        b"half_pixel" => CoordinateMode::HalfPixel,
        b"pytorch_half_pixel" => CoordinateMode::PytorchHalfPixel,
        b"align_corners" => CoordinateMode::AlignCorners,
        b"asymmetric" => CoordinateMode::Asymmetric,
        other => {
            return Err(NyError::UnsupportedOp(format!(
                "standard ONNX Resize node '{name}' uses \
                 coordinate_transformation_mode='{}' with mode='linear'; ny replicates \
                 half_pixel, pytorch_half_pixel, align_corners and asymmetric \
                 operation-for-operation and refuses the rest rather than reading the wrong \
                 source cells",
                String::from_utf8_lossy(other)
            )));
        }
    };

    if int_attribute(node, "exclude_outside").unwrap_or(0) != 0 {
        return Err(NyError::UnsupportedOp(format!(
            "standard ONNX Resize node '{name}' sets exclude_outside=1; that zeroes out-of-range \
             taps and renormalizes the surviving coefficients, a different map at every border \
             cell than the edge-clamped kernel ny lowers"
        )));
    }
    if int_attribute(node, "antialias").unwrap_or(0) != 0 {
        return Err(NyError::UnsupportedOp(format!(
            "standard ONNX Resize node '{name}' sets antialias=1; that widens the kernel support \
             beyond the two taps this lowering represents"
        )));
    }
    if let Some(policy) = string_attribute(node, "keep_aspect_ratio_policy") {
        if policy != b"stretch" {
            return Err(NyError::UnsupportedOp(format!(
                "standard ONNX Resize node '{name}' uses keep_aspect_ratio_policy='{}'; that \
                 re-derives the per-axis output extents from the requested sizes, which ny does \
                 not replicate",
                String::from_utf8_lossy(policy)
            )));
        }
    }
    if node
        .attribute
        .iter()
        .any(|attribute| attribute.name == "axes")
    {
        return Err(NyError::UnsupportedOp(format!(
            "standard ONNX Resize node '{name}' carries an 'axes' attribute; ny's linear lowering \
             requires scales/sizes to address every input axis in order"
        )));
    }
    if node.output.len() != 1 || node.output[0].is_empty() {
        return Err(NyError::ModelLoad(format!(
            "standard ONNX Resize node '{name}' requires exactly one non-empty output"
        )));
    }
    if let Some(roi) = node.input.get(1).filter(|value| !value.is_empty()) {
        let empty = weights.get(roi).is_some_and(ndarray::ArrayBase::is_empty);
        if !empty {
            return Err(NyError::UnsupportedOp(format!(
                "standard ONNX Resize node '{name}' supplies a non-empty roi '{roi}'; that is the \
                 tf_crop_and_resize geometry, which ny does not represent"
            )));
        }
    }

    let scales_name = node.input.get(2).filter(|value| !value.is_empty());
    let sizes_name = node.input.get(3).filter(|value| !value.is_empty());
    if scales_name.is_some() == sizes_name.is_some() {
        return Err(NyError::ModelLoad(format!(
            "standard ONNX Resize node '{name}' requires exactly one non-empty scales or sizes \
             input"
        )));
    }

    let data = &node.input[0];
    let shape = tensor_shapes.get(data).ok_or_else(|| {
        NyError::UnsupportedOp(format!(
            "standard ONNX Resize node '{name}' needs a known static shape for '{data}' to \
             materialize its resampling matrices"
        ))
    })?;
    let input_shape = shape
        .iter()
        .map(|&dim| usize::try_from(dim).ok().filter(|&d| d > 0))
        .collect::<Option<Vec<usize>>>()
        .ok_or_else(|| {
            NyError::UnsupportedOp(format!(
                "standard ONNX Resize node '{name}': '{data}' has a dynamic or non-positive \
                 dimension in {shape:?}"
            ))
        })?;
    let rank = input_shape.len();
    if !(3..=4).contains(&rank) {
        return Err(NyError::UnsupportedOp(format!(
            "standard ONNX Resize node '{name}' has a rank-{rank} input; ny's linear lowering \
             resamples the two trailing spatial axes of a rank-3 or rank-4 tensor"
        )));
    }

    let scales = match scales_name {
        Some(scales_name) => Some(weights.get(scales_name).cloned().ok_or_else(|| {
            NyError::UnsupportedOp(format!(
                "standard ONNX Resize node '{name}' requires an exact constant FLOAT scales \
                 operand; '{scales_name}' is not available at load time"
            ))
        })?),
        None => None,
    };
    let sizes = match sizes_name {
        Some(sizes_name) => {
            if weights.get_integers(sizes_name).is_none() {
                return Err(NyError::UnsupportedOp(format!(
                    "standard ONNX Resize node '{name}' requires an exact constant INT64 sizes \
                     operand; '{sizes_name}' is not an authenticated INT64 constant"
                )));
            }
            Some(read_tensor_i64s(weights, sizes_name).ok_or_else(|| {
                NyError::UnsupportedOp(format!(
                    "standard ONNX Resize node '{name}' could not materialize its sizes operand \
                     '{sizes_name}'"
                ))
            })?)
        }
        None => None,
    };

    let plans = resolve_axis_plans(&name, &input_shape, scales.as_ref(), sizes.as_deref())?;
    for (axis, plan) in plans.iter().enumerate().take(rank - 2) {
        if plan.out_len != input_shape[axis] {
            return Err(NyError::UnsupportedOp(format!(
                "standard ONNX Resize node '{name}' resizes non-spatial axis {axis} \
                 ({} -> {}); ny's linear lowering resamples only the two trailing axes",
                input_shape[axis], plan.out_len
            )));
        }
    }

    let (in_h, in_w) = (input_shape[rank - 2], input_shape[rank - 1]);
    let out_h = plans[rank - 2].out_len;
    let out_w = plans[rank - 1].out_len;
    let resample_w = out_w != in_w;
    let resample_h = out_h != in_h;
    if !resample_w && !resample_h {
        // Every admitted coordinate map degenerates to `src == dst` when
        // `in == out`, so both matrices are the exact identity. Emitting an
        // identity Linear would still be correct; leaving the node to the
        // ordinary Resize path keeps this module out of the no-op case.
        return Err(NyError::UnsupportedOp(format!(
            "standard ONNX Resize node '{name}' with mode='linear' changes no spatial extent \
             ({in_h}x{in_w}); re-export without the no-op Resize"
        )));
    }

    let mut specs = Vec::with_capacity(4);
    let mut current = data.clone();
    let weight_prefix = format!("{name}__ny_resize{node_index}");

    if resample_w {
        let matrix =
            linear_axis_matrix(&name, "width", mode, plans[rank - 1].x_scale, in_w, out_w)?;
        let (spec_name, out_name) = if resample_h {
            (
                format!("{weight_prefix}_w"),
                format!("{weight_prefix}_w_out"),
            )
        } else {
            (name.clone(), node.output[0].clone())
        };
        specs.push(resample_linear_spec(
            &spec_name,
            format!("{weight_prefix}_w_weight"),
            matrix,
            &current,
            &out_name,
            weights,
        )?);
        current = out_name;
    }

    if resample_h {
        let matrix =
            linear_axis_matrix(&name, "height", mode, plans[rank - 2].x_scale, in_h, out_h)?;
        let transposed_in = format!("{weight_prefix}_ht_in");
        specs.push(swap_last_two_spec(
            &format!("{weight_prefix}_ht"),
            rank,
            &current,
            &transposed_in,
        ));
        let transposed_out = format!("{weight_prefix}_ht_out");
        specs.push(resample_linear_spec(
            &format!("{weight_prefix}_h"),
            format!("{weight_prefix}_h_weight"),
            matrix,
            &transposed_in,
            &transposed_out,
            weights,
        )?);
        specs.push(swap_last_two_spec(
            &name,
            rank,
            &transposed_out,
            &node.output[0],
        ));
    }

    Ok(Some(specs))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    fn opsets(version: i64) -> HashMap<String, i64> {
        HashMap::from([(String::new(), version)])
    }

    fn string_attr(name: &str, value: &str) -> onnx_proto::AttributeProto {
        onnx_proto::AttributeProto {
            name: name.to_string(),
            s: Some(value.as_bytes().to_vec()),
            r#type: onnx_proto::attribute_type::STRING,
            ..Default::default()
        }
    }

    fn int_attr(name: &str, value: i64) -> onnx_proto::AttributeProto {
        onnx_proto::AttributeProto {
            name: name.to_string(),
            i: Some(value),
            r#type: onnx_proto::attribute_type::INT,
            ..Default::default()
        }
    }

    fn resize_node(attributes: Vec<onnx_proto::AttributeProto>) -> NodeProto {
        NodeProto {
            name: "resize".to_string(),
            op_type: "Resize".to_string(),
            input: vec!["x".to_string(), String::new(), "scales".to_string()],
            output: vec!["out".to_string()],
            attribute: attributes,
            ..Default::default()
        }
    }

    fn scale_weights(scales: &[f32]) -> WeightStore {
        let mut weights = WeightStore::new();
        weights.insert("scales".to_string(), arr1(scales).into_dyn());
        weights
    }

    fn shapes(shape: &[i64]) -> HashMap<String, Vec<i64>> {
        HashMap::from([("x".to_string(), shape.to_vec())])
    }

    fn linear(coordinate: &str) -> Vec<onnx_proto::AttributeProto> {
        vec![
            string_attr("mode", "linear"),
            string_attr("coordinate_transformation_mode", coordinate),
        ]
    }

    /// Matrix rows are convex combinations: non-negative, summing to exactly 1.
    fn assert_row_stochastic(matrix: &ArrayD<f32>) {
        let (rows, columns) = (matrix.shape()[0], matrix.shape()[1]);
        for row in 0..rows {
            let mut sum = 0.0_f32;
            for column in 0..columns {
                let value = matrix[[row, column]];
                assert!(value >= 0.0, "negative coefficient at ({row}, {column})");
                sum += value;
            }
            assert_eq!(sum, 1.0, "row {row} is not a convex combination");
        }
    }

    #[test]
    fn half_pixel_matrix_reproduces_the_onnx_two_by_four_upscale() {
        let matrix =
            linear_axis_matrix("n", "width", CoordinateMode::HalfPixel, 2.0, 2, 4).unwrap();
        // src = (d + 0.5)/2 - 0.5 = [-0.25, 0.25, 0.75, 1.25] -> clamp [0, 1].
        assert_eq!(
            matrix.iter().copied().collect::<Vec<_>>(),
            vec![1.0, 0.0, 0.75, 0.25, 0.25, 0.75, 0.0, 1.0]
        );
        assert_row_stochastic(&matrix);
    }

    #[test]
    fn align_corners_and_asymmetric_are_different_maps() {
        let align =
            linear_axis_matrix("n", "width", CoordinateMode::AlignCorners, 2.0, 2, 4).unwrap();
        // src = d * 1 / 3 = [0, 1/3, 2/3, 1].
        let third = 1.0_f32 / 3.0;
        assert_eq!(align[[0, 0]], 1.0);
        assert_eq!(align[[3, 1]], 1.0);
        assert!((align[[1, 1]] - third).abs() < 1e-7);
        assert_row_stochastic(&align);

        let asymmetric =
            linear_axis_matrix("n", "width", CoordinateMode::Asymmetric, 2.0, 2, 4).unwrap();
        // src = d / 2 = [0, 0.5, 1, 1.5] -> clamp [0, 1].
        assert_eq!(
            asymmetric.iter().copied().collect::<Vec<_>>(),
            vec![1.0, 0.0, 0.5, 0.5, 0.0, 1.0, 0.0, 1.0]
        );
        assert_row_stochastic(&asymmetric);
        assert_ne!(
            align.iter().copied().collect::<Vec<_>>(),
            asymmetric.iter().copied().collect::<Vec<_>>(),
            "the two conventions must not collapse into one matrix"
        );
    }

    #[test]
    fn pytorch_half_pixel_degenerates_to_zero_for_a_single_output_cell() {
        let matrix =
            linear_axis_matrix("n", "width", CoordinateMode::PytorchHalfPixel, 0.25, 4, 1).unwrap();
        assert_eq!(
            matrix.iter().copied().collect::<Vec<_>>(),
            vec![1.0, 0.0, 0.0, 0.0]
        );

        // half_pixel at the same geometry samples src = (0.5)/0.25 - 0.5 = 1.5.
        let half = linear_axis_matrix("n", "width", CoordinateMode::HalfPixel, 0.25, 4, 1).unwrap();
        assert_eq!(
            half.iter().copied().collect::<Vec<_>>(),
            vec![0.0, 0.5, 0.5, 0.0]
        );
    }

    #[test]
    fn downscaling_rows_stay_convex() {
        for mode in [
            CoordinateMode::HalfPixel,
            CoordinateMode::PytorchHalfPixel,
            CoordinateMode::AlignCorners,
            CoordinateMode::Asymmetric,
        ] {
            let matrix = linear_axis_matrix("n", "width", mode, 0.5, 8, 4).unwrap();
            assert_row_stochastic(&matrix);
        }
    }

    #[test]
    fn oversized_matrices_are_refused() {
        // 2048 x 2048 is exactly MAX_RESAMPLE_AXIS_MATRIX_ELEMS.
        let side = 1_usize << 11;
        assert_eq!(side * side, MAX_RESAMPLE_AXIS_MATRIX_ELEMS);
        assert!(
            linear_axis_matrix("n", "width", CoordinateMode::HalfPixel, 1.0, side, side).is_ok()
        );
        assert!(
            linear_axis_matrix("n", "width", CoordinateMode::HalfPixel, 1.0, side, side + 1,)
                .is_err()
        );
    }

    #[test]
    fn nearest_resize_is_left_untouched() {
        let mut weights = scale_weights(&[1.0, 1.0, 2.0, 2.0]);
        assert!(canonicalize_linear_resize(
            &resize_node(vec![string_attr("mode", "nearest")]),
            0,
            &mut weights,
            &shapes(&[1, 1, 2, 2]),
            &opsets(17),
        )
        .unwrap()
        .is_none());
        assert!(
            canonicalize_linear_resize(
                &resize_node(Vec::new()),
                0,
                &mut weights,
                &shapes(&[1, 1, 2, 2]),
                &opsets(17),
            )
            .unwrap()
            .is_none(),
            "an omitted mode is ONNX's nearest default"
        );
    }

    #[test]
    fn both_axes_lower_to_linear_transpose_linear_transpose() {
        let mut weights = scale_weights(&[1.0, 1.0, 2.0, 2.0]);
        let specs = canonicalize_linear_resize(
            &resize_node(linear("half_pixel")),
            3,
            &mut weights,
            &shapes(&[1, 1, 2, 2]),
            &opsets(17),
        )
        .unwrap()
        .unwrap();

        assert_eq!(specs.len(), 4);
        assert_eq!(specs[0].layer_type, LayerType::Linear);
        assert_eq!(specs[1].layer_type, LayerType::Transpose);
        assert_eq!(specs[2].layer_type, LayerType::Linear);
        assert_eq!(specs[3].layer_type, LayerType::Transpose);
        assert_eq!(specs[0].inputs[0], "x");
        assert_eq!(specs[3].outputs, ["out"]);
        assert_eq!(
            specs[1].attributes["perm"],
            AttributeValue::Ints(vec![0, 1, 3, 2])
        );
        assert_eq!(specs[0].attributes["transB"], AttributeValue::Int(1));
        // Both resampling matrices were registered as frozen weights.
        for spec in [&specs[0], &specs[2]] {
            let weight = spec.weights.as_ref().expect("Linear carries a WeightRef");
            assert_eq!(weight.shape, vec![4, 2]);
            assert!(weights.contains_key(&weight.name));
        }
    }

    #[test]
    fn a_width_only_resize_needs_no_transpose() {
        let mut weights = scale_weights(&[1.0, 1.0, 1.0, 2.0]);
        let specs = canonicalize_linear_resize(
            &resize_node(linear("half_pixel")),
            0,
            &mut weights,
            &shapes(&[1, 1, 2, 2]),
            &opsets(17),
        )
        .unwrap()
        .unwrap();
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].layer_type, LayerType::Linear);
        assert_eq!(specs[0].outputs, ["out"]);
    }

    #[test]
    fn sizes_drive_the_scale_when_scales_is_absent() {
        let mut node = resize_node(linear("half_pixel"));
        node.input = vec![
            "x".to_string(),
            String::new(),
            String::new(),
            "sizes".to_string(),
        ];
        let mut weights = WeightStore::new();
        weights.insert_integers("sizes".to_string(), arr1(&[1_i64, 1, 2, 4]).into_dyn());
        weights.insert(
            "sizes".to_string(),
            arr1(&[1.0_f32, 1.0, 2.0, 4.0]).into_dyn(),
        );

        let specs =
            canonicalize_linear_resize(&node, 0, &mut weights, &shapes(&[1, 1, 2, 2]), &opsets(18))
                .unwrap()
                .unwrap();
        assert_eq!(specs.len(), 1, "only the width axis changes");
        let weight = specs[0].weights.as_ref().unwrap();
        assert_eq!(weight.shape, vec![4, 2]);
    }

    #[test]
    fn unrepresented_conventions_are_refused_by_name() {
        for coordinate in ["tf_crop_and_resize", "half_pixel_symmetric"] {
            let mut weights = scale_weights(&[1.0, 1.0, 2.0, 2.0]);
            let error = canonicalize_linear_resize(
                &resize_node(linear(coordinate)),
                0,
                &mut weights,
                &shapes(&[1, 1, 2, 2]),
                &opsets(18),
            )
            .expect_err("unmodelled coordinate mode must be refused");
            assert!(error.to_string().contains(coordinate), "{error}");
        }
    }

    #[test]
    fn cubic_exclude_outside_antialias_and_axes_are_refused_by_name() {
        let cases: Vec<(Vec<onnx_proto::AttributeProto>, &str)> = vec![
            (vec![string_attr("mode", "cubic")], "cubic"),
            (
                {
                    let mut attributes = linear("half_pixel");
                    attributes.push(int_attr("exclude_outside", 1));
                    attributes
                },
                "exclude_outside",
            ),
            (
                {
                    let mut attributes = linear("half_pixel");
                    attributes.push(int_attr("antialias", 1));
                    attributes
                },
                "antialias",
            ),
            (
                {
                    let mut attributes = linear("half_pixel");
                    attributes.push(onnx_proto::AttributeProto {
                        name: "axes".to_string(),
                        ints: vec![2, 3],
                        r#type: onnx_proto::attribute_type::INTS,
                        ..Default::default()
                    });
                    attributes
                },
                "axes",
            ),
            (
                {
                    let mut attributes = linear("half_pixel");
                    attributes.push(string_attr("keep_aspect_ratio_policy", "not_larger"));
                    attributes
                },
                "keep_aspect_ratio_policy",
            ),
        ];
        for (attributes, needle) in cases {
            let mut weights = scale_weights(&[1.0, 1.0, 2.0, 2.0]);
            let error = canonicalize_linear_resize(
                &resize_node(attributes),
                0,
                &mut weights,
                &shapes(&[1, 1, 2, 2]),
                &opsets(18),
            )
            .expect_err("unrepresented configuration must be refused");
            assert!(error.to_string().contains(needle), "{needle}: {error}");
        }
    }

    #[test]
    fn deprecated_upsample_linear_and_opset_10_resize_are_refused() {
        let mut upsample = resize_node(vec![string_attr("mode", "linear")]);
        upsample.op_type = "Upsample".to_string();
        upsample.input = vec!["x".to_string(), "scales".to_string()];
        let mut weights = scale_weights(&[1.0, 1.0, 2.0, 2.0]);
        assert!(canonicalize_linear_resize(
            &upsample,
            0,
            &mut weights,
            &shapes(&[1, 1, 2, 2]),
            &opsets(9),
        )
        .is_err());

        assert!(canonicalize_linear_resize(
            &resize_node(vec![string_attr("mode", "linear")]),
            0,
            &mut weights,
            &shapes(&[1, 1, 2, 2]),
            &opsets(10),
        )
        .is_err());
    }

    #[test]
    fn non_integral_scales_and_resized_batch_axes_are_refused() {
        // 2 * 1.5 = 3.0 is exact, so the authored scale and out/in agree and
        // the node lowers.
        let mut weights = scale_weights(&[1.0, 1.0, 2.0, 1.5]);
        assert!(
            canonicalize_linear_resize(
                &resize_node(linear("half_pixel")),
                0,
                &mut weights,
                &shapes(&[1, 1, 2, 2]),
                &opsets(18),
            )
            .is_ok_and(|specs| specs.is_some()),
            "an exactly integral scale must lower"
        );

        // 3 * 1.5 = 4.5: ONNX floors the extent to 4 but keeps 1.5 in the
        // coordinate transform, so the two readings disagree — refuse.
        let mut weights = scale_weights(&[1.0, 1.0, 2.0, 1.5]);
        let error = canonicalize_linear_resize(
            &resize_node(linear("half_pixel")),
            0,
            &mut weights,
            &shapes(&[1, 1, 2, 3]),
            &opsets(18),
        )
        .expect_err("a non-integral output extent must be refused");
        assert!(
            error.to_string().contains("exact positive integer"),
            "{error}"
        );

        let mut weights = scale_weights(&[1.0, 2.0, 2.0, 2.0]);
        let error = canonicalize_linear_resize(
            &resize_node(linear("half_pixel")),
            0,
            &mut weights,
            &shapes(&[1, 1, 2, 2]),
            &opsets(18),
        )
        .expect_err("resizing the channel axis must be refused");
        assert!(error.to_string().contains("non-spatial axis"), "{error}");
    }

    #[test]
    fn unknown_shapes_and_dynamic_controls_are_refused() {
        let mut weights = scale_weights(&[1.0, 1.0, 2.0, 2.0]);
        assert!(canonicalize_linear_resize(
            &resize_node(linear("half_pixel")),
            0,
            &mut weights,
            &HashMap::new(),
            &opsets(18),
        )
        .is_err());

        assert!(canonicalize_linear_resize(
            &resize_node(linear("half_pixel")),
            0,
            &mut weights,
            &shapes(&[1, 1, -1, 2]),
            &opsets(18),
        )
        .is_err());

        let mut empty = WeightStore::new();
        assert!(canonicalize_linear_resize(
            &resize_node(linear("half_pixel")),
            0,
            &mut empty,
            &shapes(&[1, 1, 2, 2]),
            &opsets(18),
        )
        .is_err());
    }
}
