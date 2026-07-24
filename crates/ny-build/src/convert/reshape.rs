// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::ArrayD;
use ny_core::{reshape_copy_axis_from_sentinel, reshape_copy_axis_sentinel, NyError, Result};
use ny_propagate::layers::{
    ExpandLikeLastAxisLayer, FlattenLayer, MulConstantLayer, ReshapeLayer, SliceLayer,
    SqueezeLayer, TileLayer, TransposeLayer, UnsqueezeLayer,
};
use ny_propagate::Layer;
use tracing::{debug, warn};

use super::{AttributeValue, ConvertContext, LayerSpec};

fn parse_slice_scalar_value(
    spec: &LayerSpec,
    field: &str,
    value: f32,
    allow_positive_infinity: bool,
) -> Result<i64> {
    if value.is_nan() {
        return Err(NyError::ModelLoad(format!(
            "Slice '{}': NaN in {} value ({})",
            spec.name, field, value
        )));
    }
    if value.is_infinite() {
        if allow_positive_infinity && value.is_sign_positive() {
            return Ok(i64::MAX);
        }
        let detail = if field == "start" {
            "infinite start value"
        } else if field == "end" && value.is_sign_negative() {
            "negative infinite end value"
        } else {
            "non-finite value"
        };
        return Err(NyError::ModelLoad(format!(
            "Slice '{}': {} ({})",
            spec.name, detail, value
        )));
    }

    let truncated = value.trunc();
    if truncated < i64::MIN as f32 || truncated > i64::MAX as f32 {
        return Err(NyError::ModelLoad(format!(
            "Slice '{}': {} value {} out of valid range",
            spec.name, field, value
        )));
    }

    // ONNX's "slice to the end of the axis" sentinel is `INT64_MAX`, but the
    // converter reads Slice bounds out of the f32 constant store, so the
    // sentinel arrives here as the *finite* f32 `2^63` (i64::MAX is not f32
    // exact-representable; the decoder warns about the lossy cast). Without this
    // promotion the clamp below turns it into `2^48`, which is numerically
    // equivalent for a real data axis (both clamp to `axis_len` in
    // `SliceLayer::validate_range`) but DESTROYS the sentinel identity that
    // `resolve_batch_axis_slice` relies on: an ONNX axis-0 `[0:INT64_MAX)` slice
    // on a multi-dimensional activation with an unknown batch extent then fails
    // its full-coverage test and is rejected as an unsound partial batch slice
    // instead of being recognised as the no-op it is (#linearizenn-slice-sentinel).
    //
    // DIRECTION / SOUNDNESS: restoring the sentinel can only make the resolved
    // end LARGER (`2^48` → "to end"), and `end` is clamped to the axis length
    // downstream, so the selected extent is bit-identical for every real tensor.
    // The threshold is deliberately above any realizable tensor extent (2^62).
    // A non-sentinel ONNX bound in that range is therefore observationally
    // equivalent to "to end" for every tensor this process can represent, so
    // promotion cannot change the selected elements.
    const SLICE_END_SENTINEL_FLOOR: f32 = (1u64 << 62) as f32;
    if allow_positive_infinity && truncated >= SLICE_END_SENTINEL_FLOOR {
        return Ok(i64::MAX);
    }

    // Clamp finite slice indices to a sane range. Real indices are bounded by the
    // tensor dims; any magnitude beyond this is meaningless because the downstream
    // normalization clamps to `[0, dim]` (`(dim + v).max(0)` / `v.min(dim)`), and
    // an out-of-range axis still resolves to `Err` via `.get()`. Clamping is thus
    // semantically equivalent AND keeps the `ndim + axis` / `dim + start` i64
    // arithmetic from overflowing on an adversarial ONNX Slice attribute (Trust
    // verifier: the arithmetic_overflow_add/sub obligations in convert_slice). The
    // `+inf` "slice to end" sentinel (i64::MAX) is returned above, before this, so
    // it is preserved.
    const SLICE_INDEX_BOUND: i64 = 1 << 48;
    Ok((truncated as i64).clamp(-SLICE_INDEX_BOUND, SLICE_INDEX_BOUND))
}

fn adjust_copy_axis_for_unbatched_shape(dim: i64, target_idx: usize) -> i64 {
    let Some(axis) = reshape_copy_axis_from_sentinel(dim) else {
        return dim;
    };
    if axis == 0 {
        return 1;
    }
    let internal_axis = axis - 1;
    if internal_axis == target_idx {
        return 0;
    }
    reshape_copy_axis_sentinel(internal_axis).unwrap_or(dim)
}

fn unbatched_target_shape_from_onnx_shape(onnx_shape: &[i64]) -> Vec<i64> {
    if onnx_shape.len() > 1 {
        onnx_shape[1..]
            .iter()
            .enumerate()
            .map(|(target_idx, &dim)| adjust_copy_axis_for_unbatched_shape(dim, target_idx))
            .collect()
    } else {
        onnx_shape.to_vec()
    }
}

/// Verbatim internal reshape target for UNBATCHED models (#cctsdb B5): the
/// ONNX target describes the internal tensor directly — no leading-dim strip,
/// no copy-axis index shift. Copy-axis sentinels referencing their own target
/// index normalize to ONNX's public `0` sentinel; other sentinels keep their
/// (already verbatim) axis reference.
fn verbatim_target_shape_from_onnx_shape(onnx_shape: &[i64]) -> Vec<i64> {
    onnx_shape
        .iter()
        .enumerate()
        .map(
            |(target_idx, &dim)| match reshape_copy_axis_from_sentinel(dim) {
                Some(axis) if axis == target_idx => 0,
                _ => dim,
            },
        )
        .collect()
}

fn validate_reshape_target_shape(spec: &LayerSpec, target_shape: &[i64]) -> Result<()> {
    let mut infer_idx = None;
    for (idx, dim) in target_shape.iter().copied().enumerate() {
        if dim == -1 {
            if let Some(first_idx) = infer_idx {
                return Err(NyError::InvalidSpec(format!(
                    "Reshape '{}': target_shape has multiple inferred dimensions (-1) at indices {} and {}",
                    spec.name, first_idx, idx
                )));
            }
            infer_idx = Some(idx);
        }
    }

    if let Some((idx, dim)) = target_shape
        .iter()
        .copied()
        .enumerate()
        .find(|(_, dim)| *dim < -1)
    {
        return Err(NyError::InvalidSpec(format!(
            "Reshape '{}': target_shape contains invalid negative dimension {} at index {} \
             (only -1 infer and 0 copy are supported)",
            spec.name, dim, idx
        )));
    }

    Ok(())
}

fn validate_reshape_target_shape_with_copy_axes(
    spec: &LayerSpec,
    target_shape: &[i64],
) -> Result<()> {
    let mut infer_idx = None;
    for (idx, dim) in target_shape.iter().copied().enumerate() {
        if dim == -1 {
            if let Some(first_idx) = infer_idx {
                return Err(NyError::InvalidSpec(format!(
                    "Reshape '{}': target_shape has multiple inferred dimensions (-1) at indices {} and {}",
                    spec.name, first_idx, idx
                )));
            }
            infer_idx = Some(idx);
        } else if reshape_copy_axis_from_sentinel(dim).is_none() && dim < -1 {
            return Err(NyError::InvalidSpec(format!(
                "Reshape '{}': target_shape contains invalid negative dimension {} at index {} \
                 (only -1 infer and 0 copy are supported)",
                spec.name, dim, idx
            )));
        }
    }

    Ok(())
}

fn reshape_shape_from_integer_tensor(
    spec: &LayerSpec,
    shape_tensor: &ArrayD<i64>,
    float_tensor: Option<&ArrayD<f32>>,
) -> Result<Vec<i64>> {
    let onnx_target_shape: Vec<i64> = shape_tensor.iter().copied().collect();
    if let Some(float_tensor) = float_tensor {
        for (idx, (&int_dim, &float_dim)) in onnx_target_shape
            .iter()
            .zip(float_tensor.iter())
            .enumerate()
        {
            if reshape_copy_axis_from_sentinel(int_dim).is_some() && float_dim != 0.0 {
                return Err(NyError::InvalidSpec(format!(
                    "Reshape '{}': invalid internal copy-axis sentinel at index {}",
                    spec.name, idx
                )));
            }
        }
    }
    validate_reshape_target_shape_with_copy_axes(spec, &onnx_target_shape)?;
    Ok(onnx_target_shape)
}

fn reshape_shape_from_float_tensor(
    spec: &LayerSpec,
    shape_tensor: &ArrayD<f32>,
) -> Result<Option<Vec<i64>>> {
    // f32->i64 cast: shape values originally stored as INT64 in ONNX can go through
    // a lossy f32 round-trip. Values outside f32's exact-integer range
    // (+/-2^24 = +/-16_777_216) may have been silently rounded during loading.
    if shape_tensor.iter().any(|&v| !v.is_finite()) {
        warn!(
            "Reshape '{}': shape tensor contains NaN or Inf values; cannot construct valid shape",
            spec.name
        );
        return Ok(None);
    }
    const F32_INT_EXACT_LIMIT: f32 = 16_777_216.0; // 2^24
    let onnx_target_shape: Vec<i64> = shape_tensor
        .iter()
        .copied()
        .enumerate()
        .map(|(idx, v)| {
            let truncated = v.trunc();
            if truncated != v {
                return Err(NyError::InvalidSpec(format!(
                    "Reshape '{}': shape tensor contains non-integer dimension {} at index {}",
                    spec.name, v, idx
                )));
            }
            // #2360: Reject values beyond f32 exact-integer range. The value has
            // already been rounded by the f32 representation, so accepting it
            // means using the wrong dimension silently.
            if v.abs() > F32_INT_EXACT_LIMIT {
                return Err(NyError::ModelLoad(format!(
                    "Reshape '{}': shape value {v} exceeds f32 exact-integer range +/-2^24; \
                     the i64->f32->i64 round-trip may have changed this dimension",
                    spec.name
                )));
            }
            Ok(v as i64)
        })
        .collect::<Result<_>>()?;
    validate_reshape_target_shape(spec, &onnx_target_shape)?;
    Ok(Some(onnx_target_shape))
}

fn build_attribute_slice_layer(
    ctx: &ConvertContext,
    spec: &LayerSpec,
    axis_raw: i64,
    start_raw: i64,
    end_raw: i64,
    mode: &str,
) -> Result<Layer> {
    let axis = i32::try_from(axis_raw).map_err(|_| {
        NyError::ModelLoad(format!(
            "Slice '{}': axis {} out of i32 range",
            spec.name, axis_raw
        ))
    })?;
    if start_raw < 0 || end_raw < 0 {
        return Err(NyError::ModelLoad(format!(
            "Slice '{}': negative attribute-based indices not supported (start={}, end={})",
            spec.name, start_raw, end_raw
        )));
    }
    let start = usize::try_from(start_raw).map_err(|_| {
        NyError::ModelLoad(format!(
            "Slice '{}': start {} out of usize range",
            spec.name, start_raw
        ))
    })?;
    let end = usize::try_from(end_raw).map_err(|_| {
        NyError::ModelLoad(format!(
            "Slice '{}': end {} out of usize range",
            spec.name, end_raw
        ))
    })?;
    if ctx.model_unbatched {
        // Unbatched model (#cctsdb B5): ONNX axes are verbatim data axes.
        debug!(
            "Converting Slice '{}' ({mode}) unbatched model: ONNX axis={} verbatim, start={}, end={}",
            spec.name, axis, start, end
        );
        return Ok(Layer::Slice(SliceLayer::new(axis, start, end)));
    }
    if axis == 0 {
        return ctx.resolve_batch_axis_slice(spec, start_raw, end_raw, 1, mode);
    }
    // Trailing-relative remap: correct whether the runtime tensor kept its
    // ONNX rank (leading size-1 retained) or had the batch dim stripped.
    // See `ConvertContext::remap_axis_trailing` (#pensieve ReduceSum no-op).
    let data_name =
        spec.inputs.first().map(String::as_str).ok_or_else(|| {
            NyError::ModelLoad(format!("Slice '{}' has no data input", spec.name))
        })?;
    let adjusted_axis = ctx.remap_axis_trailing(
        "Slice",
        &spec.name,
        data_name,
        i64::from(axis),
        crate::convert::LegacyBatchAxisPolicy::RejectZero,
    )?;
    let adjusted_axis = i32::try_from(adjusted_axis).map_err(|_| {
        NyError::ModelLoad(format!(
            "Slice '{}': remapped axis {} does not fit i32",
            spec.name, adjusted_axis
        ))
    })?;
    debug!(
        "Converting Slice '{}' ({mode}) with ONNX axis={}, adjusted axis={}, start={}, end={}",
        spec.name, axis, adjusted_axis, start, end
    );
    Ok(Layer::Slice(SliceLayer::new(adjusted_axis, start, end)))
}

impl ConvertContext<'_> {
    /// Try to convert Reshape to a layer. Returns None if shape is dynamic.
    ///
    /// Shape inference for dynamic Reshape (from Concat of known values) is handled
    /// during model loading in `extract_layers_and_weights`, not here. By this point,
    /// the shape should already be in weights or evaluated constants if it was inferable.
    pub(crate) fn try_convert_reshape(&self, spec: &LayerSpec) -> Result<Option<Layer>> {
        // Check for shape in attributes (native/GGUF models)
        if let Some(AttributeValue::Ints(shape)) = spec.attributes.get("shape") {
            validate_reshape_target_shape(spec, shape)?;
            debug!(
                "Reshape {} using shape {:?} from attributes",
                spec.name, shape
            );
            return Ok(Some(Layer::Reshape(ReshapeLayer::new(shape.clone()))));
        }

        // ONNX Reshape: inputs are (data, shape)
        // The shape must be a constant tensor (may have been inferred from Concat)
        if spec.inputs.len() < 2 {
            debug!("Reshape {} has < 2 inputs, cannot convert", spec.name);
            return Ok(None);
        }

        let shape_name = &spec.inputs[1];
        debug!(
            "Reshape {} looking for shape tensor '{}'",
            spec.name, shape_name
        );

        // Shape can come from model weights or from a pre-evaluated constant chain.
        // Prefer exact integer storage when available so folded Shape/Gather chains
        // retain symbolic dimensions instead of the legacy f32 compatibility view.
        let onnx_target_shape = if let Some(shape_tensor) = self.weights.get_integers(shape_name) {
            Some(reshape_shape_from_integer_tensor(
                spec,
                shape_tensor,
                self.weights.get(shape_name),
            )?)
        } else if let Some(shape_tensor) = self
            .weights
            .get(shape_name)
            .or_else(|| self.evaluated_constants.get(shape_name))
        {
            reshape_shape_from_float_tensor(spec, shape_tensor)?
        } else {
            None
        };

        if let Some(onnx_target_shape) = onnx_target_shape {
            // Adjust for unbatched operation: ny-propagate strips the batch dimension.
            // ONNX Reshape target shapes include the batch dimension as the first element.
            // Drop it to match the unbatched tensor layout, consistent with how Slice,
            // Gather, Flatten, and Squeeze adjust their axes for unbatched mode (#3198).
            //
            // Example: ONNX target [-1, 6, 8] → unbatched [6, 8]
            //          ONNX target [1, -1]     → unbatched [-1]
            //
            // Exception (#cctsdb B5): in a globally UNBATCHED model no tensor ever
            // carried a batch axis, so the ONNX target maps verbatim. Stripping
            // would mangle real data dims (e.g. cctsdb head Reshape target
            // [48, 2, 16] → [2, 16] loses the 48).
            let target_shape = if self.model_unbatched {
                verbatim_target_shape_from_onnx_shape(&onnx_target_shape)
            } else {
                unbatched_target_shape_from_onnx_shape(&onnx_target_shape)
            };
            debug!(
                "Reshape {} using ONNX shape {:?}, unbatched shape {:?} from constant tensor '{}'",
                spec.name, onnx_target_shape, target_shape, shape_name
            );
            return Ok(Some(Layer::Reshape(ReshapeLayer::new(target_shape))));
        }

        debug!(
            "Reshape {} shape tensor '{}' not found in weights or evaluated constants",
            spec.name, shape_name
        );
        Ok(None)
    }
    pub(crate) fn convert_flatten(&self, spec: &LayerSpec) -> Result<Layer> {
        // ONNX Flatten: collapses dimensions according to axis attribute
        // Default axis is 1 (flatten all dimensions except batch)
        let onnx_axis = match spec.attributes.get("axis") {
            // SAFETY(#3100): i64->i32 checked conversion; ONNX axis values are small
            Some(AttributeValue::Int(a)) => i32::try_from(*a).map_err(|_| {
                NyError::ModelLoad(format!(
                    "Flatten '{}': axis {} out of i32 range",
                    spec.name, a
                ))
            })?,
            _ => 1, // ONNX default axis is 1
        };

        // IMPORTANT: ny-propagate works with unbatched tensors (batch dimension squeezed out).
        // ONNX axis=1 means "keep batch, flatten rest" → in batched mode: (N, C, H, W) → (N, C*H*W).
        // In unbatched mode, axis adjusts to 0: (C, H, W) → [1, C*H*W] (2D).
        //
        // We use FlattenLayer(0) which produces 2D [1, C*H*W]. This preserves shape
        // compatibility with CROWN backward propagation, which produces 2D bounds from
        // downstream Linear layers. Using Reshape([-1]) would produce 1D [C*H*W],
        // triggering CROWN-IBP shape mismatch fallback and losing CROWN tightening (#3300).
        //
        // Linear layers handle both 1D and 2D inputs (propagate_ibp_1d vs propagate_ibp_nd),
        // so the 2D output from FlattenLayer(0) is compatible with downstream layers.
        //
        // ONNX axis 0 normally targets the batch dimension, which does not exist in
        // unbatched mode. Ref: alpha-beta-CROWN raises ValueError for axis==0 in
        // Squeeze/Unsqueeze; same logic applies to Flatten. (#2310)
        //
        // Exception (genuine rank-≤1 data axis): when the input's recorded ONNX shape is
        // rank ≤ 1 (`data_had_batch_axis == Some(false)`), there was no batch axis to
        // strip (`unbatched_target_shape_from_onnx_shape` keeps it verbatim), so ONNX
        // axis 0 is a genuine data axis and maps directly to internal axis 0. Mirrors
        // `resolve_gather_axis` and the Slice Case 2b convention. Load-time axis
        // interpretation only — no bound math.
        if onnx_axis == 0 {
            let genuine_data_axis = spec
                .inputs
                .first()
                .is_some_and(|input| self.data_had_batch_axis(input) == Some(false));
            if !genuine_data_axis {
                return Err(NyError::ModelLoad(format!(
                    "Flatten '{}': axis=0 targets batch dimension which does not exist in unbatched mode",
                    spec.name
                )));
            }
            debug!(
                "Converting Flatten '{}': axis=0 on a known rank-≤1 input (no stripped batch \
                 axis) — genuine data axis, FlattenLayer(0)",
                spec.name
            );
            return Ok(Layer::Flatten(FlattenLayer::new(0)));
        }

        let adjusted_axis = if self.model_unbatched {
            onnx_axis // unbatched model: ONNX axes are verbatim (#cctsdb B5)
        } else if onnx_axis >= 1 {
            onnx_axis - 1
        } else {
            onnx_axis // negative axes pass through
        };

        debug!(
            "Converting Flatten '{}' with ONNX axis={}, adjusted axis={} (for unbatched operation)",
            spec.name, onnx_axis, adjusted_axis
        );

        Ok(Layer::Flatten(FlattenLayer::new(adjusted_axis)))
    }
    pub(crate) fn convert_tile(&self, spec: &LayerSpec) -> Result<Layer> {
        // Tile layer: repeats tensor along specified axis
        // Attributes: "axis" (i64) - axis to repeat along, "reps" (i64) - repetition count
        let axis = match spec.attributes.get("axis") {
            // SAFETY(#3100): i64->i32 checked conversion
            Some(AttributeValue::Int(a)) => i32::try_from(*a).map_err(|_| {
                NyError::ModelLoad(format!("Tile '{}': axis {} out of i32 range", spec.name, a))
            })?,
            _ => {
                return Err(NyError::ModelLoad(
                    "Tile layer requires 'axis' attribute".to_string(),
                ))
            }
        };

        let reps = match spec.attributes.get("reps") {
            Some(AttributeValue::Int(r)) if *r > 0 => *r as usize,
            Some(AttributeValue::Int(r)) => {
                return Err(NyError::ModelLoad(format!(
                    "Tile '{}': reps must be positive (got {})",
                    spec.name, r
                )))
            }
            _ => {
                return Err(NyError::ModelLoad(
                    "Tile layer requires 'reps' attribute".to_string(),
                ))
            }
        };

        debug!(
            "Converting Tile '{}' with axis={}, reps={}",
            spec.name, axis, reps
        );
        Ok(Layer::Tile(TileLayer::new(axis, reps)))
    }

    pub(crate) fn convert_expand(&self, spec: &LayerSpec) -> Result<Layer> {
        if spec.inputs.len() < 2 {
            return Err(NyError::ModelLoad(format!(
                "Expand '{}' requires 2 inputs (data, shape), got {}",
                spec.name,
                spec.inputs.len()
            )));
        }
        let data_name = &spec.inputs[0];
        let shape_name = &spec.inputs[1];
        let data_is_constant = self.constant_value(data_name).is_some();
        let shape_is_constant = self.constant_value(shape_name).is_some();
        // Variable data broadcast to a CONSTANT static target shape (#cctsdb):
        // ONNX Expand is multidirectional broadcasting, which is EXACTLY
        // elementwise multiplication by a ones tensor of the target shape
        // (x * 1 = x is exact in IEEE-754, including signed zeros through the
        // interval endpoints; MulConstant already implements two-sided ONNX
        // broadcast). Currently wired for unbatched models only, where the
        // ONNX target shape is the verbatim internal shape (cctsdb coordinate
        // grids Expand_102/110); batched models keep the legacy behavior.
        if self.model_unbatched && !data_is_constant {
            if let Some(target) = self.expand_static_target_shape(shape_name) {
                debug!(
                    "Converting Expand '{}' to MulConstant(ones {:?}) broadcast (unbatched model)",
                    spec.name, target
                );
                let ones = ArrayD::from_elem(ndarray::IxDyn(&target), 1.0_f32);
                return Ok(Layer::MulConstant(MulConstantLayer::new(ones)));
            }
        }
        if data_is_constant || shape_is_constant {
            return Err(NyError::UnsupportedOp(format!(
                "Expand '{}': constant-side Expand with inputs {:?} needs constant folding",
                spec.name, spec.inputs
            )));
        }
        debug!(
            "Converting Expand '{}' to narrow last-axis runtime lowering",
            spec.name
        );
        Ok(Layer::ExpandLikeLastAxis(ExpandLikeLastAxisLayer::new()))
    }

    /// Static Expand target shape from a constant shape tensor: all dims must
    /// be strictly positive concrete integers (no `-1` copy markers, no
    /// symbolic copy-axis sentinels) and the element count bounded. `None`
    /// falls back to the legacy conversion paths.
    fn expand_static_target_shape(&self, shape_name: &str) -> Option<Vec<usize>> {
        const MAX_EXPAND_ELEMENTS: usize = 10_000_000;
        let dims: Vec<i64> = if let Some(ints) = self.weights.get_integers(shape_name) {
            ints.iter().copied().collect()
        } else {
            let floats = self.constant_value(shape_name)?;
            floats
                .iter()
                .map(|&v| {
                    (v.is_finite() && v.trunc() == v && v.abs() <= 16_777_216.0).then_some(v as i64)
                })
                .collect::<Option<_>>()?
        };
        if dims.is_empty() || dims.iter().any(|&d| d <= 0) {
            return None;
        }
        let target: Vec<usize> = dims.iter().map(|&d| d as usize).collect();
        let elems = ny_core::checked_shape_product(&target)?;
        (elems <= MAX_EXPAND_ELEMENTS).then_some(target)
    }

    /// Resolve an ONNX `Slice` whose `axis == 0` for unbatched propagation.
    ///
    /// ONNX axis 0 nominally targets the batch dimension, which `ny` strips before
    /// propagation, so a naive `axis - 1` remap is invalid. There are three cases:
    ///
    /// 1. **No-op full-batch slice** (`start == 0`, `step == 1`, and `end` covers the
    ///    whole axis-0 extent — either the `INT64_MAX` sentinel or `end >= batch`):
    ///    the slice selects the entire dimension, so it changes nothing. We emit an
    ///    identity (`MulConstant(1.0)`), which is sound regardless of whether axis 0
    ///    was a size-1 batch dim or a genuine data axis.
    ///
    /// 2. **Genuinely-unbatched real-axis slice**: when the sliced tensor is a
    ///    constant (a weight/evaluated constant never carried a batch dim — same
    ///    reasoning as `resolve_gather_axis` for embedding tables, #3312), ONNX axis 0
    ///    is a real data axis. We emit a `Slice` on internal axis 0.
    ///
    /// 2b. **1-D non-constant activation**: when the sliced tensor's recorded ONNX
    ///    shape is one-dimensional, there is no batch axis to strip — the model is
    ///    genuinely unbatched at this tensor (e.g. a `Reshape`/`Flatten` collapsed the
    ///    detection head to a flat `[N]` vector, or the graph input itself is `[N]`).
    ///    This is exactly the convention `unbatched_target_shape_from_onnx_shape`
    ///    encodes: a length-≤1 ONNX shape is kept as-is, never batch-stripped. So ONNX
    ///    axis 0 of a 1-D tensor is the only, real data axis and maps directly to
    ///    internal axis 0. This is the cctsdb_yolo_2023 pattern (input `[12296]`,
    ///    `Slice[0:12288)`). Slicing the correct real axis is exact and sound.
    ///
    /// 3. **Ambiguous / unsound**: a genuine partial slice of a multi-dimensional
    ///    non-constant tensor's batch dimension cannot be performed soundly in
    ///    unbatched mode (we would be dropping batch elements that do not exist
    ///    internally). We keep a clear error rather than guess.
    ///
    /// `start_raw`/`end_raw` are the raw ONNX indices (`end_raw == i64::MAX` is the
    /// "to end" sentinel); both are already validated non-negative by the callers.
    fn resolve_batch_axis_slice(
        &self,
        spec: &LayerSpec,
        start_raw: i64,
        end_raw: i64,
        step: i64,
        mode: &str,
    ) -> Result<Layer> {
        // ONNX shape of the sliced data input (retains the batch dimension). Axis 0
        // is the batch extent we are reasoning about.
        let batch_extent = spec
            .inputs
            .first()
            .and_then(|name| self.tensor_shapes.get(name))
            .and_then(|shape| shape.first().copied());

        // Case 1: no-op slice covering the entire axis-0 extent → identity.
        let covers_full_axis =
            end_raw == i64::MAX || matches!(batch_extent, Some(b) if b > 0 && end_raw >= b);
        if start_raw == 0 && step == 1 && covers_full_axis {
            debug!(
                "Converting Slice '{}' ({mode}) axis=0 full-coverage [{}:{}) on batch extent {:?} \
                 → identity (no-op, MulConstant(1.0))",
                spec.name, start_raw, end_raw, batch_extent
            );
            return Ok(Layer::MulConstant(MulConstantLayer::scalar(1.0)));
        }

        // Case 2: constant data tensor never carried a batch dim → axis 0 is a real
        // data axis; slice it directly (mirrors resolve_gather_axis, #3312).
        let data_is_constant = spec
            .inputs
            .first()
            .is_some_and(|name| self.is_constant(name));
        if data_is_constant {
            let start = usize::try_from(start_raw).map_err(|_| {
                NyError::ModelLoad(format!(
                    "Slice '{}': start {} out of usize range",
                    spec.name, start_raw
                ))
            })?;
            // INT64_MAX sentinel is already handled by Case 1; a constant partial
            // slice here has a concrete end. Clamp the sentinel defensively.
            let end = if end_raw == i64::MAX {
                usize::MAX
            } else {
                usize::try_from(end_raw).map_err(|_| {
                    NyError::ModelLoad(format!(
                        "Slice '{}': end {} out of usize range",
                        spec.name, end_raw
                    ))
                })?
            };
            debug!(
                "Converting Slice '{}' ({mode}) axis=0 on constant data → real data-axis \
                 Slice(axis=0, {}:{})",
                spec.name, start, end
            );
            return Ok(Layer::Slice(SliceLayer::new(0, start, end)));
        }

        // Case 2b: a 1-D non-constant activation has no batch axis to strip. Per the
        // `unbatched_target_shape_from_onnx_shape` convention in this file, a length-≤1
        // ONNX shape is kept verbatim (never batch-stripped), so ONNX axis 0 is the
        // only, genuine data axis. Map it directly to internal axis 0. This soundly
        // handles flattened detection heads (cctsdb_yolo_2023: input `[12296]`, slice
        // `[0:12288)`). We require a *known* 1-D shape — an absent shape stays ambiguous
        // and falls through to the conservative error below.
        let input_ndim = spec
            .inputs
            .first()
            .and_then(|name| self.tensor_shapes.get(name))
            .map(|shape| shape.len());
        if input_ndim == Some(1) {
            let start = usize::try_from(start_raw).map_err(|_| {
                NyError::ModelLoad(format!(
                    "Slice '{}': start {} out of usize range",
                    spec.name, start_raw
                ))
            })?;
            // INT64_MAX sentinel (full coverage) is already handled by Case 1; a partial
            // 1-D slice here has a concrete end. Clamp the sentinel defensively.
            let end = if end_raw == i64::MAX {
                usize::MAX
            } else {
                usize::try_from(end_raw).map_err(|_| {
                    NyError::ModelLoad(format!(
                        "Slice '{}': end {} out of usize range",
                        spec.name, end_raw
                    ))
                })?
            };
            debug!(
                "Converting Slice '{}' ({mode}) axis=0 on 1-D non-constant tensor (extent {:?}) \
                 → real data-axis Slice(axis=0, {}:{})",
                spec.name, batch_extent, start, end
            );
            return Ok(Layer::Slice(SliceLayer::new(0, start, end)));
        }

        // Case 3: a genuine partial slice of a non-constant batch dimension — unsound
        // to perform in unbatched mode. Keep a clear error; never silently drop.
        Err(NyError::ModelLoad(format!(
            "Slice '{}': axis=0 targets batch dimension which does not exist in unbatched mode \
             (partial batch slice [{}:{}) on extent {:?} is not a no-op and the data is not a \
             constant data axis)",
            spec.name, start_raw, end_raw, batch_extent
        )))
    }

    /// Exact affine lowering for a variable-start Slice of a constant
    /// arithmetic progression (#cctsdb B3a).
    ///
    /// `Slice(range, starts=[x], ends=[x+w], axes=[0], steps=[1])` over a
    /// rank-1 constant `range[k] = r0 + step*k` has
    /// `out[k] = range[x+k] = step*x + (r0 + step*k)` — an AFFINE function of
    /// the scalar starts activation, lowered to a `Linear` layer
    /// (weight = step column, bias = r0 + step*k). Exact and CROWN-compatible.
    ///
    /// The static extent `w` comes from the output tensor shape recorded by
    /// the affine-extent const-fold pass (B2). Right-edge caveat: when the
    /// true graph clamps (`x + w > len`), the runtime output has fewer
    /// elements; this lowering emits the UNCLAMPED extrapolation for the
    /// extra positions — the out-of-range sentinel value (`len` for a 0..len
    /// range), which the bounded-index ScatterND (B4) rejects, matching
    /// exactly the writes the clipped graph performs. See the design's B3/B4
    /// exactness argument.
    ///
    /// Returns `Ok(None)` when the pattern does not apply (caller keeps the
    /// conservative reject).
    fn try_lower_variable_start_slice_of_progression(
        &self,
        spec: &LayerSpec,
    ) -> Result<Option<Layer>> {
        let data_name = &spec.inputs[0];
        let starts_name = &spec.inputs[1];

        // Data: rank-1 constant arithmetic progression with at least 2 elements.
        let Some(data) = self.constant_value(data_name) else {
            return Ok(None);
        };
        if data.ndim() != 1 || data.len() < 2 {
            return Ok(None);
        }
        let values: Vec<f32> = data.iter().copied().collect();
        let step = values[1] - values[0];
        let is_progression = values
            .windows(2)
            .all(|pair| (pair[1] - pair[0] - step).abs() == 0.0);
        if !is_progression {
            return Ok(None);
        }

        // Starts: a 1-element activation (scalar index). Shape must be known.
        let starts_is_scalar = self
            .tensor_shapes
            .get(starts_name)
            .is_some_and(|shape| shape.iter().product::<i64>() == 1 && shape.len() <= 1);
        if !starts_is_scalar {
            return Ok(None);
        }

        // Axes: the single axis 0 of the rank-1 data (default or explicit).
        if let Some(axes_name) = spec.inputs.get(3).filter(|name| !name.is_empty()) {
            let Some(axes) = self.constant_value(axes_name) else {
                return Ok(None);
            };
            let axes: Vec<f32> = axes.iter().copied().collect();
            if axes.len() != 1 || (axes[0] != 0.0 && axes[0] != -1.0) {
                return Ok(None);
            }
        }
        // Steps: 1 (or absent).
        if let Some(steps_name) = spec.inputs.get(4).filter(|name| !name.is_empty()) {
            let Some(steps) = self.constant_value(steps_name) else {
                return Ok(None);
            };
            let steps: Vec<f32> = steps.iter().copied().collect();
            if steps != vec![1.0] {
                return Ok(None);
            }
        }

        // Static extent w from the recorded output shape (B2 affine pass).
        let Some(output_name) = spec.outputs.first() else {
            return Ok(None);
        };
        let Some(output_shape) = self.tensor_shapes.get(output_name) else {
            return Ok(None);
        };
        if output_shape.len() != 1 || output_shape[0] <= 0 {
            return Ok(None);
        }
        let extent = output_shape[0] as usize;
        if extent > data.len() {
            return Ok(None);
        }

        // out[k] = step * x + (r0 + step * k), computed in f64 for exact
        // integer-valued biases.
        let r0 = values[0] as f64;
        let step_f64 = step as f64;
        let weight = ndarray::Array2::<f32>::from_shape_fn((extent, 1), |(_, _)| step_f64 as f32);
        let bias =
            ndarray::Array1::<f32>::from_shape_fn(extent, |k| (r0 + step_f64 * k as f64) as f32);
        let linear = ny_propagate::layers::LinearLayer::new(weight, Some(bias))?;
        debug!(
            "Slice '{}': lowered variable-start slice of arithmetic progression \
             (r0={}, step={}, extent={}) to exact affine Linear",
            spec.name, values[0], step, extent
        );
        Ok(Some(Layer::Linear(linear)))
    }

    pub(crate) fn convert_slice(&self, spec: &LayerSpec) -> Result<Layer> {
        // Slice layer: extracts contiguous range along specified axis
        // Two formats supported:
        // 1. Attribute-based (older opset / from Split): "axis", "start", "end"
        // 2. Input-based (ONNX opset 10+):
        //    - input[0]: data (tensor to slice)
        //    - input[1]: starts (1D tensor)
        //    - input[2]: ends (1D tensor)
        //    - input[3]: axes (optional 1D tensor, default all axes starting from 0)
        //    - input[4]: steps (optional 1D tensor, default 1)
        //
        // Try attribute-based first (from Split op conversion)
        if let (
            Some(AttributeValue::Int(a)),
            Some(AttributeValue::Int(s)),
            Some(AttributeValue::Int(e)),
        ) = (
            spec.attributes.get("axis"),
            spec.attributes.get("start"),
            spec.attributes.get("end"),
        ) {
            return build_attribute_slice_layer(self, spec, *a, *s, *e, "attribute-based");
        }

        // Opset 1-9 Slice stores plural attribute names (axes, starts, ends).
        if let (Some(AttributeValue::Ints(starts_list)), Some(AttributeValue::Ints(ends_list))) =
            (spec.attributes.get("starts"), spec.attributes.get("ends"))
        {
            let axes_list = match spec.attributes.get("axes") {
                Some(AttributeValue::Ints(values)) => values.clone(),
                None => (0..starts_list.len() as i64).collect(),
                _ => {
                    return Err(NyError::ModelLoad(format!(
                        "Slice '{}': invalid plural attribute encoding for axes",
                        spec.name
                    )));
                }
            };
            if starts_list.len() != 1 || ends_list.len() != 1 || axes_list.len() != 1 {
                return Err(NyError::ModelLoad(format!(
                    "Slice '{}': multi-axis opset 9 Slice not supported for propagation (axes={axes_list:?})",
                    spec.name
                )));
            }
            return build_attribute_slice_layer(
                self,
                spec,
                axes_list[0],
                starts_list[0],
                ends_list[0],
                "opset-9 attribute-based",
            );
        }

        // Input-based Slice (ONNX opset 10+)
        // inputs: [data, starts, ends, axes?, steps?]
        if spec.inputs.len() >= 3 {
            let starts_name = &spec.inputs[1];
            let ends_name = &spec.inputs[2];
            // Get starts tensor (input[1])
            let starts = match self.constant_value(starts_name) {
                Some(starts) => starts,
                None => {
                    // Variable-start Slice of a constant arithmetic
                    // progression (#cctsdb B3a): exact affine lowering.
                    if let Some(layer) = self.try_lower_variable_start_slice_of_progression(spec)? {
                        return Ok(layer);
                    }
                    return Err(NyError::UnsupportedOp(format!(
                        "Slice '{}': starts input '{}' not found in constant values (needs constant folding)",
                        spec.name, starts_name
                    )));
                }
            };

            // Get ends tensor (input[2])
            let ends = self.constant_value(ends_name).ok_or_else(|| {
                NyError::UnsupportedOp(format!(
                    "Slice '{}': ends input '{}' not found in constant values (needs constant folding)",
                    spec.name, ends_name
                ))
            })?;

            // Get axes tensor (input[3], optional)
            let axes = spec.inputs.get(3).and_then(|name| {
                if name.is_empty() {
                    None
                } else {
                    self.constant_value(name)
                }
            });

            // Get steps tensor (input[4], optional)
            let steps = spec.inputs.get(4).and_then(|name| {
                if name.is_empty() {
                    None
                } else {
                    self.constant_value(name)
                }
            });

            // For now, we only support single-axis slicing with step=1
            // This is the most common case in VNN-COMP models
            if starts.len() != 1 || ends.len() != 1 {
                return Err(NyError::ModelLoad(format!(
                    "Slice '{}': only single-axis slicing supported (got {} axes)",
                    spec.name,
                    starts.len()
                )));
            }

            if let Some(axes_arr) = axes.as_ref() {
                if axes_arr.len() > 1 {
                    return Err(NyError::ModelLoad(format!(
                        "Slice '{}': only single-axis slicing supported (got {} axes entries)",
                        spec.name,
                        axes_arr.len()
                    )));
                }
            }

            if let Some(steps_arr) = steps.as_ref() {
                if steps_arr.len() > 1 {
                    return Err(NyError::ModelLoad(format!(
                        "Slice '{}': only single-axis slicing supported (got {} steps entries)",
                        spec.name,
                        steps_arr.len()
                    )));
                }
            }

            // Constant-folded Slice bounds can lose their original ONNX integer type
            // when they round-trip through f32 storage. Match the converter's
            // truncation behavior for finite values so fixed-aux shape chains keep
            // folding instead of leaking into graph inputs (#3500).
            let start_f32 = starts.iter().next().copied().unwrap_or(0.0);
            let end_f32 = ends.iter().next().copied().unwrap_or(f32::INFINITY);
            let start = parse_slice_scalar_value(spec, "start", start_f32, false)?;
            let end = parse_slice_scalar_value(spec, "end", end_f32, true)?;

            // Get axis (default 0 if not specified) — resolve before normalizing
            // negative indices since we need the axis to find the dimension size.
            let axis_f32 = axes.as_ref().and_then(|arr| arr.iter().next().copied());
            let axis = axis_f32
                .map(|v| parse_slice_scalar_value(spec, "axis", v, false))
                .transpose()?
                .map(|v| {
                    i32::try_from(v).map_err(|_| {
                        NyError::ModelLoad(format!(
                            "Slice '{}': axis value {} out of valid range",
                            spec.name, v
                        ))
                    })
                })
                .transpose()?
                .unwrap_or(0);

            // Normalize negative start/end indices per ONNX spec: if negative,
            // add the dimension size along the slice axis. This matches the
            // normalization already used in const_fold/ops/slice.rs.
            let mut end_offset_from_axis_end = None;
            let (start, end) = if start < 0 || end < 0 {
                let data_input = &spec.inputs[0];
                // Static (positive) dimension along the slice axis, when the
                // inferred shape carries one. A shape may be present but hold
                // a dynamic marker (e.g. -1) at the slice axis; that case is
                // handled like a missing shape below.
                let static_dim = self.tensor_shapes.get(data_input).and_then(|input_shape| {
                    let ndim = input_shape.len() as i64;
                    let resolved_axis = if (axis as i64) < 0 {
                        (ndim + axis as i64) as usize
                    } else {
                        axis as usize
                    };
                    input_shape.get(resolved_axis).copied().filter(|&d| d > 0)
                });
                if let Some(dim) = static_dim {
                    let norm_start = if start < 0 {
                        (dim + start).max(0)
                    } else {
                        start.min(dim)
                    };
                    let norm_end = if end < 0 {
                        (dim + end).max(0)
                    } else if end == i64::MAX {
                        dim
                    } else {
                        end.min(dim)
                    };
                    debug!(
                        "Slice '{}': normalized negative indices: start {} -> {}, end {} -> {} (dim={})",
                        spec.name, start, norm_start, end, norm_end, dim
                    );
                    (norm_start, norm_end)
                } else if start >= 0 && end < 0 {
                    // No static dim (shape unknown, axis out of range, or the
                    // axis dimension is dynamic): a non-negative start with a
                    // negative end is still exactly representable by counting
                    // the end offset from the axis end at propagation time.
                    end_offset_from_axis_end = Some(usize::try_from(-end).map_err(|_| {
                        NyError::ModelLoad(format!(
                            "Slice '{}': negative end {} out of valid range",
                            spec.name, end
                        ))
                    })?);
                    (start, i64::MAX)
                } else {
                    return Err(NyError::ModelLoad(format!(
                        "Slice '{}': negative start requires a static input dimension for '{}' \
                         along axis {}, but the shape is unknown or dynamic there",
                        spec.name, data_input, axis
                    )));
                }
            } else {
                (start, end)
            };

            // Verify step is 1 (or not specified)
            if let Some(steps_arr) = steps.as_ref() {
                let step_f32 = steps_arr.iter().next().copied().unwrap_or(1.0);
                let step = parse_slice_scalar_value(spec, "step", step_f32, false)?;
                if step != 1 {
                    return Err(NyError::ModelLoad(format!(
                        "Slice '{}': only step=1 supported (got step={})",
                        spec.name, step
                    )));
                }
            }

            // start/end are guaranteed non-negative after normalization above.
            let start_usize = start as usize;
            let end_usize = end as usize;

            // Adjust axis for unbatched operation: ny-propagate strips batch dimension.
            // ONNX axis >= 1 becomes axis - 1 in unbatched mode (same as Squeeze/Unsqueeze).
            // ONNX axis == 0 targets batch dimension — handle no-op / real-axis / error
            // cases via resolve_batch_axis_slice (start/end are unchanged for the
            // non-negative inputs this branch cares about; the INT64_MAX "to end"
            // sentinel is preserved when end was not negative).
            if self.model_unbatched {
                // Unbatched model (#cctsdb B5): ONNX axes are verbatim data axes.
                debug!(
                    "Converting Slice '{}' (input-based) unbatched model: ONNX axis={} verbatim, \
                     start={}, end={}",
                    spec.name, axis, start_usize, end_usize
                );
                let slice = if let Some(end_offset) = end_offset_from_axis_end {
                    SliceLayer::new_with_end_offset(axis, start_usize, end_offset)
                } else {
                    SliceLayer::new(axis, start_usize, end_usize)
                };
                return Ok(Layer::Slice(slice));
            }
            if axis == 0 {
                if end_offset_from_axis_end.is_some() {
                    return Err(NyError::UnsupportedOp(format!(
                        "Slice '{}': axis=0 with relative negative end is not supported in unbatched mode",
                        spec.name
                    )));
                }
                return self.resolve_batch_axis_slice(spec, start, end, 1, "input-based");
            }
            // Trailing-relative remap: correct under both internal runtime
            // layouts (leading batch dim stripped OR retained); refuses
            // ambiguous cases. See `ConvertContext::remap_axis_trailing`.
            let data_name = spec.inputs.first().map(String::as_str).ok_or_else(|| {
                NyError::ModelLoad(format!("Slice '{}' has no data input", spec.name))
            })?;
            let adjusted_axis = self.remap_axis_trailing(
                "Slice",
                &spec.name,
                data_name,
                i64::from(axis),
                crate::convert::LegacyBatchAxisPolicy::RejectZero,
            )?;
            let adjusted_axis = i32::try_from(adjusted_axis).map_err(|_| {
                NyError::ModelLoad(format!(
                    "Slice '{}': remapped axis {} does not fit i32",
                    spec.name, adjusted_axis
                ))
            })?;
            debug!(
                "Converting Slice '{}' (input-based) with ONNX axis={}, adjusted axis={}, start={}, end={}",
                spec.name, axis, adjusted_axis, start_usize, end_usize
            );
            let slice = if let Some(end_offset) = end_offset_from_axis_end {
                SliceLayer::new_with_end_offset(adjusted_axis, start_usize, end_offset)
            } else {
                SliceLayer::new(adjusted_axis, start_usize, end_usize)
            };
            return Ok(Layer::Slice(slice));
        }

        // Neither attribute-based nor input-based worked
        Err(NyError::ModelLoad(format!(
            "Slice '{}': requires either attributes (axis, start, end) or inputs (data, starts, ends)",
            spec.name
        )))
    }
    pub(crate) fn convert_transpose(&self, spec: &LayerSpec) -> Result<Layer> {
        // ONNX Transpose: has 'perm' attribute specifying the permutation
        let axes = match spec.attributes.get("perm") {
            Some(AttributeValue::Ints(perm)) => {
                // #2983: Guard negative permutation indices.
                perm.iter()
                    .map(|&v| {
                        usize::try_from(v).map_err(|_| {
                            NyError::ModelLoad(format!(
                                "Transpose '{}': negative perm index {}",
                                spec.name, v
                            ))
                        })
                    })
                    .collect::<Result<Vec<usize>>>()?
            }
            _ => {
                // Default: reverse all dimensions
                Vec::new() // TransposeLayer handles empty axes as batched transpose
            }
        };

        Ok(Layer::Transpose(TransposeLayer::new(axes)))
    }

    /// Adjust an ONNX Squeeze axis for unbatched propagation.
    ///
    /// Delegates to [`ConvertContext::remap_axis_trailing`]: trailing-relative
    /// (negative) axes select the same size-1 dimension whether the runtime
    /// tensor kept its ONNX rank or had its leading batch dim stripped; the
    /// legacy `axis - 1` guess was only correct for the stripped layout.
    /// `onnx_axis == 0` on a rank>1 input targets the (possibly stripped)
    /// batch dimension and refuses conversion, as before (#2310); on a
    /// rank-≤1 input it is a genuine data axis (Slice Case 2b convention).
    /// Load-time axis interpretation only — no bound math.
    fn adjust_squeeze_axis_for_unbatched(&self, onnx_axis: i32, spec: &LayerSpec) -> Result<i32> {
        let data_name = spec.inputs.first().map(String::as_str).ok_or_else(|| {
            NyError::ModelLoad(format!("Squeeze '{}' has no data input", spec.name))
        })?;
        let remapped = self.remap_axis_trailing(
            "Squeeze",
            &spec.name,
            data_name,
            i64::from(onnx_axis),
            crate::convert::LegacyBatchAxisPolicy::RejectZero,
        )?;
        i32::try_from(remapped).map_err(|_| {
            NyError::ModelLoad(format!(
                "Squeeze '{}': remapped axis {remapped} does not fit i32",
                spec.name
            ))
        })
    }

    pub(crate) fn convert_squeeze(&self, spec: &LayerSpec) -> Result<Layer> {
        // ONNX Squeeze: removes dimension of size 1 at specified axis/axes
        // Opset < 13: axes in 'axes' attribute (list of ints)
        // Opset >= 13: axes from second input tensor
        //
        // We only support single-axis squeeze since SqueezeLayer takes a single axis.
        // For multi-axis squeeze, the model loader should split into multiple ops.

        // Try attribute-based first (opset < 13)
        if let Some(AttributeValue::Ints(axes)) = spec.attributes.get("axes") {
            if axes.len() != 1 {
                return Err(NyError::ModelLoad(format!(
                    "Squeeze '{}': only single-axis supported (got {} axes)",
                    spec.name,
                    axes.len()
                )));
            }
            // SAFETY(#3100): i64->i32 checked conversion
            let onnx_axis = i32::try_from(axes[0]).map_err(|_| {
                NyError::ModelLoad(format!(
                    "Squeeze '{}': axis {} out of i32 range",
                    spec.name, axes[0]
                ))
            })?;

            // Adjust for unbatched operation: ny-propagate doesn't have batch dimension
            // ONNX axis=1 means dimension after batch, which is axis=0 unbatched
            // ONNX axis=0 targets the batch dimension, which does not exist in unbatched mode.
            // Ref: alpha-beta-CROWN raises ValueError("Squeezing with axes == 0 is not allowed")
            // in BoundSqueeze.bound_backward (reshape.py:265). (#2310)
            let adjusted_axis = self.adjust_squeeze_axis_for_unbatched(onnx_axis, spec)?;

            debug!(
                "Converting Squeeze '{}' (attribute-based) with ONNX axis={}, adjusted axis={}",
                spec.name, onnx_axis, adjusted_axis
            );
            return Ok(Layer::Squeeze(SqueezeLayer::new(adjusted_axis)));
        }

        // Input-based (opset >= 13): axes from second input
        if spec.inputs.len() >= 2 {
            let axes_name = &spec.inputs[1];
            if let Some(axes_tensor) = self.weights.get(axes_name) {
                if axes_tensor.len() != 1 {
                    return Err(NyError::ModelLoad(format!(
                        "Squeeze '{}': only single-axis supported (got {} axes from input)",
                        spec.name,
                        axes_tensor.len()
                    )));
                }
                let axis_f32 = axes_tensor.iter().next().copied().unwrap_or(0.0);
                // SAFETY(#3100): Guard against NaN/Inf/out-of-range f32 axis values
                // before casting to i32. Finiteness + range check: a finite f32 like
                // 1e20 would saturate `as i32` to i32::MAX (garbage axis).
                if !axis_f32.is_finite() || axis_f32 > i32::MAX as f32 || axis_f32 < i32::MIN as f32
                {
                    return Err(NyError::ModelLoad(format!(
                        "Squeeze '{}': axis value {} out of valid range",
                        spec.name, axis_f32
                    )));
                }
                let onnx_axis = axis_f32 as i32;

                // Adjust for unbatched operation
                // ONNX axis=0 targets the batch dimension — reject unless it is a genuine
                // rank-≤1 data axis. (#2310)
                let adjusted_axis = self.adjust_squeeze_axis_for_unbatched(onnx_axis, spec)?;

                debug!(
                    "Converting Squeeze '{}' (input-based) with ONNX axis={}, adjusted axis={}",
                    spec.name, onnx_axis, adjusted_axis
                );
                return Ok(Layer::Squeeze(SqueezeLayer::new(adjusted_axis)));
            }
        }

        // ONNX Squeeze treats missing axes as "remove every dimension of size 1".
        // Lowering that directly to a multi-axis Squeeze would widen the layer surface, so
        // use the known output shape when shape info is available.
        if let Some(output_name) = spec.outputs.first() {
            if let Some(onnx_output_shape) = self.tensor_shapes.get(output_name) {
                let target_shape = if self.model_unbatched {
                    verbatim_target_shape_from_onnx_shape(onnx_output_shape)
                } else {
                    unbatched_target_shape_from_onnx_shape(onnx_output_shape)
                };
                debug!(
                    "Converting Squeeze '{}' (implicit axes) using ONNX output shape {:?}, unbatched shape {:?}",
                    spec.name, onnx_output_shape, target_shape
                );
                return Ok(Layer::Reshape(ReshapeLayer::new(target_shape)));
            }
        }

        Err(NyError::ModelLoad(format!(
            "Squeeze '{}': requires explicit axes or a known output shape",
            spec.name
        )))
    }

    /// Adjust an ONNX Unsqueeze axis for unbatched propagation.
    ///
    /// The ONNX Unsqueeze axis is defined w.r.t. the OUTPUT rank (input
    /// rank + 1), so the trailing-relative remap resolves against the
    /// recorded ONNX shape of the OUTPUT tensor: the remapped negative axis
    /// selects the same insertion point whether the runtime tensor kept its
    /// ONNX rank or had its leading batch dim stripped (the runtime layers
    /// resolve Unsqueeze axes against the runtime OUTPUT rank). The legacy
    /// `axis - 1` guess was only correct for the stripped layout.
    ///
    /// `onnx_axis == 0` keeps the historical "insert a new leading
    /// dimension" behavior (shape-computation subgraphs on scalars, e.g.
    /// lsnc quadrotor2d_output) rather than refusing.
    fn adjust_unsqueeze_axis_for_unbatched(&self, onnx_axis: i32, spec: &LayerSpec) -> Result<i32> {
        if self.model_unbatched {
            // Unbatched model (#cctsdb B5): ONNX axes are verbatim data axes.
            return Ok(onnx_axis);
        }
        if onnx_axis == 0 {
            debug!(
                "Unsqueeze '{}': axis=0 in unbatched mode — inserting leading dimension (not batch dim)",
                spec.name
            );
            return Ok(0);
        }
        let output_name = spec.outputs.first().map(String::as_str).ok_or_else(|| {
            NyError::ModelLoad(format!("Unsqueeze '{}' has no output", spec.name))
        })?;
        let remapped = self.remap_axis_trailing(
            "Unsqueeze",
            &spec.name,
            output_name,
            i64::from(onnx_axis),
            crate::convert::LegacyBatchAxisPolicy::RejectZero,
        )?;
        i32::try_from(remapped).map_err(|_| {
            NyError::ModelLoad(format!(
                "Unsqueeze '{}': remapped axis {remapped} does not fit i32",
                spec.name
            ))
        })
    }

    pub(crate) fn convert_unsqueeze(&self, spec: &LayerSpec) -> Result<Layer> {
        // ONNX Unsqueeze: inserts dimension of size 1 at specified axis/axes
        // Opset < 13: axes in 'axes' attribute (list of ints)
        // Opset >= 13: axes from second input tensor
        //
        // We only support single-axis unsqueeze since UnsqueezeLayer takes a single axis.
        // For multi-axis unsqueeze, the model loader should split into multiple ops.

        // Try attribute-based first (opset < 13)
        if let Some(AttributeValue::Ints(axes)) = spec.attributes.get("axes") {
            if axes.len() != 1 {
                return Err(NyError::ModelLoad(format!(
                    "Unsqueeze '{}': only single-axis supported (got {} axes)",
                    spec.name,
                    axes.len()
                )));
            }
            // SAFETY(#3100): i64->i32 checked conversion
            let onnx_axis = i32::try_from(axes[0]).map_err(|_| {
                NyError::ModelLoad(format!(
                    "Unsqueeze '{}': axis {} out of i32 range",
                    spec.name, axes[0]
                ))
            })?;

            // Adjust for unbatched operation: trailing-relative remap against
            // the recorded OUTPUT rank (ONNX Unsqueeze axes are output-rank
            // relative); axis=0 keeps the "insert leading dim" behavior for
            // shape-computation subgraphs. See adjust_unsqueeze_axis_for_unbatched.
            let adjusted_axis = self.adjust_unsqueeze_axis_for_unbatched(onnx_axis, spec)?;

            debug!(
                "Converting Unsqueeze '{}' (attribute-based) with ONNX axis={}, adjusted axis={}",
                spec.name, onnx_axis, adjusted_axis
            );
            return Ok(Layer::Unsqueeze(UnsqueezeLayer::new(adjusted_axis)));
        }

        // Input-based (opset >= 13): axes from second input
        if spec.inputs.len() >= 2 {
            let axes_name = &spec.inputs[1];
            if let Some(axes_tensor) = self.weights.get(axes_name) {
                if axes_tensor.len() != 1 {
                    return Err(NyError::ModelLoad(format!(
                        "Unsqueeze '{}': only single-axis supported (got {} axes from input)",
                        spec.name,
                        axes_tensor.len()
                    )));
                }
                let axis_f32 = axes_tensor.iter().next().copied().unwrap_or(0.0);
                // SAFETY(#3100): Guard against NaN/Inf/out-of-range f32 axis values
                // before casting to i32. Range check prevents saturation on large finite values.
                if !axis_f32.is_finite() || axis_f32 > i32::MAX as f32 || axis_f32 < i32::MIN as f32
                {
                    return Err(NyError::ModelLoad(format!(
                        "Unsqueeze '{}': axis value {} out of valid range",
                        spec.name, axis_f32
                    )));
                }
                let onnx_axis = axis_f32 as i32;

                // Adjust for unbatched operation: trailing-relative remap
                // against the recorded OUTPUT rank; axis=0 keeps the "insert
                // leading dim" behavior (see attribute-based path comment).
                let adjusted_axis = self.adjust_unsqueeze_axis_for_unbatched(onnx_axis, spec)?;

                debug!(
                    "Converting Unsqueeze '{}' (input-based) with ONNX axis={}, adjusted axis={}",
                    spec.name, onnx_axis, adjusted_axis
                );
                return Ok(Layer::Unsqueeze(UnsqueezeLayer::new(adjusted_axis)));
            }
        }

        // Unsqueeze always requires explicit axes
        Err(NyError::ModelLoad(format!(
            "Unsqueeze '{}': requires explicit axes (attribute or input)",
            spec.name
        )))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    use ndarray::arr1;
    use ny_core::{LayerType, NyError};
    use ny_propagate::Layer;

    use super::ConvertContext;
    use crate::{AttributeValue, LayerSpec, WeightStore};

    /// Build a ConvertContext from a WeightStore.
    fn make_context(weights: &WeightStore) -> ConvertContext<'_> {
        static SHAPES: std::sync::LazyLock<HashMap<String, Vec<i64>>> =
            std::sync::LazyLock::new(HashMap::new);
        static CONSTANTS: std::sync::LazyLock<HashSet<String>> =
            std::sync::LazyLock::new(HashSet::new);
        ConvertContext::new(weights, &SHAPES, &CONSTANTS)
    }

    fn make_context_with_evaluated<'a>(
        weights: &'a WeightStore,
        evaluated: &'a HashMap<String, ndarray::ArrayD<f32>>,
    ) -> ConvertContext<'a> {
        static SHAPES: std::sync::LazyLock<HashMap<String, Vec<i64>>> =
            std::sync::LazyLock::new(HashMap::new);
        static CONSTANTS: std::sync::LazyLock<HashSet<String>> =
            std::sync::LazyLock::new(HashSet::new);
        ConvertContext::with_evaluated_constants(weights, &SHAPES, &CONSTANTS, evaluated)
    }

    fn make_context_with_shapes<'a>(
        weights: &'a WeightStore,
        shapes: &'a HashMap<String, Vec<i64>>,
    ) -> ConvertContext<'a> {
        static CONSTANTS: std::sync::LazyLock<HashSet<String>> =
            std::sync::LazyLock::new(HashSet::new);
        ConvertContext::new(weights, shapes, &CONSTANTS)
    }

    /// Build a `LayerSpec` with given attributes (attribute-based path, opset < 13).
    fn attr_spec(name: &str, layer_type: LayerType, axes: Vec<i64>) -> LayerSpec {
        LayerSpec {
            name: name.to_string(),
            layer_type,
            inputs: vec!["data".to_string()],
            outputs: vec!["output".to_string()],
            weights: None,
            attributes: HashMap::from([("axes".to_string(), AttributeValue::Ints(axes))]),
        }
    }

    /// Build a `LayerSpec` for the input-based path (opset >= 13).
    fn input_spec(name: &str, layer_type: LayerType) -> LayerSpec {
        LayerSpec {
            name: name.to_string(),
            layer_type,
            inputs: vec!["data".to_string(), "axes_tensor".to_string()],
            outputs: vec!["output".to_string()],
            weights: None,
            attributes: HashMap::new(),
        }
    }

    /// Build a `LayerSpec` with no attributes and no axes input (missing axes).
    fn bare_spec(name: &str, layer_type: LayerType) -> LayerSpec {
        LayerSpec {
            name: name.to_string(),
            layer_type,
            inputs: vec!["data".to_string()],
            outputs: vec!["output".to_string()],
            weights: None,
            attributes: HashMap::new(),
        }
    }

    fn squeeze_axis(layer: &Layer) -> i32 {
        match layer {
            Layer::Squeeze(sq) => sq.axis,
            other => unreachable!("expected Layer::Squeeze, got {:?}", other),
        }
    }

    fn unsqueeze_axis(layer: &Layer) -> i32 {
        match layer {
            Layer::Unsqueeze(usq) => usq.axis,
            other => unreachable!("expected Layer::Unsqueeze, got {:?}", other),
        }
    }

    // ---- Squeeze: attribute-based (opset < 13) ----

    #[test]
    fn squeeze_attr_axis_1_remaps_trailing() {
        // ONNX axis 1 on a recorded rank-3 [1,1,6] tensor -> trailing-relative -2,
        // correct whether the runtime tensor is [1,1,6] (retained) or [1,6] (stripped).
        let ws = WeightStore::new();
        let shapes = HashMap::from([("data".to_string(), vec![1_i64, 1, 6])]);
        let m = make_context_with_shapes(&ws, &shapes);
        let spec = attr_spec("sq", LayerType::Squeeze, vec![1]);
        let layer = m.convert_squeeze(&spec).unwrap();
        assert_eq!(squeeze_axis(&layer), -2);
    }

    #[test]
    fn squeeze_attr_axis_2_remaps_trailing() {
        let ws = WeightStore::new();
        let shapes = HashMap::from([("data".to_string(), vec![1_i64, 4, 1])]);
        let m = make_context_with_shapes(&ws, &shapes);
        let spec = attr_spec("sq", LayerType::Squeeze, vec![2]);
        let layer = m.convert_squeeze(&spec).unwrap();
        assert_eq!(squeeze_axis(&layer), -1);
    }

    #[test]
    fn squeeze_attr_positive_axis_unknown_rank_keeps_legacy() {
        // No recorded shape: legacy `axis - 1` (synthesized-subgraph compat).
        let ws = WeightStore::new();
        let m = make_context(&ws);
        let spec = attr_spec("sq", LayerType::Squeeze, vec![1]);
        let layer = m.convert_squeeze(&spec).unwrap();
        assert_eq!(squeeze_axis(&layer), 0);
    }

    #[test]
    fn squeeze_attr_axis_0_rejected() {
        // ONNX axis 0 targets the batch dimension — rejected in unbatched mode (#2310)
        // Ref: alpha-beta-CROWN raises ValueError("Squeezing with axes == 0 is not allowed")
        let ws = WeightStore::new();
        let shapes = HashMap::from([("data".to_string(), vec![1_i64, 6])]);
        let m = make_context_with_shapes(&ws, &shapes);
        let spec = attr_spec("sq", LayerType::Squeeze, vec![0]);
        let err = m.convert_squeeze(&spec).unwrap_err();
        assert!(
            err.to_string().contains("batch dimension"),
            "Expected batch dimension rejection, got: {err}"
        );
    }

    #[test]
    fn squeeze_attr_axis_0_rank1_genuine_data_axis_loads_and_shapes_correctly() {
        // A rank-1 (no batch axis) input `[1]` with ONNX axis=0 loads WITHOUT the
        // unbatched-mode error: `data_had_batch_axis == Some(false)` → SqueezeLayer(0),
        // which removes the genuine size-1 data axis → scalar shape [].
        let ws = WeightStore::new();
        let shapes = HashMap::from([("data".to_string(), vec![1_i64])]);
        let m = make_context_with_shapes(&ws, &shapes);
        let spec = attr_spec("sq_genuine", LayerType::Squeeze, vec![0]);
        let layer = m
            .convert_squeeze(&spec)
            .expect("axis=0 Squeeze on a genuine rank-1 data axis must load without error");
        // Trailing-relative encoding: the sole data axis is stored as -1.
        assert_eq!(squeeze_axis(&layer), -1);

        let Layer::Squeeze(sq) = &layer else {
            panic!("expected Layer::Squeeze, got {:?}", layer);
        };
        use ny_propagate::layers::BoundPropagation;
        let lower = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[1]), vec![0.0_f32]).unwrap();
        let upper = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[1]), vec![1.0_f32]).unwrap();
        let input = ny_tensor::BoundedTensor::new(lower, upper).unwrap();
        let out = sq
            .propagate_ibp(&input)
            .expect("squeeze ibp on rank-1 size-1 axis");
        assert_eq!(out.shape(), &[] as &[usize]);
    }

    #[test]
    fn squeeze_attr_negative_axis_passthrough() {
        // Negative axes pass through without adjustment
        let ws = WeightStore::new();
        let m = make_context(&ws);
        let spec = attr_spec("sq", LayerType::Squeeze, vec![-1]);
        let layer = m.convert_squeeze(&spec).unwrap();
        assert_eq!(squeeze_axis(&layer), -1);
    }

    #[test]
    fn squeeze_attr_multi_axis_rejected() {
        let ws = WeightStore::new();
        let m = make_context(&ws);
        let spec = attr_spec("sq", LayerType::Squeeze, vec![1, 2]);
        let err = m.convert_squeeze(&spec).unwrap_err();
        assert!(
            err.to_string().contains("only single-axis supported"),
            "Expected multi-axis rejection, got: {err}"
        );
    }

    // ---- Squeeze: input-based (opset >= 13) ----

    #[test]
    fn squeeze_input_axis_1_remaps_trailing() {
        let mut ws = WeightStore::new();
        ws.insert("axes_tensor".to_string(), arr1(&[1.0_f32]).into_dyn());
        let shapes = HashMap::from([("data".to_string(), vec![1_i64, 1, 6])]);
        let m = make_context_with_shapes(&ws, &shapes);
        let spec = input_spec("sq", LayerType::Squeeze);
        let layer = m.convert_squeeze(&spec).unwrap();
        assert_eq!(squeeze_axis(&layer), -2);
    }

    #[test]
    fn squeeze_input_axis_3_remaps_trailing() {
        let mut ws = WeightStore::new();
        ws.insert("axes_tensor".to_string(), arr1(&[3.0_f32]).into_dyn());
        let shapes = HashMap::from([("data".to_string(), vec![1_i64, 4, 5, 1])]);
        let m = make_context_with_shapes(&ws, &shapes);
        let spec = input_spec("sq", LayerType::Squeeze);
        let layer = m.convert_squeeze(&spec).unwrap();
        assert_eq!(squeeze_axis(&layer), -1);
    }

    #[test]
    fn squeeze_input_negative_axis_passthrough() {
        let mut ws = WeightStore::new();
        ws.insert("axes_tensor".to_string(), arr1(&[-2.0_f32]).into_dyn());
        let m = make_context(&ws);
        let spec = input_spec("sq", LayerType::Squeeze);
        let layer = m.convert_squeeze(&spec).unwrap();
        assert_eq!(squeeze_axis(&layer), -2);
    }

    #[test]
    fn squeeze_input_multi_axis_rejected() {
        let mut ws = WeightStore::new();
        ws.insert("axes_tensor".to_string(), arr1(&[1.0_f32, 2.0]).into_dyn());
        let m = make_context(&ws);
        let spec = input_spec("sq", LayerType::Squeeze);
        let err = m.convert_squeeze(&spec).unwrap_err();
        assert!(
            err.to_string().contains("only single-axis supported"),
            "Expected multi-axis rejection, got: {err}"
        );
    }

    #[test]
    fn squeeze_input_axis_0_rejected() {
        // ONNX axis 0 targets the batch dimension — rejected in unbatched mode (#2310)
        let mut ws = WeightStore::new();
        ws.insert("axes_tensor".to_string(), arr1(&[0.0_f32]).into_dyn());
        let shapes = HashMap::from([("data".to_string(), vec![1_i64, 6])]);
        let m = make_context_with_shapes(&ws, &shapes);
        let spec = input_spec("sq", LayerType::Squeeze);
        let err = m.convert_squeeze(&spec).unwrap_err();
        assert!(
            err.to_string().contains("batch dimension"),
            "Expected batch dimension rejection, got: {err}"
        );
    }

    // ---- Squeeze: missing axes ----

    #[test]
    fn squeeze_no_axes_uses_known_output_shape() {
        let ws = WeightStore::new();
        let shapes = HashMap::from([("output".to_string(), Vec::new())]);
        let m = make_context_with_shapes(&ws, &shapes);
        let spec = bare_spec("sq", LayerType::Squeeze);
        let layer = m.convert_squeeze(&spec).unwrap();
        assert!(reshape_target(&layer).is_empty());
    }

    #[test]
    fn squeeze_no_axes_without_known_output_shape_rejected() {
        let ws = WeightStore::new();
        let m = make_context(&ws);
        let spec = bare_spec("sq", LayerType::Squeeze);
        let err = m.convert_squeeze(&spec).unwrap_err();
        assert!(
            err.to_string()
                .contains("requires explicit axes or a known output shape"),
            "Expected missing-axes rejection, got: {err}"
        );
    }

    // ---- Unsqueeze: attribute-based (opset < 13) ----

    #[test]
    fn unsqueeze_attr_axis_1_remaps_trailing() {
        // ONNX Unsqueeze axes are OUTPUT-rank relative: axis 1 with recorded
        // output rank 3 -> trailing-relative -2.
        let ws = WeightStore::new();
        let shapes = HashMap::from([("output".to_string(), vec![1_i64, 1, 6])]);
        let m = make_context_with_shapes(&ws, &shapes);
        let spec = attr_spec("usq", LayerType::Unsqueeze, vec![1]);
        let layer = m.convert_unsqueeze(&spec).unwrap();
        assert_eq!(unsqueeze_axis(&layer), -2);
    }

    #[test]
    fn unsqueeze_attr_axis_2_remaps_trailing() {
        let ws = WeightStore::new();
        let shapes = HashMap::from([("output".to_string(), vec![1_i64, 4, 1])]);
        let m = make_context_with_shapes(&ws, &shapes);
        let spec = attr_spec("usq", LayerType::Unsqueeze, vec![2]);
        let layer = m.convert_unsqueeze(&spec).unwrap();
        assert_eq!(unsqueeze_axis(&layer), -1);
    }

    #[test]
    fn unsqueeze_attr_positive_axis_unknown_rank_keeps_legacy() {
        // No recorded OUTPUT shape: legacy `axis - 1` (synthesized-subgraph compat).
        let ws = WeightStore::new();
        let m = make_context(&ws);
        let spec = attr_spec("usq", LayerType::Unsqueeze, vec![1]);
        let layer = m.convert_unsqueeze(&spec).unwrap();
        assert_eq!(unsqueeze_axis(&layer), 0);
    }

    #[test]
    fn unsqueeze_attr_axis_0_allowed() {
        // ONNX axis 0 — allowed in unbatched mode for shape-computation subgraphs
        // (lsnc quadrotor2d_output uses Unsqueeze(axis=0) on scalars)
        let ws = WeightStore::new();
        let m = make_context(&ws);
        let spec = attr_spec("usq", LayerType::Unsqueeze, vec![0]);
        let layer = m.convert_unsqueeze(&spec).unwrap();
        assert_eq!(unsqueeze_axis(&layer), 0);
    }

    #[test]
    fn unsqueeze_attr_negative_axis_passthrough() {
        let ws = WeightStore::new();
        let m = make_context(&ws);
        let spec = attr_spec("usq", LayerType::Unsqueeze, vec![-1]);
        let layer = m.convert_unsqueeze(&spec).unwrap();
        assert_eq!(unsqueeze_axis(&layer), -1);
    }

    #[test]
    fn unsqueeze_attr_multi_axis_rejected() {
        let ws = WeightStore::new();
        let m = make_context(&ws);
        let spec = attr_spec("usq", LayerType::Unsqueeze, vec![1, 2]);
        let err = m.convert_unsqueeze(&spec).unwrap_err();
        assert!(
            err.to_string().contains("only single-axis supported"),
            "Expected multi-axis rejection, got: {err}"
        );
    }

    // ---- Unsqueeze: input-based (opset >= 13) ----

    #[test]
    fn unsqueeze_input_axis_1_remaps_trailing() {
        let mut ws = WeightStore::new();
        ws.insert("axes_tensor".to_string(), arr1(&[1.0_f32]).into_dyn());
        let shapes = HashMap::from([("output".to_string(), vec![1_i64, 1, 6])]);
        let m = make_context_with_shapes(&ws, &shapes);
        let spec = input_spec("usq", LayerType::Unsqueeze);
        let layer = m.convert_unsqueeze(&spec).unwrap();
        assert_eq!(unsqueeze_axis(&layer), -2);
    }

    #[test]
    fn unsqueeze_input_axis_3_remaps_trailing() {
        let mut ws = WeightStore::new();
        ws.insert("axes_tensor".to_string(), arr1(&[3.0_f32]).into_dyn());
        let shapes = HashMap::from([("output".to_string(), vec![1_i64, 4, 5, 1])]);
        let m = make_context_with_shapes(&ws, &shapes);
        let spec = input_spec("usq", LayerType::Unsqueeze);
        let layer = m.convert_unsqueeze(&spec).unwrap();
        assert_eq!(unsqueeze_axis(&layer), -1);
    }

    #[test]
    fn unsqueeze_input_negative_axis_passthrough() {
        let mut ws = WeightStore::new();
        ws.insert("axes_tensor".to_string(), arr1(&[-1.0_f32]).into_dyn());
        let m = make_context(&ws);
        let spec = input_spec("usq", LayerType::Unsqueeze);
        let layer = m.convert_unsqueeze(&spec).unwrap();
        assert_eq!(unsqueeze_axis(&layer), -1);
    }

    #[test]
    fn unsqueeze_input_multi_axis_rejected() {
        let mut ws = WeightStore::new();
        ws.insert("axes_tensor".to_string(), arr1(&[0.0_f32, 1.0]).into_dyn());
        let m = make_context(&ws);
        let spec = input_spec("usq", LayerType::Unsqueeze);
        let err = m.convert_unsqueeze(&spec).unwrap_err();
        assert!(
            err.to_string().contains("only single-axis supported"),
            "Expected multi-axis rejection, got: {err}"
        );
    }

    #[test]
    fn unsqueeze_input_axis_0_allowed() {
        // ONNX axis 0 — allowed in unbatched mode for shape-computation subgraphs
        let mut ws = WeightStore::new();
        ws.insert("axes_tensor".to_string(), arr1(&[0.0_f32]).into_dyn());
        let m = make_context(&ws);
        let spec = input_spec("usq", LayerType::Unsqueeze);
        let layer = m.convert_unsqueeze(&spec).unwrap();
        assert_eq!(unsqueeze_axis(&layer), 0);
    }

    // ---- Unsqueeze: missing axes ----

    #[test]
    fn unsqueeze_no_axes_rejected() {
        let ws = WeightStore::new();
        let m = make_context(&ws);
        let spec = bare_spec("usq", LayerType::Unsqueeze);
        let err = m.convert_unsqueeze(&spec).unwrap_err();
        assert!(
            err.to_string().contains("requires explicit axes"),
            "Expected missing-axes rejection, got: {err}"
        );
    }

    // ---- Slice: NaN/Inf guard regression tests (#2756) ----

    /// Build a `LayerSpec` for input-based Slice (opset 10+).
    /// `axes_name` provides the axes input; when `None`, axis defaults to 0.
    fn slice_spec_with_axes(
        starts_name: &str,
        ends_name: &str,
        axes_name: Option<&str>,
    ) -> LayerSpec {
        let mut inputs = vec![
            "data".to_string(),
            starts_name.to_string(),
            ends_name.to_string(),
        ];
        if let Some(name) = axes_name {
            inputs.push(name.to_string());
        }
        LayerSpec {
            name: "slice_test".to_string(),
            layer_type: LayerType::Slice,
            inputs,
            outputs: vec!["output".to_string()],
            weights: None,
            attributes: HashMap::new(),
        }
    }

    /// Build a `LayerSpec` for input-based Slice with no axes input (axis defaults to 0).
    fn slice_spec(starts_name: &str, ends_name: &str) -> LayerSpec {
        slice_spec_with_axes(starts_name, ends_name, None)
    }

    fn slice_attr_spec_plural(axes: &[i64], starts: &[i64], ends: &[i64]) -> LayerSpec {
        LayerSpec {
            name: "slice_test".to_string(),
            layer_type: LayerType::Slice,
            inputs: vec!["data".to_string()],
            outputs: vec!["output".to_string()],
            weights: None,
            attributes: HashMap::from([
                ("axes".to_string(), AttributeValue::Ints(axes.to_vec())),
                ("starts".to_string(), AttributeValue::Ints(starts.to_vec())),
                ("ends".to_string(), AttributeValue::Ints(ends.to_vec())),
            ]),
        }
    }

    #[test]
    fn slice_nan_start_rejected() {
        let mut ws = WeightStore::new();
        ws.insert("starts".to_string(), arr1(&[f32::NAN]).into_dyn());
        ws.insert("ends".to_string(), arr1(&[10.0_f32]).into_dyn());
        let m = make_context(&ws);
        let spec = slice_spec("starts", "ends");
        let err = m.convert_slice(&spec).unwrap_err();
        assert!(
            err.to_string().contains("NaN"),
            "Expected NaN rejection for start, got: {err}"
        );
    }

    #[test]
    fn slice_nan_end_rejected() {
        let mut ws = WeightStore::new();
        ws.insert("starts".to_string(), arr1(&[0.0_f32]).into_dyn());
        ws.insert("ends".to_string(), arr1(&[f32::NAN]).into_dyn());
        let m = make_context(&ws);
        let spec = slice_spec("starts", "ends");
        let err = m.convert_slice(&spec).unwrap_err();
        assert!(
            err.to_string().contains("NaN"),
            "Expected NaN rejection for end, got: {err}"
        );
    }

    #[test]
    fn slice_positive_inf_end_maps_to_max() {
        // +Infinity in ends is the "slice to end" sentinel → i64::MAX
        // Use ONNX axis=1 (trailing-relative -1 on rank 2) since axis=0
        // targets the batch dimension and is rejected.
        let mut ws = WeightStore::new();
        ws.insert("starts".to_string(), arr1(&[0.0_f32]).into_dyn());
        ws.insert("ends".to_string(), arr1(&[f32::INFINITY]).into_dyn());
        ws.insert("axes".to_string(), arr1(&[1.0_f32]).into_dyn());
        let shapes = HashMap::from([("data".to_string(), vec![1_i64, 10])]);
        let m = make_context_with_shapes(&ws, &shapes);
        let spec = slice_spec_with_axes("starts", "ends", Some("axes"));
        let layer = m.convert_slice(&spec).unwrap();
        match layer {
            Layer::Slice(sl) => {
                assert_eq!(sl.start, 0);
                assert_eq!(sl.end, i64::MAX as usize);
                assert_eq!(sl.axis, -1);
            }
            other => panic!("expected Layer::Slice, got {:?}", other),
        }
    }

    // ---- Slice: INT64_MAX "to end" sentinel through f32 storage
    // (#linearizenn-slice-sentinel) ----

    /// The real-world shape of the sentinel: ONNX stores `ends = [INT64_MAX]` in an
    /// INT64 Constant, the loader decodes it into the f32 weight store (warning about
    /// the lossy cast), and the converter sees the finite float `2^63`. It must still
    /// resolve to the "slice to end" sentinel, not a clamped finite index.
    #[test]
    fn slice_int64_max_end_via_f32_preserves_to_end_sentinel() {
        let mut ws = WeightStore::new();
        ws.insert("starts".to_string(), arr1(&[2.0_f32]).into_dyn());
        ws.insert("ends".to_string(), arr1(&[i64::MAX as f32]).into_dyn());
        ws.insert("axes".to_string(), arr1(&[1.0_f32]).into_dyn());
        let shapes = HashMap::from([("data".to_string(), vec![1_i64, 4])]);
        let m = make_context_with_shapes(&ws, &shapes);
        let spec = slice_spec_with_axes("starts", "ends", Some("axes"));
        match m.convert_slice(&spec).unwrap() {
            Layer::Slice(sl) => {
                assert_eq!(sl.start, 2);
                assert_eq!(
                    sl.end,
                    i64::MAX as usize,
                    "INT64_MAX end must survive the i64->f32 round-trip as the \
                     'to end' sentinel, not a clamped finite index"
                );
            }
            other => panic!("expected Layer::Slice, got {other:?}"),
        }
    }

    /// The behavioural consequence: an ONNX axis-0 `[0:INT64_MAX)` slice on a
    /// non-constant tensor with an UNKNOWN batch extent is a full-coverage no-op, and
    /// `resolve_batch_axis_slice` can only recognise that through the sentinel. With
    /// the sentinel clamped to a finite index this was rejected as an unsound partial
    /// batch slice.
    #[test]
    fn slice_axis0_int64_max_end_via_f32_is_identity_with_unknown_batch() {
        let mut ws = WeightStore::new();
        ws.insert("starts".to_string(), arr1(&[0.0_f32]).into_dyn());
        ws.insert("ends".to_string(), arr1(&[i64::MAX as f32]).into_dyn());
        let m = make_context(&ws);
        let spec = slice_spec("starts", "ends");
        match m.convert_slice(&spec).unwrap() {
            Layer::MulConstant(_) => {}
            other => panic!("expected identity Layer::MulConstant, got {other:?}"),
        }
    }

    /// A large but realizable finite end below the sentinel floor stays finite (it
    /// must not silently widen into "take the whole axis").
    #[test]
    fn slice_large_finite_end_is_not_promoted_to_sentinel() {
        let mut ws = WeightStore::new();
        ws.insert("starts".to_string(), arr1(&[0.0_f32]).into_dyn());
        ws.insert("ends".to_string(), arr1(&[1.0e9_f32]).into_dyn());
        ws.insert("axes".to_string(), arr1(&[1.0_f32]).into_dyn());
        let shapes = HashMap::from([("data".to_string(), vec![1_i64, 10])]);
        let m = make_context_with_shapes(&ws, &shapes);
        let spec = slice_spec_with_axes("starts", "ends", Some("axes"));
        match m.convert_slice(&spec).unwrap() {
            Layer::Slice(sl) => {
                assert_eq!(sl.end, 1_000_000_000);
                assert_ne!(sl.end, i64::MAX as usize);
            }
            other => panic!("expected Layer::Slice, got {other:?}"),
        }
    }

    #[test]
    fn slice_negative_inf_start_rejected() {
        let mut ws = WeightStore::new();
        ws.insert("starts".to_string(), arr1(&[f32::NEG_INFINITY]).into_dyn());
        ws.insert("ends".to_string(), arr1(&[10.0_f32]).into_dyn());
        ws.insert("axes".to_string(), arr1(&[1.0_f32]).into_dyn());
        let m = make_context(&ws);
        let spec = slice_spec_with_axes("starts", "ends", Some("axes"));
        let err = m.convert_slice(&spec).unwrap_err();
        assert!(
            err.to_string().contains("infinite start"),
            "Expected infinite start rejection, got: {err}"
        );
    }

    #[test]
    fn slice_negative_inf_end_rejected() {
        let mut ws = WeightStore::new();
        ws.insert("starts".to_string(), arr1(&[0.0_f32]).into_dyn());
        ws.insert("ends".to_string(), arr1(&[f32::NEG_INFINITY]).into_dyn());
        ws.insert("axes".to_string(), arr1(&[1.0_f32]).into_dyn());
        let m = make_context(&ws);
        let spec = slice_spec_with_axes("starts", "ends", Some("axes"));
        let err = m.convert_slice(&spec).unwrap_err();
        assert!(
            err.to_string().contains("negative infinite end"),
            "Expected negative infinite end rejection, got: {err}"
        );
    }

    #[test]
    fn slice_axis_zero_rejected_in_unbatched_mode() {
        // ONNX axis=0 targets the batch dimension, which is stripped in
        // unbatched mode. The converter must reject this.
        let mut ws = WeightStore::new();
        ws.insert("starts".to_string(), arr1(&[0.0_f32]).into_dyn());
        ws.insert("ends".to_string(), arr1(&[5.0_f32]).into_dyn());
        ws.insert("axes".to_string(), arr1(&[0.0_f32]).into_dyn());
        let m = make_context(&ws);
        let spec = slice_spec_with_axes("starts", "ends", Some("axes"));
        let err = m.convert_slice(&spec).unwrap_err();
        assert!(
            err.to_string().contains("batch dimension"),
            "Expected batch dimension rejection for axis=0, got: {err}"
        );
    }

    #[test]
    fn slice_default_axis_zero_rejected_in_unbatched_mode() {
        // When no axes input is provided, axis defaults to 0 (batch dimension).
        // This must be rejected in unbatched mode.
        let mut ws = WeightStore::new();
        ws.insert("starts".to_string(), arr1(&[0.0_f32]).into_dyn());
        ws.insert("ends".to_string(), arr1(&[5.0_f32]).into_dyn());
        let m = make_context(&ws);
        let spec = slice_spec("starts", "ends");
        let err = m.convert_slice(&spec).unwrap_err();
        assert!(
            err.to_string().contains("batch dimension"),
            "Expected batch dimension rejection for default axis=0, got: {err}"
        );
    }

    #[test]
    fn slice_axis_zero_full_coverage_int_max_becomes_identity() {
        // ONNX Slice on axis=0 with start=0, end=INT64_MAX ("to end") is a no-op on
        // the batch dimension. It must lower to an identity (MulConstant 1.0), sound
        // because selecting the whole axis changes nothing.
        let mut ws = WeightStore::new();
        ws.insert("starts".to_string(), arr1(&[0.0_f32]).into_dyn());
        // i64::MAX is far beyond f32 exact-int range; the parser maps +Inf sentinel.
        ws.insert("ends".to_string(), arr1(&[f32::INFINITY]).into_dyn());
        ws.insert("axes".to_string(), arr1(&[0.0_f32]).into_dyn());
        let m = make_context(&ws);
        let spec = slice_spec_with_axes("starts", "ends", Some("axes"));
        let layer = m.convert_slice(&spec).unwrap();
        match layer {
            Layer::MulConstant(_) => {}
            other => panic!("expected identity Layer::MulConstant, got {:?}", other),
        }
    }

    #[test]
    fn slice_axis_zero_full_coverage_via_known_batch_becomes_identity() {
        // start=0, end=1 with a known batch extent of 1 covers the whole axis-0
        // dimension → no-op → identity.
        let mut ws = WeightStore::new();
        ws.insert("starts".to_string(), arr1(&[0.0_f32]).into_dyn());
        ws.insert("ends".to_string(), arr1(&[1.0_f32]).into_dyn());
        ws.insert("axes".to_string(), arr1(&[0.0_f32]).into_dyn());
        let shapes = HashMap::from([("data".to_string(), vec![1_i64, 3, 4])]);
        let m = make_context_with_shapes(&ws, &shapes);
        let spec = slice_spec_with_axes("starts", "ends", Some("axes"));
        let layer = m.convert_slice(&spec).unwrap();
        match layer {
            Layer::MulConstant(_) => {}
            other => panic!("expected identity Layer::MulConstant, got {:?}", other),
        }
    }

    #[test]
    fn slice_axis_zero_on_constant_data_is_real_axis_slice() {
        // When the sliced tensor is a constant (never carried a batch dim), ONNX
        // axis=0 is a genuine data axis. A partial slice must lower to Slice(axis=0).
        let mut ws = WeightStore::new();
        // "data" is a constant weight → is_constant("data") == true.
        ws.insert(
            "data".to_string(),
            arr1(&[1.0_f32, 2.0, 3.0, 4.0]).into_dyn(),
        );
        ws.insert("starts".to_string(), arr1(&[1.0_f32]).into_dyn());
        ws.insert("ends".to_string(), arr1(&[3.0_f32]).into_dyn());
        ws.insert("axes".to_string(), arr1(&[0.0_f32]).into_dyn());
        let m = make_context(&ws);
        let spec = slice_spec_with_axes("starts", "ends", Some("axes"));
        let layer = m.convert_slice(&spec).unwrap();
        match layer {
            Layer::Slice(sl) => {
                assert_eq!(sl.axis, 0, "real data axis maps to internal axis 0");
                assert_eq!(sl.start, 1);
                assert_eq!(sl.end, 3);
            }
            other => panic!("expected Layer::Slice(axis=0), got {:?}", other),
        }
    }

    #[test]
    fn slice_axis_zero_partial_nonconstant_batch_rejected() {
        // A genuine partial slice of a non-constant batch dim with batch>1 cannot be
        // performed soundly in unbatched mode → keep the error (never silently drop).
        let mut ws = WeightStore::new();
        ws.insert("starts".to_string(), arr1(&[0.0_f32]).into_dyn());
        ws.insert("ends".to_string(), arr1(&[2.0_f32]).into_dyn());
        ws.insert("axes".to_string(), arr1(&[0.0_f32]).into_dyn());
        let shapes = HashMap::from([("data".to_string(), vec![4_i64, 3])]);
        let m = make_context_with_shapes(&ws, &shapes);
        let spec = slice_spec_with_axes("starts", "ends", Some("axes"));
        let err = m.convert_slice(&spec).unwrap_err();
        assert!(
            err.to_string().contains("batch dimension"),
            "Expected batch dimension rejection for partial batch slice, got: {err}"
        );
    }

    #[test]
    fn slice_axis_zero_one_dim_nonconstant_is_real_axis_slice_cctsdb() {
        // cctsdb_yolo_2023 pattern: a flattened detection head produces a 1-D
        // non-constant tensor `[12296]`, and ONNX `Slice[0:12288)` on axis 0 selects
        // the first 12288 elements. A 1-D ONNX shape has no batch axis to strip, so
        // axis 0 is a genuine data axis → real Slice(axis=0, 0:12288), not an error.
        let mut ws = WeightStore::new();
        ws.insert("starts".to_string(), arr1(&[0.0_f32]).into_dyn());
        ws.insert("ends".to_string(), arr1(&[12288.0_f32]).into_dyn());
        ws.insert("axes".to_string(), arr1(&[0.0_f32]).into_dyn());
        // "data" is NOT a constant (no weight/eval entry); only its shape is known.
        let shapes = HashMap::from([("data".to_string(), vec![12296_i64])]);
        let m = make_context_with_shapes(&ws, &shapes);
        let spec = slice_spec_with_axes("starts", "ends", Some("axes"));
        let layer = m.convert_slice(&spec).unwrap();
        match layer {
            Layer::Slice(sl) => {
                assert_eq!(sl.axis, 0, "1-D real data axis maps to internal axis 0");
                assert_eq!(sl.start, 0);
                assert_eq!(sl.end, 12288);
            }
            other => panic!("expected Layer::Slice(axis=0), got {:?}", other),
        }
    }

    #[test]
    fn slice_axis_zero_one_dim_nonconstant_ibp_is_sound() {
        // Convert a 1-D non-constant axis-0 Slice[1:4) and verify IBP produces the
        // correct sliced shape/values — exact bounds (no widening), proving soundness.
        use ndarray::{ArrayD, IxDyn};
        use ny_propagate::layers::BoundPropagation;
        use ny_tensor::BoundedTensor;

        let mut ws = WeightStore::new();
        ws.insert("starts".to_string(), arr1(&[1.0_f32]).into_dyn());
        ws.insert("ends".to_string(), arr1(&[4.0_f32]).into_dyn());
        ws.insert("axes".to_string(), arr1(&[0.0_f32]).into_dyn());
        let shapes = HashMap::from([("data".to_string(), vec![6_i64])]);
        let m = make_context_with_shapes(&ws, &shapes);
        let spec = slice_spec_with_axes("starts", "ends", Some("axes"));
        let layer = m.convert_slice(&spec).unwrap();
        let Layer::Slice(sl) = layer else {
            panic!("expected Layer::Slice");
        };

        let lower =
            ArrayD::from_shape_vec(IxDyn(&[6]), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let upper =
            ArrayD::from_shape_vec(IxDyn(&[6]), vec![0.5, 1.5, 2.5, 3.5, 4.5, 5.5]).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();
        let output = sl.propagate_ibp(&input).unwrap();

        assert_eq!(
            output.lower().shape(),
            &[3],
            "sliced shape is [1:4) = 3 elems"
        );
        let expected_lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap();
        let expected_upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.5, 2.5, 3.5]).unwrap();
        assert_eq!(output.lower(), &expected_lower);
        assert_eq!(output.upper(), &expected_upper);
    }

    #[test]
    fn slice_axis_zero_one_dim_size_one_full_coverage_still_identity() {
        // A genuine size-1 axis-0 slice [0:1) covering the whole extent stays a no-op
        // identity (Case 1), even for a 1-D tensor — the 1-D real-axis path must not
        // shadow the full-coverage no-op handling.
        let mut ws = WeightStore::new();
        ws.insert("starts".to_string(), arr1(&[0.0_f32]).into_dyn());
        ws.insert("ends".to_string(), arr1(&[1.0_f32]).into_dyn());
        ws.insert("axes".to_string(), arr1(&[0.0_f32]).into_dyn());
        let shapes = HashMap::from([("data".to_string(), vec![1_i64])]);
        let m = make_context_with_shapes(&ws, &shapes);
        let spec = slice_spec_with_axes("starts", "ends", Some("axes"));
        let layer = m.convert_slice(&spec).unwrap();
        match layer {
            Layer::MulConstant(_) => {}
            other => panic!("expected identity Layer::MulConstant, got {:?}", other),
        }
    }

    #[test]
    fn slice_attribute_axis_zero_full_coverage_becomes_identity() {
        // Attribute-based Slice (from Split lowering) on axis=0 covering full extent.
        let ws = WeightStore::new();
        let shapes = HashMap::from([("data".to_string(), vec![1_i64, 5])]);
        let m = make_context_with_shapes(&ws, &shapes);
        let spec = LayerSpec {
            name: "slice_attr".to_string(),
            layer_type: LayerType::Slice,
            inputs: vec!["data".to_string()],
            outputs: vec!["output".to_string()],
            weights: None,
            attributes: HashMap::from([
                ("axis".to_string(), AttributeValue::Int(0)),
                ("start".to_string(), AttributeValue::Int(0)),
                ("end".to_string(), AttributeValue::Int(1)),
            ]),
        };
        let layer = m.convert_slice(&spec).unwrap();
        match layer {
            Layer::MulConstant(_) => {}
            other => panic!("expected identity Layer::MulConstant, got {:?}", other),
        }
    }

    // ---- Reshape: NaN/Inf shape guard (#2848 self-audit) ----

    /// Build a `LayerSpec` for Reshape (input-based, shape from weights).
    fn reshape_spec(name: &str, shape_name: &str) -> LayerSpec {
        LayerSpec {
            name: name.to_string(),
            layer_type: LayerType::Reshape,
            inputs: vec!["data".to_string(), shape_name.to_string()],
            outputs: vec!["output".to_string()],
            weights: None,
            attributes: HashMap::new(),
        }
    }

    #[test]
    fn reshape_nan_shape_returns_none() {
        // NaN in shape tensor must be caught — `NaN as i64` silently produces 0
        let mut ws = WeightStore::new();
        ws.insert(
            "shape".to_string(),
            arr1(&[3.0_f32, f32::NAN, 4.0]).into_dyn(),
        );
        let m = make_context(&ws);
        let spec = reshape_spec("r_nan", "shape");
        assert!(
            m.try_convert_reshape(&spec).unwrap().is_none(),
            "Reshape with NaN shape should return None"
        );
    }

    #[test]
    fn reshape_inf_shape_returns_none() {
        // +Infinity in shape tensor must be caught — `Inf as i64` saturates to i64::MAX
        let mut ws = WeightStore::new();
        ws.insert(
            "shape".to_string(),
            arr1(&[3.0_f32, f32::INFINITY]).into_dyn(),
        );
        let m = make_context(&ws);
        let spec = reshape_spec("r_inf", "shape");
        assert!(
            m.try_convert_reshape(&spec).unwrap().is_none(),
            "Reshape with Inf shape should return None"
        );
    }

    #[test]
    fn reshape_finite_shape_succeeds() {
        let mut ws = WeightStore::new();
        ws.insert("shape".to_string(), arr1(&[3.0_f32, 4.0]).into_dyn());
        let m = make_context(&ws);
        let spec = reshape_spec("r_ok", "shape");
        let layer = m.try_convert_reshape(&spec).unwrap();
        assert!(layer.is_some(), "Reshape with finite shape should succeed");
    }

    #[test]
    fn reshape_invalid_negative_dim_returns_invalid_spec_2587() {
        let mut ws = WeightStore::new();
        ws.insert("shape".to_string(), arr1(&[-2.0_f32, 3.0]).into_dyn());
        let m = make_context(&ws);
        let spec = reshape_spec("r_invalid_neg", "shape");
        let err = m
            .try_convert_reshape(&spec)
            .expect_err("negative reshape dims below -1 should be rejected during conversion");
        assert!(
            matches!(err, NyError::InvalidSpec(_)),
            "expected InvalidSpec for invalid negative reshape dim, got {err:?}"
        );
        assert!(
            err.to_string().contains("-2"),
            "error should mention the invalid dimension, got: {err}"
        );
    }

    #[test]
    fn reshape_multiple_inferred_dims_returns_invalid_spec_2587() {
        let mut ws = WeightStore::new();
        ws.insert("shape".to_string(), arr1(&[-1.0_f32, -1.0, 3.0]).into_dyn());
        let m = make_context(&ws);
        let spec = reshape_spec("r_duplicate_infer", "shape");
        let err = m
            .try_convert_reshape(&spec)
            .expect_err("multiple -1 reshape dims should be rejected during conversion");
        assert!(
            matches!(err, NyError::InvalidSpec(_)),
            "expected InvalidSpec for duplicate inferred reshape dims, got {err:?}"
        );
        assert!(
            err.to_string().contains("multiple inferred dimensions"),
            "error should mention duplicate inferred dims, got: {err}"
        );
    }

    #[test]
    fn reshape_non_integer_shape_returns_invalid_spec_2587() {
        let mut ws = WeightStore::new();
        ws.insert("shape".to_string(), arr1(&[1.0_f32, 2.5]).into_dyn());
        let m = make_context(&ws);
        let spec = reshape_spec("r_non_integer", "shape");
        let err = m
            .try_convert_reshape(&spec)
            .expect_err("non-integer reshape dims should be rejected during conversion");
        assert!(
            matches!(err, NyError::InvalidSpec(_)),
            "expected InvalidSpec for non-integer reshape dim, got {err:?}"
        );
        assert!(
            err.to_string().contains("2.5"),
            "error should mention the non-integer dimension, got: {err}"
        );
    }

    #[test]
    fn reshape_evaluated_constant_shape_succeeds_2587() {
        let weights = WeightStore::new();
        let evaluated = HashMap::from([("shape".to_string(), arr1(&[1.0_f32, 6.0]).into_dyn())]);
        let m = make_context_with_evaluated(&weights, &evaluated);
        let spec = reshape_spec("r_eval_ok", "shape");
        let layer = m
            .try_convert_reshape(&spec)
            .expect("conversion should succeed")
            .expect("reshape should build from evaluated constants");
        assert_eq!(reshape_target(&layer), &[6]);
    }

    #[test]
    fn reshape_evaluated_constant_invalid_negative_dim_returns_invalid_spec_2587() {
        let weights = WeightStore::new();
        let evaluated = HashMap::from([("shape".to_string(), arr1(&[-2.0_f32, 3.0]).into_dyn())]);
        let m = make_context_with_evaluated(&weights, &evaluated);
        let spec = reshape_spec("r_eval_invalid_neg", "shape");
        let err = m
            .try_convert_reshape(&spec)
            .expect_err("evaluated constant reshape dims below -1 should be rejected");
        assert!(
            matches!(err, NyError::InvalidSpec(_)),
            "expected InvalidSpec for evaluated constant invalid negative dim, got {err:?}"
        );
        assert!(
            err.to_string().contains("-2"),
            "error should mention the invalid dimension, got: {err}"
        );
    }

    // ---- Reshape: batch dimension stripping (#3198) ----

    /// Extract the target_shape from a Layer::Reshape, panicking on mismatch.
    fn reshape_target(layer: &Layer) -> &[i64] {
        match layer {
            Layer::Reshape(r) => &r.target_shape,
            other => unreachable!("expected Layer::Reshape, got {:?}", other),
        }
    }

    #[test]
    fn reshape_batch_strip_two_dim() {
        // ONNX shape [3, 4] → unbatched [4] (first element is batch dim)
        let mut ws = WeightStore::new();
        ws.insert("shape".to_string(), arr1(&[3.0_f32, 4.0]).into_dyn());
        let m = make_context(&ws);
        let spec = reshape_spec("r_strip2", "shape");
        let layer = m
            .try_convert_reshape(&spec)
            .expect("conversion should succeed")
            .expect("reshape should build");
        assert_eq!(reshape_target(&layer), &[4]);
    }

    #[test]
    fn reshape_batch_strip_multi_dim() {
        // ONNX shape [-1, 6, 8] → unbatched [6, 8] (pensieve-like case)
        // First element (-1 = batch dim) is stripped
        let mut ws = WeightStore::new();
        ws.insert("shape".to_string(), arr1(&[-1.0_f32, 6.0, 8.0]).into_dyn());
        let m = make_context(&ws);
        let spec = reshape_spec("r_pensieve", "shape");
        let layer = m
            .try_convert_reshape(&spec)
            .expect("conversion should succeed")
            .expect("reshape should build");
        assert_eq!(reshape_target(&layer), &[6, 8]);
    }

    #[test]
    fn reshape_single_dim_not_stripped() {
        // ONNX shape [-1] (single element) → kept as [-1], no batch dim to strip
        let mut ws = WeightStore::new();
        ws.insert("shape".to_string(), arr1(&[-1.0_f32]).into_dyn());
        let m = make_context(&ws);
        let spec = reshape_spec("r_single", "shape");
        let layer = m
            .try_convert_reshape(&spec)
            .expect("conversion should succeed")
            .expect("reshape should build");
        assert_eq!(reshape_target(&layer), &[-1]);
    }

    #[test]
    fn reshape_batch_strip_with_dynamic_dim() {
        // ONNX shape [0, -1, 6] → unbatched [-1, 6]
        // 0 means "keep dim as-is" in ONNX, but first element is batch dim → stripped
        let mut ws = WeightStore::new();
        ws.insert("shape".to_string(), arr1(&[0.0_f32, -1.0, 6.0]).into_dyn());
        let m = make_context(&ws);
        let spec = reshape_spec("r_dynamic", "shape");
        let layer = m
            .try_convert_reshape(&spec)
            .expect("conversion should succeed")
            .expect("reshape should build");
        assert_eq!(reshape_target(&layer), &[-1, 6]);
    }

    #[test]
    fn reshape_prefers_exact_integer_shape_over_float_compatibility_view() {
        let mut ws = WeightStore::new();
        ws.insert(
            "shape".to_string(),
            arr1(&[1.0_f32, 0.0, 16.0, 128.0]).into_dyn(),
        );
        ws.insert_integers(
            "shape".to_string(),
            arr1(&[
                1_i64,
                ny_core::reshape_copy_axis_sentinel(1).expect("axis in range"),
                16,
                128,
            ])
            .into_dyn(),
        );
        let m = make_context(&ws);
        let spec = reshape_spec("r_exact_integer_shape", "shape");
        let layer = m
            .try_convert_reshape(&spec)
            .expect("conversion should succeed")
            .expect("reshape should build");
        assert_eq!(reshape_target(&layer), &[0, 16, 128]);
    }

    #[test]
    fn reshape_exact_integer_shape_allows_dynamic_copy_plus_literal_infer() {
        let mut ws = WeightStore::new();
        ws.insert("shape".to_string(), arr1(&[1.0_f32, 0.0, -1.0]).into_dyn());
        ws.insert_integers(
            "shape".to_string(),
            arr1(&[
                1_i64,
                ny_core::reshape_copy_axis_sentinel(1).expect("axis in range"),
                -1,
            ])
            .into_dyn(),
        );
        let m = make_context(&ws);
        let spec = reshape_spec("r_copy_and_infer", "shape");
        let layer = m
            .try_convert_reshape(&spec)
            .expect("conversion should succeed")
            .expect("reshape should build");
        assert_eq!(reshape_target(&layer), &[0, -1]);
    }

    #[test]
    fn reshape_exact_integer_shape_preserves_moved_dynamic_copy_axis() {
        let mut ws = WeightStore::new();
        ws.insert(
            "shape".to_string(),
            arr1(&[1.0_f32, -1.0, 0.0, 128.0]).into_dyn(),
        );
        ws.insert_integers(
            "shape".to_string(),
            arr1(&[
                1_i64,
                -1,
                ny_core::reshape_copy_axis_sentinel(3).expect("axis in range"),
                128,
            ])
            .into_dyn(),
        );
        let m = make_context(&ws);
        let spec = reshape_spec("r_moved_copy_axis", "shape");
        let layer = m
            .try_convert_reshape(&spec)
            .expect("conversion should succeed")
            .expect("reshape should build");
        assert_eq!(
            reshape_target(&layer),
            &[
                -1,
                ny_core::reshape_copy_axis_sentinel(2).expect("axis in range"),
                128
            ]
        );
    }

    #[test]
    fn reshape_batch_strip_preserves_negative_one() {
        // ONNX shape [1, -1] → unbatched [-1] (common pattern: batch=1, flatten rest)
        let mut ws = WeightStore::new();
        ws.insert("shape".to_string(), arr1(&[1.0_f32, -1.0]).into_dyn());
        let m = make_context(&ws);
        let spec = reshape_spec("r_batch1", "shape");
        let layer = m
            .try_convert_reshape(&spec)
            .expect("conversion should succeed")
            .expect("reshape should build");
        assert_eq!(reshape_target(&layer), &[-1]);
    }

    // ---- #2360 regression: precision-loss shape rejection ----

    #[test]
    fn reshape_precision_loss_shape_rejected_2360() {
        // f32 at 2^24 range has precision of 2, so 16_777_218.0 is the next
        // representable value above the 2^24 exact-integer limit. Any value that
        // made it through the i64→f32 round-trip at this magnitude could have
        // lost precision from the original i64.
        let big_val = 16_777_218.0_f32;
        let mut ws = WeightStore::new();
        ws.insert("shape".to_string(), arr1(&[1.0_f32, big_val]).into_dyn());
        let m = make_context(&ws);
        let spec = reshape_spec("r_precision_loss", "shape");
        let result = m.try_convert_reshape(&spec);
        assert!(
            result.is_err(),
            "shape exceeding f32 exact-integer range must be rejected"
        );
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("exact-integer range"),
            "error should mention exact-integer range: {msg}"
        );
    }

    // ---- Flatten: converter tests (#3198) ----

    /// Build a `LayerSpec` for Flatten with given axis attribute.
    fn flatten_spec(name: &str, axis: i64) -> LayerSpec {
        LayerSpec {
            name: name.to_string(),
            layer_type: LayerType::Flatten,
            inputs: vec!["data".to_string()],
            outputs: vec!["output".to_string()],
            weights: None,
            attributes: HashMap::from([("axis".to_string(), AttributeValue::Int(axis))]),
        }
    }

    #[test]
    fn flatten_axis_1_produces_flatten_layer_0() {
        // ONNX axis=1 (default) → adjusted axis=0 → FlattenLayer(0) for 2D [1, total].
        // FlattenLayer(0) preserves 2D shape compatibility with CROWN backward
        // propagation, which produces 2D bounds from downstream Linear layers (#3300).
        let ws = WeightStore::new();
        let m = make_context(&ws);
        let spec = flatten_spec("f_default", 1);
        let layer = m.convert_flatten(&spec).unwrap();
        match layer {
            Layer::Flatten(f) => assert_eq!(f.axis, 0),
            other => panic!("expected Layer::Flatten(0), got {:?}", other),
        }
    }

    #[test]
    fn flatten_axis_2_produces_flatten_layer() {
        // ONNX axis=2 → adjusted axis=1 → FlattenLayer(1), not Reshape
        let ws = WeightStore::new();
        let m = make_context(&ws);
        let spec = flatten_spec("f_axis2", 2);
        let layer = m.convert_flatten(&spec).unwrap();
        match layer {
            Layer::Flatten(f) => assert_eq!(f.axis, 1),
            other => panic!("expected Layer::Flatten, got {:?}", other),
        }
    }

    #[test]
    fn flatten_axis_0_rejected() {
        // ONNX axis=0 targets batch dimension → rejected in unbatched mode
        // (input shape unknown → ambiguous → conservative reject).
        let ws = WeightStore::new();
        let m = make_context(&ws);
        let spec = flatten_spec("f_batch", 0);
        let err = m.convert_flatten(&spec).unwrap_err();
        assert!(
            err.to_string().contains("batch dimension"),
            "Expected batch dimension rejection, got: {err}"
        );
    }

    #[test]
    fn flatten_axis_0_rank1_genuine_data_axis_loads_and_shapes_correctly() {
        // A rank-1 (no batch axis) input `[6]` with ONNX axis=0 loads WITHOUT the
        // unbatched-mode error: `data_had_batch_axis == Some(false)` → FlattenLayer(0).
        let ws = WeightStore::new();
        let shapes = HashMap::from([("data".to_string(), vec![6_i64])]);
        let m = make_context_with_shapes(&ws, &shapes);
        let spec = flatten_spec("f_genuine", 0);
        let layer = m
            .convert_flatten(&spec)
            .expect("axis=0 Flatten on a genuine rank-1 data axis must load without error");
        let Layer::Flatten(f) = &layer else {
            panic!("expected Layer::Flatten, got {:?}", layer);
        };
        assert_eq!(f.axis, 0);

        // FlattenLayer(0) on rank-1 [6] → [1, 6].
        use ny_propagate::layers::BoundPropagation;
        let lower =
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[6]), vec![0.0_f32; 6]).unwrap();
        let upper =
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[6]), vec![1.0_f32; 6]).unwrap();
        let input = ny_tensor::BoundedTensor::new(lower, upper).unwrap();
        let out = f.propagate_ibp(&input).expect("flatten ibp");
        assert_eq!(out.shape(), &[1, 6]);
    }

    #[test]
    fn slice_finite_values_convert_correctly() {
        // ONNX axis=1 on a recorded rank-2 tensor -> trailing-relative -1.
        let mut ws = WeightStore::new();
        ws.insert("starts".to_string(), arr1(&[2.0_f32]).into_dyn());
        ws.insert("ends".to_string(), arr1(&[8.0_f32]).into_dyn());
        ws.insert("axes".to_string(), arr1(&[1.0_f32]).into_dyn());
        let shapes = HashMap::from([("data".to_string(), vec![1_i64, 10])]);
        let m = make_context_with_shapes(&ws, &shapes);
        let spec = slice_spec_with_axes("starts", "ends", Some("axes"));
        let layer = m.convert_slice(&spec).unwrap();
        match layer {
            Layer::Slice(sl) => {
                assert_eq!(sl.start, 2);
                assert_eq!(sl.end, 8);
                // ONNX axis=1 on rank-2 -> trailing-relative -1 (correct under
                // both runtime layouts).
                assert_eq!(sl.axis, -1);
            }
            other => panic!("expected Layer::Slice, got {:?}", other),
        }
    }

    #[test]
    fn slice_uses_evaluated_constant_end_3500() {
        let mut ws = WeightStore::new();
        ws.insert("starts".to_string(), arr1(&[0.0_f32]).into_dyn());
        ws.insert("axes".to_string(), arr1(&[1.0_f32]).into_dyn());
        let evaluated = HashMap::from([
            ("ends".to_string(), arr1(&[8.0_f32]).into_dyn()),
            // recorded rank source for the data tensor (rank 2)
            ("data_rank_probe".to_string(), arr1(&[0.0_f32]).into_dyn()),
        ]);
        let shapes = HashMap::from([("data".to_string(), vec![1_i64, 10])]);
        static CONSTANTS: std::sync::LazyLock<HashSet<String>> =
            std::sync::LazyLock::new(HashSet::new);
        let m = ConvertContext::with_evaluated_constants(&ws, &shapes, &CONSTANTS, &evaluated);
        let spec = slice_spec_with_axes("starts", "ends", Some("axes"));
        let layer = m.convert_slice(&spec).unwrap();
        match layer {
            Layer::Slice(sl) => {
                assert_eq!(sl.start, 0);
                assert_eq!(sl.end, 8);
                assert_eq!(sl.axis, -1);
            }
            other => panic!("expected Layer::Slice, got {:?}", other),
        }
    }

    #[test]
    fn slice_plural_attributes_convert_correctly_opset9() {
        let ws = WeightStore::new();
        let shapes = HashMap::from([("data".to_string(), vec![1_i64, 10])]);
        let m = make_context_with_shapes(&ws, &shapes);
        let spec = slice_attr_spec_plural(&[1], &[2], &[8]);
        let layer = m.convert_slice(&spec).unwrap();
        match layer {
            Layer::Slice(sl) => {
                assert_eq!(sl.start, 2);
                assert_eq!(sl.end, 8);
                assert_eq!(sl.axis, -1);
            }
            other => panic!("expected Layer::Slice, got {:?}", other),
        }
    }

    #[test]
    fn slice_plural_attributes_multi_axis_rejected_opset9() {
        let ws = WeightStore::new();
        let m = make_context(&ws);
        let spec = slice_attr_spec_plural(&[1, 2], &[0, 1], &[4, 5]);
        let err = m.convert_slice(&spec).unwrap_err();
        assert!(
            err.to_string().contains("multi-axis opset 9 Slice"),
            "Expected multi-axis opset 9 rejection, got: {err}"
        );
    }

    // -----------------------------------------------------------------------
    // Variable-start Slice of an arithmetic progression -> exact affine Linear
    // (#cctsdb B3a)
    // -----------------------------------------------------------------------

    fn progression_slice_spec() -> LayerSpec {
        LayerSpec {
            name: "slice_range".to_string(),
            layer_type: LayerType::Slice,
            inputs: vec![
                "range".to_string(),
                "x".to_string(),
                "x_plus_w".to_string(),
                "axes".to_string(),
                "steps".to_string(),
            ],
            outputs: vec!["window".to_string()],
            weights: None,
            attributes: HashMap::new(),
        }
    }

    fn progression_weights() -> WeightStore {
        let mut weights = WeightStore::new();
        // range = [0, 1, ..., 7]
        weights.insert(
            "range".to_string(),
            arr1(&(0..8).map(|v| v as f32).collect::<Vec<_>>()).into_dyn(),
        );
        weights.insert("axes".to_string(), arr1(&[0.0_f32]).into_dyn());
        weights.insert("steps".to_string(), arr1(&[1.0_f32]).into_dyn());
        weights
    }

    fn progression_shapes() -> HashMap<String, Vec<i64>> {
        HashMap::from([("x".to_string(), vec![1]), ("window".to_string(), vec![3])])
    }

    /// Slice(range, [x], [x+w]) lowers to a Linear whose output is exactly
    /// [x, x+1, x+2]; the IBP hull over an index interval encloses every
    /// concrete instantiation of the slice.
    #[ntest::timeout(10000)]
    #[test]
    fn variable_start_slice_of_range_lowers_to_exact_affine_linear() {
        use ny_propagate::layers::BoundPropagation;
        use ny_tensor::BoundedTensor;

        let weights = progression_weights();
        let shapes = progression_shapes();
        let ctx = make_context_with_shapes(&weights, &shapes);
        let layer = ctx
            .convert_slice(&progression_slice_spec())
            .expect("variable-start slice of a progression must lower");
        let Layer::Linear(linear) = layer else {
            panic!("expected affine Linear lowering, got {layer:?}");
        };

        // Point starts x = 2 -> [2, 3, 4] up to the Linear IBP's 1-ULP
        // directed rounding (sound outward widening).
        let point =
            BoundedTensor::new(arr1(&[2.0_f32]).into_dyn(), arr1(&[2.0_f32]).into_dyn()).unwrap();
        let out = linear.propagate_ibp(&point).unwrap();
        for (k, expected) in [2.0_f32, 3.0, 4.0].into_iter().enumerate() {
            let lo = out.lower().as_slice().unwrap()[k];
            let up = out.upper().as_slice().unwrap()[k];
            assert!(
                lo <= expected && expected <= up && (up - lo) < 1e-4,
                "k={k}: expected tight enclosure of {expected}, got [{lo}, {up}]"
            );
        }

        // Interval starts x in [1, 4]: bounds must enclose data[x..x+3] for
        // every integer x in the range (the enclosure property).
        let interval =
            BoundedTensor::new(arr1(&[1.0_f32]).into_dyn(), arr1(&[4.0_f32]).into_dyn()).unwrap();
        let bounds = linear.propagate_ibp(&interval).unwrap();
        let data: Vec<f32> = (0..8).map(|v| v as f32).collect();
        for x in 1..=4_usize {
            for k in 0..3_usize {
                let concrete = if x + k < data.len() {
                    data[x + k]
                } else {
                    // Unclamped sentinel (out-of-range) position: the lowering
                    // extrapolates the progression.
                    (x + k) as f32
                };
                let lo = bounds.lower().as_slice().unwrap()[k];
                let up = bounds.upper().as_slice().unwrap()[k];
                assert!(
                    lo <= concrete && concrete <= up,
                    "x={x} k={k}: {concrete} not in [{lo}, {up}]"
                );
            }
        }
    }

    /// Non-progression data must keep the conservative reject.
    #[ntest::timeout(10000)]
    #[test]
    fn variable_start_slice_of_non_progression_rejected() {
        let mut weights = progression_weights();
        weights.insert(
            "range".to_string(),
            arr1(&[0.0_f32, 1.0, 4.0, 9.0]).into_dyn(),
        );
        let shapes = progression_shapes();
        let ctx = make_context_with_shapes(&weights, &shapes);
        let err = ctx
            .convert_slice(&progression_slice_spec())
            .expect_err("non-progression data must stay rejected");
        assert!(matches!(err, NyError::UnsupportedOp(_)));
    }

    /// Unknown output extent (no recorded static shape) keeps the reject.
    #[ntest::timeout(10000)]
    #[test]
    fn variable_start_slice_without_static_extent_rejected() {
        let weights = progression_weights();
        let shapes = HashMap::from([("x".to_string(), vec![1])]);
        let ctx = make_context_with_shapes(&weights, &shapes);
        let err = ctx
            .convert_slice(&progression_slice_spec())
            .expect_err("missing static extent must stay rejected");
        assert!(matches!(err, NyError::UnsupportedOp(_)));
    }

    /// Non-unit steps keep the reject (extent formula changes).
    #[ntest::timeout(10000)]
    #[test]
    fn variable_start_slice_non_unit_steps_rejected() {
        let mut weights = progression_weights();
        weights.insert("steps".to_string(), arr1(&[2.0_f32]).into_dyn());
        let shapes = progression_shapes();
        let ctx = make_context_with_shapes(&weights, &shapes);
        let err = ctx
            .convert_slice(&progression_slice_spec())
            .expect_err("non-unit steps must stay rejected");
        assert!(matches!(err, NyError::UnsupportedOp(_)));
    }
}
