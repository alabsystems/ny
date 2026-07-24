// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! ONNX layer conversion, split by op family into focused modules.

mod arithmetic;
mod concat;
mod const_eval;
mod conv;
mod elementwise;
#[cfg(test)]
mod elementwise_invalid_attr_tests;
#[cfg(test)]
mod elementwise_tests;
mod gather;
mod linear;
mod matmul;
mod normalization;
mod pad;
mod pooling;
mod quantization;
mod reductions;
mod reshape;
mod resize;
mod rope;
mod scatter_nd;
mod selection;

use ny_core::{LayerType, NyError, Result};
use ny_propagate::layers::{AttentionMask, NonZeroLayer, SelfAttentionLayer, WhereLayer};
use ny_propagate::Layer;
use std::collections::{HashMap, HashSet};
use tracing::debug;

use crate::{AttributeValue, LayerSpec, WeightStore};
use ndarray::ArrayD;

/// Largest integer magnitude that `f32` can represent exactly.
const F32_INT_EXACT_LIMIT: i64 = 1 << 24;

pub(super) fn i64_to_f32_checked(value: i64, context: &str) -> Result<f32> {
    if value.unsigned_abs() > F32_INT_EXACT_LIMIT as u64 {
        return Err(NyError::ModelLoad(format!(
            "i64->f32 precision loss: {value} exceeds f32 exact-integer range +/-{F32_INT_EXACT_LIMIT} (context: {context})"
        )));
    }
    Ok(value as f32)
}

/// Minimal state needed for ONNX layer conversion and graph construction.
///
/// This decouples conversion logic from the full `OnnxModel` and is the
/// first boundary needed to move construction code out of `ny-onnx` (#1752).
#[derive(Clone, Copy)]
pub struct ConvertContext<'a> {
    /// Weight storage (shared borrow from the model).
    pub weights: &'a WeightStore,
    /// Known tensor shapes keyed by tensor name.
    pub tensor_shapes: &'a HashMap<String, Vec<i64>>,
    /// Set of tensor names known to be constant (not activation-dependent).
    pub constant_tensors: &'a HashSet<String>,
    /// Pre-evaluated constant tensors (values available for embedding).
    /// These are constant chains evaluated at graph construction time,
    /// used to create unary constant layer variants when one input of
    /// a binary op is a constant tensor (not a weight).
    pub evaluated_constants: &'a HashMap<String, ArrayD<f32>>,
    /// Whether the model is globally UNBATCHED: every graph input has rank
    /// <= 1, so no tensor in the graph ever carried a batch axis and ONNX
    /// axes / reshape targets map to internal semantics VERBATIM (no `-1`
    /// axis shift, no leading-dim strip). cctsdb_yolo_2023 (input `[12296]`)
    /// is the only VNN-COMP category in this class; every other scored
    /// category has rank >= 2 inputs and keeps the legacy stripped-batch
    /// convention (#cctsdb B5 / risk 3-4). Computed via
    /// [`model_is_unbatched`]; defaults to `false` in the constructors.
    pub model_unbatched: bool,
}

/// Legacy behavior for ONNX axis 0 when the recorded rank of the data tensor
/// is UNKNOWN (ny-synthesized internal subgraphs; see
/// [`ConvertContext::remap_axis_trailing`]). Preserves each op family's
/// historic semantics exactly.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LegacyBatchAxisPolicy {
    /// Keep axis 0 with a debug note (historic reductions / Concat behavior).
    KeepZeroWarn,
    /// Reject axis 0 as targeting the nonexistent batch dim (#2848 family).
    RejectZero,
}

/// Whether a model is globally unbatched: it has at least one real graph
/// input and every graph input has rank <= 1.
///
/// For such models no batch axis was ever stripped during conversion, so
/// ONNX shapes, axes, and reshape targets describe the internal tensors
/// verbatim. Mirrors the load-time `internal_shape_from_onnx_shape` rule in
/// `graph/helpers.rs` (#cctsdb). An input-less model (all-constant test
/// fixtures) keeps the legacy convention: `false`, never vacuously true.
pub fn model_is_unbatched(model_inputs: &[crate::TensorSpec]) -> bool {
    !model_inputs.is_empty() && model_inputs.iter().all(|spec| spec.shape.len() <= 1)
}

impl ConvertContext<'_> {
    /// Create a new conversion context from borrowed model fields.
    pub fn new<'a>(
        weights: &'a WeightStore,
        tensor_shapes: &'a HashMap<String, Vec<i64>>,
        constant_tensors: &'a HashSet<String>,
    ) -> ConvertContext<'a> {
        static EMPTY_EVAL: std::sync::OnceLock<HashMap<String, ArrayD<f32>>> =
            std::sync::OnceLock::new();
        ConvertContext {
            weights,
            tensor_shapes,
            constant_tensors,
            evaluated_constants: EMPTY_EVAL.get_or_init(HashMap::new),
            model_unbatched: false,
        }
    }

    /// Builder-style setter for [`ConvertContext::model_unbatched`].
    pub fn with_model_unbatched(mut self, model_unbatched: bool) -> Self {
        self.model_unbatched = model_unbatched;
        self
    }

    /// Create a conversion context with pre-evaluated constant values.
    pub fn with_evaluated_constants<'a>(
        weights: &'a WeightStore,
        tensor_shapes: &'a HashMap<String, Vec<i64>>,
        constant_tensors: &'a HashSet<String>,
        evaluated_constants: &'a HashMap<String, ArrayD<f32>>,
    ) -> ConvertContext<'a> {
        ConvertContext {
            weights,
            tensor_shapes,
            constant_tensors,
            evaluated_constants,
            model_unbatched: false,
        }
    }

    /// Check if a tensor name is a known constant (weight, constant tensor, or evaluated).
    pub fn is_constant(&self, name: &str) -> bool {
        self.weights.get(name).is_some()
            || self.constant_tensors.contains(name)
            || self.evaluated_constants.contains_key(name)
    }

    /// Whether the named tensor's leading ONNX dimension was a (stripped) batch axis.
    ///
    /// `ny-propagate` works on unbatched tensors. During unbatching,
    /// `unbatched_target_shape_from_onnx_shape` keeps a length-≤1 ONNX shape verbatim
    /// (no batch axis to strip) and strips the leading dim only when rank > 1. So:
    /// - `Some(true)`  — recorded ONNX shape has rank > 1: leading dim was a batch axis
    ///   that has been stripped, so ONNX axis 0 does not exist in unbatched mode.
    /// - `Some(false)` — recorded ONNX shape has rank ≤ 1: kept verbatim, so ONNX axis 0
    ///   is a genuine data axis that maps directly to internal axis 0.
    /// - `None`        — shape unknown: ambiguous; callers keep the conservative reject.
    ///
    /// This is purely a load-time shape/axis interpretation; it touches no bound math.
    /// Mirrors the `data_had_batch_axis` computation in `resolve_gather_axis` (gather.rs)
    /// and the 1-D activation Case 2b in `reshape.rs`.
    pub fn data_had_batch_axis(&self, name: &str) -> Option<bool> {
        if self.model_unbatched {
            // Unbatched model: NO tensor ever carried a batch axis, whatever
            // its rank (#cctsdb B5).
            return Some(false);
        }
        self.tensor_shapes.get(name).map(|shape| shape.len() > 1)
    }

    /// Value of a constant tensor from any source (weights or evaluated constants).
    pub fn constant_value(&self, name: &str) -> Option<ArrayD<f32>> {
        self.weights
            .get(name)
            .cloned()
            .or_else(|| self.evaluated_constants.get(name).cloned())
    }

    /// Recorded ONNX rank of a tensor.
    ///
    /// Source priority matters: `tensor_shapes` (the load-time ORT shape
    /// inference / value_info record) is the authoritative ONNX shape and is
    /// consulted FIRST — `evaluated_constants` hold tensors materialized by
    /// ny's own constant evaluation in the INTERNAL (possibly batch-stripped)
    /// convention, so their rank can understate the ONNX rank. Initializer
    /// (`weights`) shapes come straight from the model file and are genuine
    /// ONNX shapes. Symbolic dims still carry rank information, so
    /// `tensor_shapes` entries with dynamic dims are fine here.
    /// `None` = rank unknown.
    pub(crate) fn recorded_onnx_rank(&self, name: &str) -> Option<usize> {
        self.tensor_shapes
            .get(name)
            .map(|shape| shape.len())
            .or_else(|| self.weights.get(name).map(|tensor| tensor.ndim()))
            .or_else(|| {
                self.evaluated_constants
                    .get(name)
                    .map(|tensor| tensor.ndim())
            })
    }

    /// Remap an ONNX axis of the activation tensor `data_name` to the
    /// TRAILING-RELATIVE (negative) internal axis encoding, which is correct
    /// under BOTH internal runtime layouts of a batched model.
    ///
    /// Background (#pensieve ReduceSum no-op miscompile): ny's internal
    /// (unbatched) convention is NOT rank-uniform. For an ONNX tensor of rank
    /// `r`, the runtime tensor is either
    ///   - the ONNX tensor with its leading batch dim stripped (rank `r-1`,
    ///     e.g. Split/Slice outputs), or
    ///   - the ONNX tensor verbatim (rank `r`, leading size-1 retained, e.g.
    ///     Flatten and rank-2 Gemm outputs, weights/constants).
    ///
    /// The legacy blanket `axis >= 1 → axis - 1` guess is only correct for the
    /// first layout; on the second it selects the WRONG dimension — on
    /// pensieve `ReduceSum(axes=[1])` on a runtime `[1, n]` tensor became a
    /// size-1-axis no-op, so the graph bounded a different function than the
    /// ONNX semantics (w = p/p = 1).
    ///
    /// Both layouts share the ONNX tensor's TRAILING dims, so expressing the
    /// axis from the end (`onnx_axis - r`, negative) selects the same semantic
    /// dimension in either layout; layers resolve negative axes against the
    /// actual runtime rank at propagation time.
    ///
    /// Fail-closed where information exists: with a KNOWN recorded rank, an
    /// out-of-range axis or the (possibly stripped) batch axis 0 of a rank>1
    /// tensor — where no single encoding is correct for both layouts — REFUSES
    /// conversion with an error rather than guessing.
    ///
    /// UNKNOWN recorded rank keeps the LEGACY adjustment (`axis - 1`, warn):
    /// real ONNX models get recorded shapes from load-time shape inference, so
    /// this branch is reached (in practice) only by ny-SYNTHESIZED internal
    /// subgraphs (LSTM unrolling, ReduceL2 lowering, test graphs) that were
    /// authored directly against the legacy stripped-batch convention — for
    /// them the legacy adjustment is correct BY CONSTRUCTION, not a guess.
    /// `zero_policy` selects the per-op legacy behavior for ONNX axis 0 in
    /// that branch (historic per-op semantics are preserved exactly).
    ///
    /// Unbatched models (#cctsdb B5) keep ONNX axes verbatim. Negative ONNX
    /// axes are already trailing-relative and pass through (range-checked when
    /// the rank is known).
    pub(crate) fn remap_axis_trailing(
        &self,
        op: &str,
        node_name: &str,
        data_name: &str,
        onnx_axis: i64,
        zero_policy: LegacyBatchAxisPolicy,
    ) -> Result<i64> {
        if self.model_unbatched {
            // Unbatched model: no tensor ever carried a batch axis; ONNX axes
            // describe the internal tensors verbatim.
            return Ok(onnx_axis);
        }
        let rank = self.recorded_onnx_rank(data_name);
        if onnx_axis < 0 {
            if let Some(rank) = rank {
                if onnx_axis < -(rank as i64) {
                    return Err(NyError::ModelLoad(format!(
                        "{op} '{node_name}': ONNX axis {onnx_axis} out of range for input \
                         '{data_name}' of recorded rank {rank}"
                    )));
                }
            }
            return Ok(onnx_axis);
        }
        let Some(rank) = rank else {
            // Legacy-compatibility branch (see doc comment): synthesized
            // internal subgraphs carry no recorded ONNX shapes and were
            // written against the stripped-batch convention.
            if onnx_axis == 0 {
                return match zero_policy {
                    LegacyBatchAxisPolicy::KeepZeroWarn => {
                        debug!(
                            "{op} '{node_name}': ONNX axis 0 (batch dim) kept as 0 in unbatched \
                             mode (unknown recorded rank; legacy behavior)"
                        );
                        Ok(0)
                    }
                    LegacyBatchAxisPolicy::RejectZero => Err(NyError::ModelLoad(format!(
                        "{op} '{node_name}': axis=0 targets batch dimension which does not exist \
                         in unbatched mode"
                    ))),
                };
            }
            debug!(
                "{op} '{node_name}': input '{data_name}' has no recorded ONNX shape; keeping \
                 legacy batch-squeeze adjustment {} -> {} (synthesized-subgraph compatibility)",
                onnx_axis,
                onnx_axis - 1
            );
            return Ok(onnx_axis - 1);
        };
        let rank_i = rank as i64;
        if onnx_axis >= rank_i {
            return Err(NyError::ModelLoad(format!(
                "{op} '{node_name}': ONNX axis {onnx_axis} out of range for input '{data_name}' \
                 of recorded rank {rank}"
            )));
        }
        if onnx_axis == 0 && rank > 1 {
            // ONNX axis 0 of a rank>1 tensor is the (possibly stripped) batch
            // axis: on a stripped runtime tensor the op is a batch no-op, on a
            // retained one it targets the leading size-1 dim. No single axes
            // encoding expresses both; refuse (consistent with Gather #2848).
            return Err(NyError::UnsupportedOp(format!(
                "{op} '{node_name}': ONNX axis 0 targets the batch dimension of rank-{rank} \
                 input '{data_name}', which is ambiguous in unbatched mode; refusing conversion"
            )));
        }
        Ok(onnx_axis - rank_i)
    }

    /// Deprecated compatibility alias for [`constant_value`](Self::constant_value).
    #[deprecated(note = "use constant_value")]
    pub fn get_constant_value(&self, name: &str) -> Option<ArrayD<f32>> {
        self.constant_value(name)
    }

    /// Convert a single [`LayerSpec`] into a bound-propagation [`Layer`].
    ///
    /// Dispatches on `spec.layer_type` to the appropriate converter. Returns
    /// [`NyError::UnsupportedOp`] for unrecognized layer types.
    pub fn convert_layer(&self, spec: &LayerSpec) -> Result<Layer> {
        if let Some(layer) = self.convert_elementwise(spec)? {
            return Ok(layer);
        }
        match &spec.layer_type {
            LayerType::Linear => self.convert_linear(spec),
            LayerType::Conv1d => self.convert_conv1d(spec),
            LayerType::Conv2d => {
                // ONNX uses one "Conv" op for every spatial rank, while the
                // loader maps it to this compatibility variant. Dispatch the
                // ranks that NY can propagate exactly and classify every other
                // rank as unsupported. In particular, a 5-D kernel is Conv3d
                // (`[out_c, in_c/group, kD, kH, kW]`), not a malformed Conv2d
                // kernel. Returning UnsupportedOp lets graph conversion insert
                // its conservative unbounded fallback instead of hard-failing
                // with a misleading ShapeMismatch.
                if let Some(kernel) = spec
                    .inputs
                    .get(1)
                    .and_then(|name| self.constant_value(name))
                {
                    return match kernel.ndim() {
                        3 => self.convert_conv1d(spec),
                        4 => self.convert_conv2d(spec),
                        rank => {
                            let spatial_rank = rank.saturating_sub(2);
                            Err(NyError::UnsupportedOp(format!(
                                "ONNX Conv layer '{}' has rank-{rank} kernel {:?} \
                                 (Conv{spatial_rank}d, spatial rank {spatial_rank}); \
                                 NY supports only Conv1d and Conv2d propagation",
                                spec.name,
                                kernel.shape(),
                            )))
                        }
                    };
                }
                self.convert_conv2d(spec)
            }
            LayerType::ConvTranspose1d => self.convert_conv_transpose1d(spec),
            LayerType::ConvTranspose2d => {
                if let Some(kernel) = spec
                    .inputs
                    .get(1)
                    .and_then(|name| self.constant_value(name))
                {
                    if kernel.ndim() == 3 {
                        return self.convert_conv_transpose1d(spec);
                    }
                }
                self.convert_conv_transpose2d(spec)
            }
            LayerType::LogSumExp => self.convert_logsumexp(spec),
            LayerType::Argmax => self.convert_argmax(spec),
            LayerType::ArgMin => self.convert_argmin(spec),
            LayerType::ArgSort => self.convert_argsort(spec),
            LayerType::Topk => self.convert_topk(spec),
            LayerType::LayerNorm => self.convert_layer_norm(spec),
            LayerType::RMSNorm => self.convert_rms_norm(spec),
            LayerType::InstanceNorm => self.convert_instance_norm(spec),
            LayerType::GroupNorm => self.convert_group_norm(spec),
            LayerType::AdaIN => self.convert_adain(spec),
            LayerType::BatchNorm => self.convert_batch_norm(spec),
            LayerType::DequantizeLinear => self.convert_dequantize_linear(spec),
            LayerType::QuantizeLinear => self.convert_quantize_linear(spec),
            LayerType::AveragePool => self.convert_average_pool(spec),
            LayerType::MaxPool => self.convert_max_pool(spec),
            LayerType::MatMul => self.convert_matmul(spec),
            LayerType::Add => self.convert_add(spec),
            LayerType::Concat => self.convert_concat(spec),
            // Shape transformation ops: try to convert, error if dynamic shape
            LayerType::Reshape => match self.try_convert_reshape(spec)? {
                Some(layer) => Ok(layer),
                None => {
                    debug!("Reshape {} has dynamic shape", spec.name);
                    Err(NyError::UnsupportedOp(format!(
                        "Reshape {} has dynamic shape (shape tensor not constant)",
                        spec.name
                    )))
                }
            },
            LayerType::Transpose => self.convert_transpose(spec),
            LayerType::Mul => {
                if let Some(scale_attr) = spec.attributes.get("scale") {
                    match scale_attr {
                        AttributeValue::Float(_) | AttributeValue::Int(_) => {}
                        _ => {
                            return Err(NyError::ModelLoad(format!(
                                "Mul {} has invalid scale attribute type",
                                spec.name
                            )));
                        }
                    }
                }
                if spec.inputs.is_empty() {
                    let input_summary = spec.inputs.join(", ");
                    return Err(NyError::ModelLoad(format!(
                        "Mul {} has no inputs (got {}, inputs=[{}])",
                        spec.name,
                        spec.inputs.len(),
                        input_summary
                    )));
                }
                if spec.attributes.contains_key("scale") {
                    if spec.inputs.len() != 1 {
                        let input_summary = spec.inputs.join(", ");
                        return Err(NyError::ModelLoad(format!(
                            "Mul {} with scale attribute requires 1 input (got {}, inputs=[{}])",
                            spec.name,
                            spec.inputs.len(),
                            input_summary
                        )));
                    }
                } else if spec.inputs.len() != 2 {
                    let input_summary = spec.inputs.join(", ");
                    return Err(NyError::ModelLoad(format!(
                        "Mul {} requires exactly 2 inputs (got {}, inputs=[{}])",
                        spec.name,
                        spec.inputs.len(),
                        input_summary
                    )));
                }
                if spec.inputs.len() >= 2 {
                    let input_a = &spec.inputs[0];
                    let input_b = &spec.inputs[1];
                    self.check_mul_broadcast(spec, input_a, input_b)?;
                }
                match self.try_convert_mul(spec)? {
                    Some(layer) => Ok(layer),
                    None => {
                        let input_details = spec
                            .inputs
                            .iter()
                            .map(|name| {
                                if let Some(weight) = self.weights.get(name) {
                                    format!("{}:const{:?}", name, weight.shape())
                                } else {
                                    format!("{}:activation", name)
                                }
                            })
                            .collect::<Vec<_>>()
                            .join(", ");
                        debug!(
                            "Skipping Mul {} (unsupported inputs for conversion, details=[{}])",
                            spec.name, input_details
                        );
                        let input_summary = spec.inputs.join(", ");
                        Err(NyError::ModelLoad(format!(
                            "Mul {} conversion failed: unsupported inputs (got {}, inputs=[{}], details=[{}])",
                            spec.name,
                            spec.inputs.len(),
                            input_summary,
                            input_details
                        )))
                    }
                }
            }
            LayerType::Div => self.convert_div(spec),
            LayerType::Sub => self.convert_sub(spec),
            LayerType::Pow => self.convert_pow(spec),
            LayerType::Min => self.convert_min(spec),
            LayerType::Max => self.convert_max(spec),
            LayerType::ReduceMean => self.convert_reduce_mean(spec),
            LayerType::ReduceSum => self.convert_reduce_sum(spec),
            LayerType::ReduceMax => self.convert_reduce_max(spec),
            LayerType::ReduceMin => self.convert_reduce_min(spec),
            LayerType::CumSum => self.convert_cumsum(spec),
            LayerType::Flatten => self.convert_flatten(spec),
            LayerType::Squeeze => self.convert_squeeze(spec),
            LayerType::Unsqueeze => self.convert_unsqueeze(spec),
            LayerType::Resize => self.convert_resize(spec),
            LayerType::Tile => self.convert_tile(spec),
            LayerType::Expand => self.convert_expand(spec),
            LayerType::Slice => self.convert_slice(spec),
            LayerType::Gather => self.convert_gather(spec),
            LayerType::ScatterND => self.convert_scatter_nd(spec),
            LayerType::Pad => self.convert_pad(spec),
            LayerType::Shape => {
                let input = spec.inputs.first().map(String::as_str).unwrap_or("<missing>");
                Err(NyError::UnsupportedOp(format!(
                    "Shape op '{}' should be constant-folded during model loading; input '{}' shape was not static",
                    spec.name, input
                )))
            }
            LayerType::Cast => Err(NyError::UnsupportedOp(
                "Cast op should be constant-folded during model loading; if present in graph conversion, input was not constant".to_string()
            )),
            LayerType::Where => Ok(Layer::Where(WhereLayer::new())),
            LayerType::NonZero => Ok(Layer::NonZero(NonZeroLayer)),
            LayerType::RoPE => self.convert_rope(spec),
            LayerType::MultiHeadAttention => {
                let mut scale = None;
                if let Some(scale_attr) = spec.attributes.get("scale") {
                    scale = Some(match scale_attr {
                        AttributeValue::Float(v) => *v,
                        AttributeValue::Int(v) => i64_to_f32_checked(
                            *v,
                            &format!("MultiHeadAttention {} scale attribute", spec.name),
                        )?,
                        _ => {
                            return Err(NyError::ModelLoad(format!(
                                "MultiHeadAttention {} has invalid scale attribute type",
                                spec.name
                            )));
                        }
                    });
                }

                let causal = matches!(spec.attributes.get("causal"), Some(AttributeValue::Int(1)))
                    || matches!(
                        spec.attributes.get("is_causal"),
                        Some(AttributeValue::Int(1))
                    );

                let mask = if causal {
                    AttentionMask::Causal
                } else {
                    AttentionMask::Standard
                };

                Ok(Layer::SelfAttention(SelfAttentionLayer::new(mask, scale)))
            }
            other => Err(NyError::UnsupportedOp(format!(
                "Layer type {:?} not yet supported",
                other
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::WeightStore;
    use ndarray::{ArrayD, IxDyn};
    use std::collections::{HashMap, HashSet};

    #[ntest::timeout(10000)]
    #[test]
    fn convert_layer_handles_elementwise_ops() {
        let tensor_shapes = HashMap::from([
            ("input".to_string(), vec![1]),
            ("output".to_string(), vec![1]),
        ]);
        let weights = WeightStore::new();
        let constant_tensors = HashSet::new();
        let ctx = ConvertContext::new(&weights, &tensor_shapes, &constant_tensors);

        let spec = LayerSpec {
            name: "relu".to_string(),
            layer_type: LayerType::ReLU,
            inputs: vec!["input".to_string()],
            outputs: vec!["output".to_string()],
            weights: None,
            attributes: HashMap::new(),
        };

        let layer = ctx.convert_layer(&spec).expect("convert layer succeeds");
        assert!(matches!(layer, Layer::ReLU(_)));
    }

    #[test]
    fn convert_layer_dispatches_conv2d_with_evaluated_3d_kernel_to_conv1d_3500() {
        let tensor_shapes = HashMap::new();
        let weights = WeightStore::new();
        let constant_tensors = HashSet::new();
        let evaluated = HashMap::from([(
            "kernel".to_string(),
            ArrayD::from_shape_vec(IxDyn(&[2, 3, 3]), vec![0.0; 2 * 3 * 3]).unwrap(),
        )]);
        let ctx = ConvertContext::with_evaluated_constants(
            &weights,
            &tensor_shapes,
            &constant_tensors,
            &evaluated,
        );

        let spec = LayerSpec {
            name: "conv".to_string(),
            layer_type: LayerType::Conv2d,
            inputs: vec!["input".to_string(), "kernel".to_string()],
            outputs: vec!["output".to_string()],
            weights: None,
            attributes: HashMap::new(),
        };

        let layer = ctx
            .convert_layer(&spec)
            .expect("evaluated 3D kernels should dispatch as Conv1d");
        assert!(matches!(layer, Layer::Conv1d(_)));
    }

    #[test]
    fn convert_layer_classifies_smart_turn_5d_kernel_as_unsupported_conv3d() {
        let tensor_shapes = HashMap::from([("pixel_values".to_string(), vec![1, 3, 32, 112, 112])]);
        let weights = WeightStore::new();
        let constant_tensors = HashSet::new();
        // Exact shape of Smart Turn's first video convolution.
        let evaluated = HashMap::from([(
            "onnx::Conv_822_DequantizeLinear_Output".to_string(),
            ArrayD::from_shape_vec(IxDyn(&[64, 3, 3, 7, 7]), vec![0.0; 64 * 3 * 3 * 7 * 7])
                .unwrap(),
        )]);
        let ctx = ConvertContext::with_evaluated_constants(
            &weights,
            &tensor_shapes,
            &constant_tensors,
            &evaluated,
        );

        let spec = LayerSpec {
            name: "/video_backbone/stem/stem.0/Conv".to_string(),
            layer_type: LayerType::Conv2d,
            inputs: vec![
                "pixel_values".to_string(),
                "onnx::Conv_822_DequantizeLinear_Output".to_string(),
            ],
            outputs: vec!["stem_output".to_string()],
            weights: None,
            attributes: HashMap::new(),
        };

        let error = ctx
            .convert_layer(&spec)
            .expect_err("5-D ONNX Conv kernel must fail closed as unsupported Conv3d");
        assert!(
            matches!(
                &error,
                NyError::UnsupportedOp(message)
                    if message.contains("/video_backbone/stem/stem.0/Conv")
                        && message.contains("rank-5 kernel [64, 3, 3, 7, 7]")
                        && message.contains("Conv3d")
                        && message.contains("spatial rank 3")
                        && message.contains("Conv1d and Conv2d")
            ),
            "unexpected error: {error:?}"
        );
    }

    #[test]
    fn convert_layer_rejects_attention_scale_precision_loss_4149() {
        let tensor_shapes = HashMap::new();
        let weights = WeightStore::new();
        let constant_tensors = HashSet::new();
        let ctx = ConvertContext::new(&weights, &tensor_shapes, &constant_tensors);

        let spec = LayerSpec {
            name: "attn".to_string(),
            layer_type: LayerType::MultiHeadAttention,
            inputs: vec!["q".to_string(), "k".to_string(), "v".to_string()],
            outputs: vec!["out".to_string()],
            weights: None,
            attributes: HashMap::from([(
                "scale".to_string(),
                AttributeValue::Int(F32_INT_EXACT_LIMIT + 1),
            )]),
        };

        let error = ctx
            .convert_layer(&spec)
            .expect_err("precision-losing attention scale should be rejected");
        assert!(
            matches!(
                &error,
                NyError::ModelLoad(msg)
                    if msg.contains("precision loss")
                        && msg.contains("MultiHeadAttention attn scale attribute")
            ),
            "unexpected error: {error:?}"
        );
    }
}
