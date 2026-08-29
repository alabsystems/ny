// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ny_core::{LayerType, NyError, Result};
use tracing::debug;

/// Map an ONNX op type string to an internal `LayerType` and whether to include
/// the op in the verification graph.
///
/// Returns `Ok((LayerType, true))` for supported data-path ops,
/// `Ok((LayerType::Unknown, false))` for known shape/constant/utility ops that
/// are safe to trace through, and `Err(UnsupportedOp)` for truly unknown ops
/// that should not be silently dropped. (#2931)
///
/// # NOT IN THIS TABLE: the vendor-domain family
///
/// This map is only ever consulted for a node in the STANDARD domain
/// (`""` / `"ai.onnx"`). `convert_node_to_layer` runs the custom-op registry
/// first and then refuses any non-standard domain outright ("Custom op missing
/// registration"), so a vendor-domain node either belongs to its registered
/// `CustomOpHandler` — which builds its own `LayerSpec` — or never gets here.
///
/// It therefore used to LIE about coverage. Arms for `ArgSort`, `Snake`,
/// `RoPE` / `RotaryPositionEmbedding`, `AdaIN` /
/// `AdaptiveInstanceNorm` / `AdaptiveInstanceNormalization`, and
/// `SimplifiedLayerNormalization` were unreachable in both directions:
///
/// * in the standard domain, `parse::schema_preflight::validate_standard_schemas`
///   refuses every one of those op types BY NAME ("is not a registered
///   main-domain operator; use an explicitly imported and registered vendor
///   domain") before conversion ever runs — none of them is a main-domain ONNX
///   operator;
/// * in a vendor domain, the registry / missing-registration gate above
///   short-circuits first.
///
/// The arms are deleted rather than wired, because wiring them would mean
/// admitting a standard-domain op type that ONNX does not define — exactly the
/// "same-named lookalike" substitution `is_standard_domain` exists to prevent.
/// `LayerType::ArgSort` / `Snake` / `RoPE` / `AdaIN` remain reachable through a
/// registered `CustomOpHandler` and through the trace bridge, which is where
/// those operators legitimately come from. `RMSNormalization` KEPT its arm:
/// it is a real main-domain operator (opset 23) and is not in the preflight's
/// refusal list, so it is genuinely reachable; only its legacy experimental
/// spelling `SimplifiedLayerNormalization` was dead.
///
/// `op_map_refusals_are_not_shadowed_by_the_schema_preflight` below pins that
/// analysis so the table cannot silently regrow a dead arm.
pub(super) fn op_type_to_layer_type(op_type: &str, name: &str) -> Result<(LayerType, bool)> {
    match op_type {
        // Basic layers
        "Gemm" => Ok((LayerType::Linear, true)),
        // MatMul: could be a Linear layer (if one input is a weight) or a binary matmul
        // We mark it as MatMul; the converter will check if it should be treated as Linear
        "MatMul" => Ok((LayerType::MatMul, true)),
        // Standard Attention-23/24 is deliberately not mapped here: its
        // packed-head, mask, cache, optional-output, and pre-MatMul scaling
        // semantics are not represented by ny's simplified ternary layer.
        // Custom-domain attention handlers run before this map.
        "Conv" => Ok((LayerType::Conv2d, true)),
        "ConvTranspose" => Ok((LayerType::ConvTranspose2d, true)),
        // Activations
        "Relu" => Ok((LayerType::ReLU, true)),
        "LeakyRelu" => Ok((LayerType::LeakyRelu, true)),
        "Gelu" => Ok((LayerType::GELU, true)),
        "Softmax" => Ok((LayerType::Softmax, true)),
        "Tanh" => Ok((LayerType::Tanh, true)),
        "Sigmoid" => Ok((LayerType::Sigmoid, true)),
        "Erf" => Ok((LayerType::Erf, true)),
        "Softplus" => Ok((LayerType::Softplus, true)),
        "Clip" => Ok((LayerType::Clip, true)),
        "Elu" => Ok((LayerType::Elu, true)),
        "Selu" => Ok((LayerType::Selu, true)),
        "PRelu" => Ok((LayerType::PRelu, true)),
        "HardSigmoid" => Ok((LayerType::HardSigmoid, true)),
        "HardSwish" => Ok((LayerType::HardSwish, true)),
        "Swish" => Ok((LayerType::SiLU, true)),
        "SiLU" => Ok((LayerType::SiLU, true)),
        "Exp" => Ok((LayerType::Exp, true)),
        "Log" => Ok((LayerType::Log, true)),
        "Celu" => Ok((LayerType::Celu, true)),
        "Mish" => Ok((LayerType::Mish, true)),
        "LogSoftmax" => Ok((LayerType::LogSoftmax, true)),
        "ThresholdedRelu" => Ok((LayerType::ThresholdedRelu, true)),
        "Shrink" => Ok((LayerType::Shrink, true)),
        "Softsign" => Ok((LayerType::Softsign, true)),
        "Floor" => Ok((LayerType::Floor, true)),
        "Ceil" => Ok((LayerType::Ceil, true)),
        "Round" => Ok((LayerType::Round, true)),
        "Sign" => Ok((LayerType::Sign, true)),
        "Reciprocal" => Ok((LayerType::Reciprocal, true)),
        "Sin" => Ok((LayerType::Sin, true)),
        "Cos" => Ok((LayerType::Cos, true)),
        "Tan" => Ok((LayerType::Tan, true)),
        "Atan" => Ok((LayerType::Arctan, true)),
        // Normalization (fused ops)
        "LayerNormalization" => Ok((LayerType::LayerNorm, true)),
        // RMSNormalization is the standard main-domain op since opset 23; its
        // exact supported schema is checked by `validate_direct_normalization_schema`
        // in convert.rs before this table is consulted.
        //
        // The legacy experimental spelling `SimplifiedLayerNormalization` is
        // deliberately NOT here — see the "NOT IN THIS TABLE" note below.
        "RMSNormalization" => Ok((LayerType::RMSNorm, true)),
        // InstanceNormalization: per-channel normalization (ONNX opset 1+).
        // Used in style transfer and audio models (avoice K2/K3/K4).
        "InstanceNormalization" => Ok((LayerType::InstanceNorm, true)),
        // GroupNormalization: per-group normalization (ONNX opset 21+).
        // Used in Demucs DConv sub-layers (dilated Conv1d + GroupNorm + GELU).
        "GroupNormalization" => Ok((LayerType::GroupNorm, true)),
        "BatchNormalization" => Ok((LayerType::BatchNorm, true)),
        // Pooling
        "AveragePool" => Ok((LayerType::AveragePool, true)),
        "GlobalAveragePool" => Ok((LayerType::AveragePool, true)),
        "MaxPool" => Ok((LayerType::MaxPool, true)),
        // Structural ops
        "Add" => Ok((LayerType::Add, true)),
        // Element-wise arithmetic ops
        "Neg" => Ok((LayerType::Neg, true)),
        "Abs" => Ok((LayerType::Abs, true)),
        "Sqrt" => Ok((LayerType::Sqrt, true)),
        "Div" => Ok((LayerType::Div, true)),
        "Sub" => Ok((LayerType::Sub, true)),
        "Pow" => Ok((LayerType::Pow, true)),
        // Conditional ops
        "Where" => Ok((LayerType::Where, true)),
        // Index/selection ops
        "NonZero" => Ok((LayerType::NonZero, true)),
        "Gather" => Ok((LayerType::Gather, true)),
        "ScatterND" => Ok((LayerType::ScatterND, true)),
        // Reduction ops
        "ReduceMean" => Ok((LayerType::ReduceMean, true)),
        "ReduceSum" => Ok((LayerType::ReduceSum, true)),
        "CumSum" => Ok((LayerType::CumSum, true)),
        // Transpose: include in layer list (has static perm attribute)
        "Transpose" => Ok((LayerType::Transpose, true)),
        // Reshape: include in layer list if shape can be determined from weights
        // The conversion will fail gracefully if shape is dynamic
        "Reshape" => Ok((LayerType::Reshape, true)),
        // Mul: include as a layer - can be constant scaling (attention 1/sqrt(d_k)) or binary
        // The convert_layer function handles both cases via try_convert_mul
        "Mul" => {
            debug!("Mul op '{}' found", name);
            Ok((LayerType::Mul, true))
        }
        // Min/Max: variadic element-wise ops in ONNX, we support binary form
        // Common uses: clamp operations, residual connections
        "Min" => {
            debug!("Min op '{}' found", name);
            Ok((LayerType::Min, true))
        }
        "Max" => {
            debug!("Max op '{}' found", name);
            Ok((LayerType::Max, true))
        }
        // Shape ops that modify tensor dimensions (now supported)
        "Resize" | "Upsample" => Ok((LayerType::Resize, true)),
        "Squeeze" => {
            debug!("Squeeze op '{}' found", name);
            Ok((LayerType::Squeeze, true))
        }
        "Unsqueeze" => {
            debug!("Unsqueeze op '{}' found", name);
            Ok((LayerType::Unsqueeze, true))
        }
        // Slice: data-path op for extracting contiguous ranges along axes.
        // Previously misclassified as a shape/constant op (#2931), but Slice
        // appears in data flow paths (ViT patch extraction, sequence slicing).
        // LayerType::Slice is fully supported by the converter.
        "Slice" => {
            debug!("Slice op '{}' found", name);
            Ok((LayerType::Slice, true))
        }
        // Shape is retained so graph building can constant-fold shape-arithmetic
        // chains using tensor_shapes populated from ONNX Runtime shape inference.
        "Shape" => {
            debug!("Shape op '{}' found", name);
            Ok((LayerType::Shape, true))
        }
        // Constant ops produce constants, not activations. They're safe to trace
        // through and are extracted/folded before graph conversion.
        "Constant" | "ConstantOfShape" | "Range" => {
            debug!("{} op '{}' skipped (shape/constant op)", op_type, name);
            Ok((LayerType::Unknown, false))
        }
        "Expand" => {
            debug!("Expand op '{}' found", name);
            Ok((LayerType::Expand, true))
        }
        "Tile" => {
            debug!("Tile op '{}' found", name);
            Ok((LayerType::Tile, true))
        }
        // Concat: can be either shape-computing or data concat
        // Data concat (e.g., CLS token + patches in ViT) should be included as a layer
        // Shape-computing concat (building Reshape target shape) will be filtered later
        "Concat" => {
            debug!("Concat op '{}' found", name);
            Ok((LayerType::Concat, true))
        }
        // Flatten: collapse dimensions according to axis parameter
        "Flatten" => {
            debug!("Flatten op '{}' found", name);
            Ok((LayerType::Flatten, true))
        }
        // Split: produces multiple outputs (slices along axis)
        // Handled specially in graph construction to create one Slice layer per output
        "Split" => {
            debug!(
                "Split op '{}' found (will expand to multiple Slice layers)",
                name
            );
            // We use LayerType::Slice to signal special Split handling
            // The graph builder will detect this has multiple outputs and expand it
            Ok((LayerType::Slice, true))
        }
        // Explicit padding: preserve as a real layer so graph conversion keeps
        // the spatial extent seen by downstream convolutions.
        "Pad" => {
            debug!("Pad op '{}' found", name);
            Ok((LayerType::Pad, true))
        }
        // Comparison ops: used for masking and routed through Compare/CompareTensor conversion.
        "Equal" | "Less" | "Greater" | "LessOrEqual" | "GreaterOrEqual" => {
            debug!("{} op '{}' found", op_type, name);
            Ok((LayerType::Compare, true))
        }
        // ReduceMax/ReduceMin: supported with fixed_max_index CROWN assumption
        "ReduceMax" => Ok((LayerType::ReduceMax, true)),
        "ReduceMin" => Ok((LayerType::ReduceMin, true)),
        // Selection ops: ArgMax/ArgMin/TopK. Piecewise-constant index outputs;
        // sound integer-index intervals (exact when argmax/min provably unique
        // over the input box). Detection heads (cctsdb_yolo) use ArgMax.
        // `ArgSort` is NOT a main-domain ONNX operator — see the note below.
        "ArgMax" => Ok((LayerType::Argmax, true)),
        "ArgMin" => Ok((LayerType::ArgMin, true)),
        "TopK" => Ok((LayerType::Topk, true)),
        // Only Casts whose drop is an EXACT identity reach this arm: a FLOAT32
        // target, or a BOOL target on a provably {0,1}-valued operand.
        // convert_node_to_layer handles every other target BEFORE consulting
        // this table — integer targets lower to LayerType::Trunc (float->int
        // casts truncate, and an identity drop is unsound for fractional
        // values, #cctsdb B1), and unmodeled targets emit LayerType::Cast,
        // which fails closed in ny-build.
        "Cast" => {
            debug!("Cast op '{}' skipped (exact identity target)", name);
            Ok((LayerType::Unknown, false))
        }
        "DequantizeLinear" => Ok((LayerType::DequantizeLinear, true)),
        "QuantizeLinear" => Ok((LayerType::QuantizeLinear, true)),
        // Identity is a no-op pass-through
        "Identity" => {
            debug!("Identity op '{}' skipped (pass-through)", name);
            Ok((LayerType::Unknown, false))
        }
        // Dropout is identity during inference (ratio attribute ignored)
        "Dropout" => {
            debug!(
                "Dropout op '{}' skipped (inference mode, pass-through)",
                name
            );
            Ok((LayerType::Unknown, false))
        }
        // Truly unknown ops: return an error instead of silently dropping.
        // This prevents verification from proceeding on a structurally incomplete
        // graph. If a new op needs to be skipped, add an explicit match arm above. (#2931)
        _ => Err(NyError::UnsupportedOp(format!(
            "ONNX op '{}' (node '{}') is not supported and cannot be silently dropped",
            op_type, name
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ny_core::LayerType;

    #[test]
    fn test_silu_swish_mapping() {
        // Both "SiLU" and "Swish" ONNX op names must map to LayerType::SiLU.
        // Regression test requested by Prover (Re: #1148).
        let (ty, include) = op_type_to_layer_type("SiLU", "silu_0").unwrap();
        assert_eq!(ty, LayerType::SiLU);
        assert!(include);

        let (ty, include) = op_type_to_layer_type("Swish", "swish_0").unwrap();
        assert_eq!(ty, LayerType::SiLU);
        assert!(include);
    }

    #[test]
    fn test_unknown_op_returns_error() {
        // Truly unknown ops must return UnsupportedOp, not silently drop (#2931).
        let result = op_type_to_layer_type("FakeOp", "fake_0");
        assert!(result.is_err());
    }

    #[test]
    fn non_identity_logical_and_reduction_ops_fail_closed() {
        for op_type in ["And", "Or", "Not", "ReduceProd"] {
            assert!(
                matches!(
                    op_type_to_layer_type(op_type, "live"),
                    Err(NyError::UnsupportedOp(_))
                ),
                "{op_type} must not be dropped as a pass-through"
            );
        }
    }

    #[test]
    fn test_scatter_nd_mapping() {
        let (ty, include) = op_type_to_layer_type("ScatterND", "scatter_0").unwrap();
        assert_eq!(ty, LayerType::ScatterND);
        assert!(include);
    }

    /// The vendor-domain family must NOT be in this table.
    ///
    /// Every one of these op types is refused by name in
    /// `parse::schema_preflight::validate_standard_schemas` when it appears in
    /// the standard domain, and is short-circuited by the custom-op registry /
    /// missing-registration gate when it appears in a vendor domain. An arm
    /// here is therefore unreachable and makes the map overstate its coverage.
    /// Pinned as an error so a re-added arm fails loudly instead of quietly
    /// lying.
    #[test]
    fn vendor_domain_op_types_are_not_claimed_by_the_standard_map() {
        for op_type in [
            "ArgSort",
            "Snake",
            "RoPE",
            "RotaryPositionEmbedding",
            "AdaIN",
            "AdaptiveInstanceNorm",
            "AdaptiveInstanceNormalization",
            "SimplifiedLayerNormalization",
        ] {
            assert!(
                matches!(
                    op_type_to_layer_type(op_type, "vendor"),
                    Err(NyError::UnsupportedOp(_))
                ),
                "'{op_type}' is not a main-domain ONNX operator; the standard op map must not \
                 claim it (it reaches ny through a registered CustomOpHandler or the trace bridge)"
            );
        }
    }

    /// `RMSNormalization` is the counterpart the deletion above must NOT touch:
    /// a genuine main-domain operator (opset 23) that is absent from the
    /// preflight's refusal list and therefore genuinely reachable here.
    #[test]
    fn standard_rms_normalization_keeps_its_reachable_arm() {
        let (ty, include) = op_type_to_layer_type("RMSNormalization", "rms_0").unwrap();
        assert_eq!(ty, LayerType::RMSNorm);
        assert!(include);
    }

    /// `Trilu`, `ScatterElements` and the deprecated `Scatter` are handled by
    /// their own canonicalizers BEFORE this table is consulted, so reaching
    /// this arm at all means the canonicalizer declined a node it owns. Keep
    /// the fail-closed answer.
    #[test]
    fn canonicalized_ops_still_fail_closed_in_the_generic_table() {
        for op_type in ["Trilu", "ScatterElements", "Scatter"] {
            assert!(
                matches!(
                    op_type_to_layer_type(op_type, "live"),
                    Err(NyError::UnsupportedOp(_))
                ),
                "{op_type} is canonicalized upstream; the generic table must not admit it"
            );
        }
    }

    #[test]
    fn test_resize_mapping() {
        let (ty, include) = op_type_to_layer_type("Resize", "resize_0").unwrap();
        assert_eq!(ty, LayerType::Resize);
        assert!(include);

        let (ty, include) = op_type_to_layer_type("Upsample", "upsample_0").unwrap();
        assert_eq!(ty, LayerType::Resize);
        assert!(include);
    }

    #[test]
    fn test_compare_mapping() {
        for op_type in ["Equal", "Less", "Greater", "LessOrEqual", "GreaterOrEqual"] {
            let (ty, include) = op_type_to_layer_type(op_type, "cmp_0").unwrap();
            assert_eq!(ty, LayerType::Compare, "{op_type} should map to Compare");
            assert!(include, "{op_type} should be included in the graph");
        }
    }
}
