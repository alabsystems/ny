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
pub(super) fn op_type_to_layer_type(op_type: &str, name: &str) -> Result<(LayerType, bool)> {
    match op_type {
        // Basic layers
        "Gemm" => Ok((LayerType::Linear, true)),
        // MatMul: could be a Linear layer (if one input is a weight) or a binary matmul
        // We mark it as MatMul; the converter will check if it should be treated as Linear
        "MatMul" => Ok((LayerType::MatMul, true)),
        // Attention ops (if present in ONNX graph)
        "Attention" | "MultiHeadAttention" => Ok((LayerType::MultiHeadAttention, true)),
        "Conv" => Ok((LayerType::Conv2d, true)),
        "ConvTranspose" => Ok((LayerType::ConvTranspose2d, true)),
        // Activations
        "Relu" => Ok((LayerType::ReLU, true)),
        "LeakyRelu" => Ok((LayerType::LeakyRelu, true)),
        "Gelu" => Ok((LayerType::GELU, true)),
        "Softmax" => Ok((LayerType::Softmax, true)),
        "Tanh" => Ok((LayerType::Tanh, true)),
        "Sigmoid" => Ok((LayerType::Sigmoid, true)),
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
        "Snake" => Ok((LayerType::Snake, true)),
        "Floor" => Ok((LayerType::Floor, true)),
        "Ceil" => Ok((LayerType::Ceil, true)),
        "Round" => Ok((LayerType::Round, true)),
        "Sign" => Ok((LayerType::Sign, true)),
        "Reciprocal" => Ok((LayerType::Reciprocal, true)),
        "Sin" => Ok((LayerType::Sin, true)),
        "Cos" => Ok((LayerType::Cos, true)),
        "Tan" => Ok((LayerType::Tan, true)),
        "Atan" => Ok((LayerType::Arctan, true)),
        // Positional encoding
        "RoPE" | "RotaryPositionEmbedding" => Ok((LayerType::RoPE, true)),
        // Normalization (fused ops)
        "LayerNormalization" => Ok((LayerType::LayerNorm, true)),
        // SimplifiedLayerNormalization is the ONNX name for RMSNorm (opset 21+).
        // RMSNormalization is a common custom op name used by some frameworks.
        "SimplifiedLayerNormalization" | "RMSNormalization" => Ok((LayerType::RMSNorm, true)),
        // InstanceNormalization: per-channel normalization (ONNX opset 1+).
        // Used in style transfer and audio models (avoice K2/K3/K4).
        "InstanceNormalization" => Ok((LayerType::InstanceNorm, true)),
        // GroupNormalization: per-group normalization (ONNX opset 21+).
        // Used in Demucs DConv sub-layers (dilated Conv1d + GroupNorm + GELU).
        "GroupNormalization" => Ok((LayerType::GroupNorm, true)),
        // AdaIN: Adaptive Instance Normalization (custom op in style-based models).
        // Used in avoice K3/K4. Not a standard ONNX op.
        "AdaIN" | "AdaptiveInstanceNorm" | "AdaptiveInstanceNormalization" => {
            Ok((LayerType::AdaIN, true))
        }
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
        // Logical ops: still traced through until boolean propagation exists.
        "And" | "Or" | "Not" => {
            debug!("{} op '{}' skipped (comparison/mask op)", op_type, name);
            Ok((LayerType::Unknown, false))
        }
        // Reduction ops that produce scalars/shapes
        "ReduceProd" => {
            debug!("{} op '{}' skipped (reduction op)", op_type, name);
            Ok((LayerType::Unknown, false))
        }
        // ReduceMax/ReduceMin: supported with fixed_max_index CROWN assumption
        "ReduceMax" => Ok((LayerType::ReduceMax, true)),
        "ReduceMin" => Ok((LayerType::ReduceMin, true)),
        // Selection ops: ArgMax/ArgMin/ArgSort/TopK. Piecewise-constant index
        // outputs; sound integer-index intervals (exact when argmax/min provably
        // unique over the input box). Detection heads (cctsdb_yolo) use ArgMax.
        "ArgMax" => Ok((LayerType::Argmax, true)),
        "ArgMin" => Ok((LayerType::ArgMin, true)),
        "ArgSort" => Ok((LayerType::ArgSort, true)),
        "TopK" => Ok((LayerType::Topk, true)),
        // Cast with a full-precision FLOAT target (f32/f64) preserves values
        // (we work in f32). Integer-target Casts never reach this arm:
        // convert_node_to_layer lowers them to LayerType::Trunc BEFORE
        // consulting this table, because float->int casts truncate and an
        // identity drop is unsound for fractional values (#cctsdb B1).
        // f16/bf16-target Casts never reach this arm either: they round with
        // up to 2^-11 relative error, so convert_node_to_layer routes them to
        // a refused LayerType::Cast (fail closed — permissive graph build
        // degrades to a sound OpaqueSkip [-inf, +inf]) instead of identity.
        "Cast" => {
            debug!(
                "Cast op '{}' skipped (f32/f64 target, exact identity)",
                name
            );
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
    fn test_scatter_nd_mapping() {
        let (ty, include) = op_type_to_layer_type("ScatterND", "scatter_0").unwrap();
        assert_eq!(ty, LayerType::ScatterND);
        assert!(include);
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
