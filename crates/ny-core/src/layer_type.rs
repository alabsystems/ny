// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

/// Layer types supported by ny.
///
/// This enum is `#[non_exhaustive]` to allow adding new layer types
/// without breaking semver. External code should include a wildcard arm
/// in exhaustive pattern matches.
/// # Naming Conventions
/// ## Capitalization
/// Activation names follow their original paper capitalization where unambiguous:
/// - `ReLU`, `SiLU`, `GELU`: All-caps suffixes from literature
/// - `LeakyRelu`, `PRelu`, `ThresholdedRelu`: Mixed case for compound names
/// - `Selu`, `Celu`, `Elu`: Lowercase suffix for historical reasons
///
/// Note: This is a cosmetic inconsistency (e.g., `ReLU` vs `LeakyRelu`).
/// The pattern reflects organic growth and ONNX/PyTorch naming.
/// Renaming would be a breaking change, so we document rather than standardize.
/// ## Cross-Enum Mismatches (Layer ↔ LayerType)
/// The `Layer` enum in ny-propagate uses different names for some variants.
/// These are cosmetic inconsistencies, not bugs — `FromStr` accepts both forms.
/// | `LayerType` (ny-core) | `Layer` (ny-propagate) | Mismatch |
/// |--------------------------|---------------------------|----------|
/// | `LeakyRelu` | `LeakyReLU` | Case: "lu" vs "LU" |
/// | `RMSNorm` | `RmsNorm` | Case: "RMS" vs "Rms" |
/// | `MaxPool` | `MaxPool2d` | Missing "2d" suffix |
/// | `InstanceNorm` | `InstanceNorm1d` | Missing "1d" suffix |
/// | `AdaIN` | `AdaIN1d` | Missing "1d" suffix |
/// # Layer Conversion
/// Some `LayerType` variants convert to different `Layer` structs during ONNX loading.
/// These conversions are handled in `ny-onnx`:
/// ## Semantic Conversions (different Layer struct)
/// | LayerType | Converts To | Notes |
/// |-----------|-------------|-------|
/// | `RMSNorm` | `RmsNormLayer` | SimplifiedLayerNormalization / RMSNormalization |
/// | `InstanceNorm` | `InstanceNorm1dLayer` | InstanceNormalization (per-channel) |
/// | `AdaIN` | `AdaIN1dLayer` | Adaptive Instance Normalization (style transfer) |
/// | `Neg` | `MulConstantLayer(-1.0)` | Exact: negation as multiplication |
/// | `Triu` | `MulConstantLayer(binary_mask)` | Exact: upper triangular mask |
/// | `Tril` | `MulConstantLayer(binary_mask)` | Exact: lower triangular mask |
/// | `MultiHeadAttention` | `SelfAttentionLayer` | Decomposed Q/K/V attention |
///
/// ## ONNX Op Name Aliases (not LayerType variants)
///
/// These ONNX op names map directly to existing LayerTypes in `ny-onnx/op_map.rs`:
///
/// | ONNX Op | Maps To | Notes |
/// |---------|---------|-------|
/// | `"Swish"` | `SiLU` | ONNX alias for SiLU (x * sigmoid(x)) |
/// | `"GlobalAveragePool"` | `AveragePool` | Treated as regular pooling |
/// | `"Split"` | Multiple `Slice` | Expands to one Slice layer per output |
///
/// ## Context-Dependent Conversions
///
/// These depend on input types (constant vs activation):
///
/// | Operation | If One Input Constant | If Both Activations |
/// |-----------|----------------------|---------------------|
/// | `Add` | `AddConstantLayer` | `AddLayer` (binary) |
/// | `Sub` | `SubConstantLayer` | `SubLayer` (binary) |
/// | `Mul` | `MulConstantLayer` | `MulBinaryLayer` |
/// | `Div` | `DivConstantLayer` | `DivLayer` (binary) |
/// | `MatMul` | `LinearLayer` (weight) | `MatMulLayer` or `BilinearCrownLayer` |
///
/// ## Dimensional Conversions
///
/// | LayerType | 3D Kernel | 4D Kernel |
/// |-----------|-----------|-----------|
/// | `Conv2d` | `Conv1dLayer` | `Conv2dLayer` |
/// | `ConvTranspose2d` | `ConvTranspose1dLayer` | `ConvTranspose2dLayer` |
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LayerType {
    // Basic layers
    Linear,
    Conv1d,
    Conv2d,
    ConvTranspose1d,
    ConvTranspose2d,
    /// Average pooling over spatial dimensions
    AveragePool,
    /// Max pooling over spatial dimensions
    MaxPool,

    // Activations
    ReLU,
    /// Leaky ReLU: y = x if x >= 0, else alpha * x (typically alpha = 0.01)
    LeakyRelu,
    GELU,
    /// SiLU (Swish): x * sigmoid(x), commonly used in LLMs
    SiLU,
    Sigmoid,
    Tanh,
    /// Gaussian error function: y = erf(x)
    Erf,
    /// Softplus: ln(1 + exp(x))
    Softplus,
    Softmax,
    /// Softmax with causal mask (for decoder attention)
    CausalSoftmax,
    /// Clip: clamp values to [min, max] range
    Clip,
    /// ELU (Exponential Linear Unit): x if x >= 0, else alpha * (exp(x) - 1)
    Elu,
    /// SELU (Scaled ELU): lambda * (x if x >= 0, else alpha * (exp(x) - 1))
    /// Uses fixed constants: alpha ≈ 1.6733, lambda ≈ 1.0507
    Selu,
    /// PRelu (Parametric ReLU): y = x if x >= 0, else slope * x
    /// Unlike LeakyRelu, slope is a learned per-channel parameter
    PRelu,
    /// HardSigmoid: y = max(0, min(1, alpha * x + beta))
    /// Default: alpha = 0.2, beta = 0.5. More efficient than Sigmoid.
    HardSigmoid,
    /// HardSwish: y = x * HardSigmoid(x)
    /// Used in MobileNetV3 for efficiency
    HardSwish,
    /// Exponential: y = exp(x)
    Exp,
    /// Natural logarithm: y = ln(x), requires x > 0
    Log,
    /// LogSumExp: log(sum(exp(x))) over specified axes
    LogSumExp,
    /// CELU (Continuous ELU): max(0, x) + min(0, alpha * (exp(x/alpha) - 1))
    /// Continuous at x=0 with continuous derivatives
    Celu,
    /// Mish: x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    /// Self-regularizing non-monotonic activation (YOLOv4, etc)
    Mish,
    /// LogSoftmax: log(softmax(x)) = x - logsumexp(x)
    /// More numerically stable than log(softmax(x))
    LogSoftmax,
    /// ThresholdedRelu: y = x if x > alpha, else 0
    /// Default alpha = 1.0 (unlike ReLU which uses 0)
    ThresholdedRelu,
    /// Shrink: soft thresholding / shrinkage operation
    /// y = x - bias if x > lambd, y = x + bias if x < -lambd, else 0
    /// Default: bias = 0.0, lambd = 0.5
    Shrink,
    /// Softsign: y = x / (1 + |x|)
    /// Output range (-1, 1), similar to tanh but computationally cheaper
    Softsign,
    /// Snake activation: y = x + (1/a) * sin²(a*x)
    /// Used in neural audio synthesis (Ziyin et al., 2020).
    /// Monotonically non-decreasing with frequency parameter a.
    Snake,

    // Rounding operations (for quantization checks)
    /// Floor: y = floor(x) - rounds towards negative infinity
    Floor,
    /// Ceil: y = ceil(x) - rounds towards positive infinity
    Ceil,
    /// Round: y = round(x) - ONNX uses ties to even; bounds also cover half-away runtimes
    Round,
    /// Trunc: y = trunc(x) - rounds toward zero (fractional part discarded).
    /// Produced by ONNX Cast with an integer target dtype on non-constant
    /// input: float->int casts truncate, so dropping them as identity is
    /// unsound for fractional intervals (trunc(0.5)=0 not in [0.5, 62]).
    Trunc,

    // Mathematical operations
    /// Sign: y = -1 if x < 0, 0 if x == 0, 1 if x > 0
    Sign,
    /// Reciprocal: y = 1/x (requires x != 0)
    Reciprocal,

    // Trigonometric (for positional encodings)
    /// Sine function: y = sin(x)
    Sin,
    /// Cosine function: y = cos(x)
    Cos,
    /// Tangent function: y = tan(x)
    Tan,
    /// Arctangent function: y = atan(x)
    Arctan,

    // Positional encoding
    /// RoPE (Rotary Position Embedding): pair-wise 2D rotation for positional encoding.
    /// Used in LLMs (Qwen, LLaMA, Mistral) for relative position information.
    /// At fixed sequence position, this is a linear operation (block-diagonal rotation matrix).
    RoPE,

    // Normalization
    LayerNorm,
    /// RMSNorm: x / sqrt(mean(x^2) + eps), used in LLMs
    RMSNorm,
    /// InstanceNorm: per-channel normalization over time/spatial dims
    /// Used in style transfer and audio models (e.g. the Kokoro TTS family)
    InstanceNorm,
    /// GroupNorm: per-group normalization across channels and spatial dims
    /// Used in Demucs DConv (dilated Conv1d + GroupNorm + GELU)
    GroupNorm,
    /// AdaIN: Adaptive Instance Normalization = style_gamma * InstanceNorm(x) + style_beta
    /// Used in style transfer and audio vocoders (e.g. the Kokoro TTS vocoder)
    AdaIN,
    BatchNorm,
    // Transformer components
    MultiHeadAttention,
    /// Token embedding lookup: indices -> embeddings
    Embedding,
    // Structural
    Add,
    Concat,
    Reshape,
    Flatten,
    /// Tensor transpose (permute axes)
    Transpose,
    /// Cast: type conversion between numeric dtypes (FLOAT, INT64, BOOL, etc.).
    /// Identity for bound propagation since all computation is in f32.
    /// Identity-typed cast node; needed by external verifier dtype-translation lanes.
    Cast,
    /// ONNX DequantizeLinear: y = (x - zero_point) * scale.
    DequantizeLinear,
    /// ONNX QuantizeLinear: y = saturate(round_to_even(x / scale) + zero_point).
    QuantizeLinear,
    /// Squeeze: remove dimension of size 1 at specified axis
    Squeeze,
    /// Unsqueeze: insert dimension of size 1 at specified axis
    Unsqueeze,
    /// Pad: extends tensor axes with constant or reflected border values.
    Pad,
    /// Resize: nearest-neighbor spatial upsample over the last two dimensions.
    Resize,

    // Bounded operations (both inputs are bounded)
    /// Matrix multiplication of two bounded tensors (e.g., Q @ K^T in attention)
    MatMul,
    /// Element-wise multiplication (e.g., attention scaling by constant)
    Mul,
    /// Element-wise minimum (variadic in ONNX, but we support binary form)
    Min,
    /// Element-wise maximum (variadic in ONNX, but we support binary form)
    Max,
    /// Two-argument arctangent: y = atan2(a, b) with output in (-pi, pi].
    Atan2,

    // Element-wise arithmetic
    /// Element-wise negation: y = -x
    Neg,
    /// Upper-triangular masking: keep elements on/above diagonal, zero below.
    Triu,
    /// Lower-triangular masking: keep elements on/below diagonal, zero above.
    Tril,
    /// Element-wise absolute value: y = |x|
    Abs,
    /// Element-wise square root: y = sqrt(x). Assumes x >= 0.
    Sqrt,
    /// Element-wise division: y = x / divisor. Divisor is a constant.
    Div,
    /// Element-wise subtraction: y = x - constant or y = constant - x
    Sub,
    /// Element-wise power: y = x^p where p is a constant.
    Pow,

    // Reduction operations
    /// Mean reduction over specified axes.
    ReduceMean,
    /// Sum reduction over specified axes.
    ReduceSum,
    /// Max reduction over specified axes: y = max(x, axis).
    /// CROWN backward uses fixed_max_index assumption (argmax at center point).
    ReduceMax,
    /// Min reduction over specified axes: y = min(x, axis).
    /// CROWN backward uses fixed_min_index assumption (argmin at center point).
    ReduceMin,
    /// Index of the maximum element along an axis (ONNX ArgMax).
    /// Piecewise-constant: output is the argmax index (exact when provably
    /// unique over the input box, else a sound integer-index interval).
    Argmax,
    /// Index of the minimum element along an axis (ONNX ArgMin).
    ArgMin,
    /// Indices that sort the input along an axis (ONNX ArgSort).
    ArgSort,
    /// Top-K values/indices along an axis (ONNX TopK).
    Topk,
    /// Cumulative sum (prefix sum) along an axis: y[i] = sum(x[0..=i]).
    /// Linear operator — IBP is exact, CROWN backward is O(T) suffix sum.
    CumSum,

    // Tiling/broadcasting operations
    /// Repeat tensor along specified axis: tile(x, reps) repeats x reps times.
    /// Used for GQA (Grouped Query Attention) to expand KV heads to match Q heads.
    /// Attributes: "axis" (i64) - axis to repeat along, "reps" (i64) - repetition count.
    Tile,
    /// Expand a singleton last axis to match a live reference tensor's last axis.
    /// Narrow runtime lowering for ONNX `Expand` on activation paths.
    Expand,

    // Comparison operations
    /// Element-wise comparison (tensor vs scalar): x > t, x >= t, etc.
    /// Output is {0.0, 1.0}. IBP uses paired bounds analysis; CROWN uses zero-slope relaxation.
    Compare,
    /// Element-wise comparison (tensor vs tensor): a > b, a >= b, etc.
    /// Output is {0.0, 1.0}. IBP-only (no meaningful CROWN linear relaxation for binary comparison).
    CompareTensor,

    // Conditional operations
    /// Element-wise conditional: Where(condition, x, y) = x if condition else y.
    /// For interval bounds, takes union of x and y bounds.
    Where,

    // Index/selection operations
    /// NonZero: returns indices of non-zero elements.
    /// Output shape is data-dependent: [rank(input), num_nonzero].
    /// For bound propagation, returns conservative bounds on possible indices.
    NonZero,
    /// Gather: index/select elements along an axis using an indices tensor.
    /// Output shape replaces the axis dimension with indices shape.
    Gather,
    /// ScatterND: scatter update values into data at computed indices.
    ScatterND,
    /// Slice: extracts a contiguous range along an axis.
    /// Used to implement Split (multi-output op) as multiple Slice layers.
    /// Attributes: "axis" (i64), "start" (i64), "end" (i64).
    Slice,
    /// Shape: returns the shape of a tensor as a 1-D integer tensor.
    /// Input: activation tensor. Output: 1-D tensor of shape dimensions (as f32).
    /// ONNX opset 1+, opset>=15 supports optional start/end attributes for range selection.
    Shape,

    // Placeholder for unsupported ops
    Unknown,
}

impl fmt::Display for LayerType {
    #[allow(unreachable_patterns)] // Needed for #[non_exhaustive] forward compatibility
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LayerType::Linear => write!(f, "Linear"),
            LayerType::Conv1d => write!(f, "Conv1d"),
            LayerType::Conv2d => write!(f, "Conv2d"),
            LayerType::ConvTranspose1d => write!(f, "ConvTranspose1d"),
            LayerType::ConvTranspose2d => write!(f, "ConvTranspose2d"),
            LayerType::AveragePool => write!(f, "AveragePool"),
            LayerType::MaxPool => write!(f, "MaxPool"),
            LayerType::ReLU => write!(f, "ReLU"),
            LayerType::LeakyRelu => write!(f, "LeakyRelu"),
            LayerType::GELU => write!(f, "GELU"),
            LayerType::SiLU => write!(f, "SiLU"),
            LayerType::Sigmoid => write!(f, "Sigmoid"),
            LayerType::Tanh => write!(f, "Tanh"),
            LayerType::Erf => write!(f, "Erf"),
            LayerType::Softplus => write!(f, "Softplus"),
            LayerType::Softmax => write!(f, "Softmax"),
            LayerType::CausalSoftmax => write!(f, "CausalSoftmax"),
            LayerType::Clip => write!(f, "Clip"),
            LayerType::Elu => write!(f, "Elu"),
            LayerType::Selu => write!(f, "Selu"),
            LayerType::PRelu => write!(f, "PRelu"),
            LayerType::HardSigmoid => write!(f, "HardSigmoid"),
            LayerType::HardSwish => write!(f, "HardSwish"),
            LayerType::Exp => write!(f, "Exp"),
            LayerType::Log => write!(f, "Log"),
            LayerType::LogSumExp => write!(f, "LogSumExp"),
            LayerType::Celu => write!(f, "Celu"),
            LayerType::Mish => write!(f, "Mish"),
            LayerType::LogSoftmax => write!(f, "LogSoftmax"),
            LayerType::ThresholdedRelu => write!(f, "ThresholdedRelu"),
            LayerType::Shrink => write!(f, "Shrink"),
            LayerType::Softsign => write!(f, "Softsign"),
            LayerType::Snake => write!(f, "Snake"),
            LayerType::Floor => write!(f, "Floor"),
            LayerType::Ceil => write!(f, "Ceil"),
            LayerType::Round => write!(f, "Round"),
            LayerType::Trunc => write!(f, "Trunc"),
            LayerType::Sign => write!(f, "Sign"),
            LayerType::Reciprocal => write!(f, "Reciprocal"),
            LayerType::Sin => write!(f, "Sin"),
            LayerType::Cos => write!(f, "Cos"),
            LayerType::Tan => write!(f, "Tan"),
            LayerType::Arctan => write!(f, "Arctan"),
            LayerType::RoPE => write!(f, "RoPE"),
            LayerType::LayerNorm => write!(f, "LayerNorm"),
            LayerType::RMSNorm => write!(f, "RMSNorm"),
            LayerType::InstanceNorm => write!(f, "InstanceNorm"),
            LayerType::GroupNorm => write!(f, "GroupNorm"),
            LayerType::AdaIN => write!(f, "AdaIN"),
            LayerType::BatchNorm => write!(f, "BatchNorm"),
            LayerType::MultiHeadAttention => write!(f, "MultiHeadAttention"),
            LayerType::Embedding => write!(f, "Embedding"),
            LayerType::Add => write!(f, "Add"),
            LayerType::Concat => write!(f, "Concat"),
            LayerType::Reshape => write!(f, "Reshape"),
            LayerType::Flatten => write!(f, "Flatten"),
            LayerType::Transpose => write!(f, "Transpose"),
            LayerType::Cast => write!(f, "Cast"),
            LayerType::DequantizeLinear => write!(f, "DequantizeLinear"),
            LayerType::QuantizeLinear => write!(f, "QuantizeLinear"),
            LayerType::Squeeze => write!(f, "Squeeze"),
            LayerType::Unsqueeze => write!(f, "Unsqueeze"),
            LayerType::Pad => write!(f, "Pad"),
            LayerType::Resize => write!(f, "Resize"),
            LayerType::MatMul => write!(f, "MatMul"),
            LayerType::Mul => write!(f, "Mul"),
            LayerType::Min => write!(f, "Min"),
            LayerType::Max => write!(f, "Max"),
            LayerType::Atan2 => write!(f, "Atan2"),
            LayerType::Neg => write!(f, "Neg"),
            LayerType::Triu => write!(f, "Triu"),
            LayerType::Tril => write!(f, "Tril"),
            LayerType::Abs => write!(f, "Abs"),
            LayerType::Sqrt => write!(f, "Sqrt"),
            LayerType::Div => write!(f, "Div"),
            LayerType::Sub => write!(f, "Sub"),
            LayerType::Pow => write!(f, "Pow"),
            LayerType::ReduceMean => write!(f, "ReduceMean"),
            LayerType::ReduceSum => write!(f, "ReduceSum"),
            LayerType::ReduceMax => write!(f, "ReduceMax"),
            LayerType::ReduceMin => write!(f, "ReduceMin"),
            LayerType::Argmax => write!(f, "Argmax"),
            LayerType::ArgMin => write!(f, "ArgMin"),
            LayerType::ArgSort => write!(f, "ArgSort"),
            LayerType::Topk => write!(f, "Topk"),
            LayerType::CumSum => write!(f, "CumSum"),
            LayerType::Tile => write!(f, "Tile"),
            LayerType::Expand => write!(f, "Expand"),
            LayerType::Compare => write!(f, "Compare"),
            LayerType::CompareTensor => write!(f, "CompareTensor"),
            LayerType::Where => write!(f, "Where"),
            LayerType::NonZero => write!(f, "NonZero"),
            LayerType::Gather => write!(f, "Gather"),
            LayerType::ScatterND => write!(f, "ScatterND"),
            LayerType::Slice => write!(f, "Slice"),
            LayerType::Shape => write!(f, "Shape"),
            LayerType::Unknown => write!(f, "Unknown"),
            // Handle non_exhaustive - future variants
            _ => write!(f, "Unknown"),
        }
    }
}

impl FromStr for LayerType {
    type Err = ();

    /// Parse a layer type from its string representation.
    ///
    /// Returns `LayerType::Unknown` for unrecognized strings rather than an error,
    /// matching the enum's purpose of gracefully handling unsupported ops.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "Linear" => LayerType::Linear,
            "Conv1d" => LayerType::Conv1d,
            "Conv2d" => LayerType::Conv2d,
            "ConvTranspose1d" => LayerType::ConvTranspose1d,
            "ConvTranspose2d" => LayerType::ConvTranspose2d,
            "AveragePool" => LayerType::AveragePool,
            "MaxPool" | "MaxPool2d" => LayerType::MaxPool,
            "ReLU" => LayerType::ReLU,
            "LeakyRelu" | "LeakyReLU" => LayerType::LeakyRelu,
            "GELU" => LayerType::GELU,
            "SiLU" => LayerType::SiLU,
            "Sigmoid" => LayerType::Sigmoid,
            "Tanh" => LayerType::Tanh,
            "Erf" => LayerType::Erf,
            "Softplus" => LayerType::Softplus,
            "Softmax" => LayerType::Softmax,
            "CausalSoftmax" => LayerType::CausalSoftmax,
            "Clip" => LayerType::Clip,
            "Elu" => LayerType::Elu,
            "Selu" => LayerType::Selu,
            "PRelu" => LayerType::PRelu,
            "HardSigmoid" => LayerType::HardSigmoid,
            "HardSwish" => LayerType::HardSwish,
            "Exp" => LayerType::Exp,
            "Log" => LayerType::Log,
            "LogSumExp" => LayerType::LogSumExp,
            "Celu" => LayerType::Celu,
            "Mish" => LayerType::Mish,
            "LogSoftmax" => LayerType::LogSoftmax,
            "ThresholdedRelu" => LayerType::ThresholdedRelu,
            "Shrink" => LayerType::Shrink,
            "Softsign" => LayerType::Softsign,
            "Snake" => LayerType::Snake,
            "Floor" => LayerType::Floor,
            "Ceil" => LayerType::Ceil,
            "Round" => LayerType::Round,
            "Trunc" => LayerType::Trunc,
            "Sign" => LayerType::Sign,
            "Reciprocal" => LayerType::Reciprocal,
            "Sin" => LayerType::Sin,
            "Cos" => LayerType::Cos,
            "Tan" => LayerType::Tan,
            "Arctan" => LayerType::Arctan,
            "RoPE" | "RotaryPositionEmbedding" => LayerType::RoPE,
            "LayerNorm" => LayerType::LayerNorm,
            "RMSNorm" | "RmsNorm" => LayerType::RMSNorm,
            "InstanceNorm" | "InstanceNorm1d" => LayerType::InstanceNorm,
            "GroupNorm" | "GroupNormalization" => LayerType::GroupNorm,
            "AdaIN" | "AdaIN1d" | "AdaptiveInstanceNorm" => LayerType::AdaIN,
            "BatchNorm" => LayerType::BatchNorm,
            "MultiHeadAttention" | "SelfAttention" => LayerType::MultiHeadAttention,
            "Embedding" => LayerType::Embedding,
            "Add" => LayerType::Add,
            "Concat" => LayerType::Concat,
            "Reshape" => LayerType::Reshape,
            "Flatten" => LayerType::Flatten,
            "Transpose" => LayerType::Transpose,
            "Cast" => LayerType::Cast,
            "DequantizeLinear" => LayerType::DequantizeLinear,
            "QuantizeLinear" => LayerType::QuantizeLinear,
            "Squeeze" => LayerType::Squeeze,
            "Unsqueeze" => LayerType::Unsqueeze,
            "Pad" => LayerType::Pad,
            "Resize" | "Upsample" => LayerType::Resize,
            "MatMul" => LayerType::MatMul,
            "Mul" | "MulBinary" => LayerType::Mul,
            "Min" | "MinBinary" => LayerType::Min,
            "Max" | "MaxBinary" => LayerType::Max,
            "Atan2" => LayerType::Atan2,
            "Neg" => LayerType::Neg,
            "Triu" => LayerType::Triu,
            "Tril" => LayerType::Tril,
            "Abs" => LayerType::Abs,
            "Sqrt" => LayerType::Sqrt,
            "Div" => LayerType::Div,
            "Sub" => LayerType::Sub,
            "Pow" => LayerType::Pow,
            "ReduceMean" => LayerType::ReduceMean,
            "ReduceSum" => LayerType::ReduceSum,
            "Argmax" | "ArgMax" => LayerType::Argmax,
            "ArgMin" | "Argmin" => LayerType::ArgMin,
            "ArgSort" | "Argsort" => LayerType::ArgSort,
            "Topk" | "TopK" => LayerType::Topk,
            "ReduceMax" => LayerType::ReduceMax,
            "ReduceMin" => LayerType::ReduceMin,
            "CumSum" => LayerType::CumSum,
            "Tile" => LayerType::Tile,
            "Expand" => LayerType::Expand,
            "Compare" => LayerType::Compare,
            "CompareTensor" => LayerType::CompareTensor,
            "Where" => LayerType::Where,
            "NonZero" => LayerType::NonZero,
            "Gather" => LayerType::Gather,
            "ScatterND" | "ScatterNd" => LayerType::ScatterND,
            "Slice" => LayerType::Slice,
            "Shape" => LayerType::Shape,
            _ => LayerType::Unknown,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// #3176: Verify all cross-enum aliases resolve to the correct LayerType.
    /// These aliases bridge the naming gap between Layer (ny-propagate) and
    /// LayerType (ny-core).
    #[test]
    fn from_str_cross_enum_aliases_3176() {
        // LeakyRelu ↔ LeakyReLU
        assert_eq!(LayerType::from_str("LeakyRelu"), Ok(LayerType::LeakyRelu));
        assert_eq!(LayerType::from_str("LeakyReLU"), Ok(LayerType::LeakyRelu));

        // RMSNorm ↔ RmsNorm
        assert_eq!(LayerType::from_str("RMSNorm"), Ok(LayerType::RMSNorm));
        assert_eq!(LayerType::from_str("RmsNorm"), Ok(LayerType::RMSNorm));

        // MaxPool ↔ MaxPool2d
        assert_eq!(LayerType::from_str("MaxPool"), Ok(LayerType::MaxPool));
        assert_eq!(LayerType::from_str("MaxPool2d"), Ok(LayerType::MaxPool));

        // InstanceNorm ↔ InstanceNorm1d
        assert_eq!(
            LayerType::from_str("InstanceNorm"),
            Ok(LayerType::InstanceNorm)
        );
        assert_eq!(
            LayerType::from_str("InstanceNorm1d"),
            Ok(LayerType::InstanceNorm)
        );

        // AdaIN ↔ AdaIN1d
        assert_eq!(LayerType::from_str("AdaIN"), Ok(LayerType::AdaIN));
        assert_eq!(LayerType::from_str("AdaIN1d"), Ok(LayerType::AdaIN));

        // Resize ↔ Upsample
        assert_eq!(LayerType::from_str("Resize"), Ok(LayerType::Resize));
        assert_eq!(LayerType::from_str("Upsample"), Ok(LayerType::Resize));
    }

    #[test]
    fn atan2_round_trips_via_display_and_from_str() {
        assert_eq!(LayerType::from_str("Atan2"), Ok(LayerType::Atan2));
        assert_eq!(LayerType::Atan2.to_string(), "Atan2");
    }

    #[test]
    fn triangular_mask_ops_round_trip_via_display_and_from_str_4270() {
        assert_eq!(LayerType::from_str("Triu"), Ok(LayerType::Triu));
        assert_eq!(LayerType::Triu.to_string(), "Triu");

        assert_eq!(LayerType::from_str("Tril"), Ok(LayerType::Tril));
        assert_eq!(LayerType::Tril.to_string(), "Tril");
    }
}
