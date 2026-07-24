// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! NY-owned, `serde`-serializable mirror of a traced computation graph.
//!
//! This module is the stable cross-repo contract for the trace bridge. An ML
//! framework (today: NN's `nn-core` `DynTensor` tracer) serializes its captured
//! computation graph into these types; [`crate::translate`] then lowers them
//! into an `ny_build::GraphModel`. By owning the schema here — rather than
//! depending on the framework's IR — NY decouples the verifier from any single
//! framework and pins the op set / payload shapes as a versioned wire format.
//!
//! ## Fidelity to the source IR
//!
//! Every type here is a faithful, field-for-field mirror of `nn-core`'s trace IR
//! (`trace_types.rs`, `trace_node.rs`, `trace_graph.rs`) as of the port. In
//! particular [`TraceOp`] mirrors all **123** variants of the source enum with
//! their exact payloads. The `tests` module asserts that count so drift is
//! caught at build time. Where the source carries an internal enum or struct as
//! a field (`CompareOp`, `KokoroFusedOp`, …) we mirror a serde-able NY-owned
//! equivalent rather than reaching across the crate boundary.
//!
//! ## Weights
//!
//! The source flattens every weight to `Vec<f32>` + shape today. We carry a
//! dtype-tagged [`WeightPayload`] instead so weights round-trip losslessly and
//! the schema is forward-compatible with quantized / low-precision intake
//! (roadmap P8) without a wire-format break.

use serde::{Deserialize, Serialize};

/// Unique identifier for a traced tensor.
///
/// Mirrors `nn-core`'s `pub type NodeId = u64`. Kept as a transparent newtype
/// so the wire format is an integer while the type system distinguishes node
/// references from arbitrary integers in translator code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct NodeId(pub u64);

impl NodeId {
    /// Returns the raw integer identifier.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

impl From<u64> for NodeId {
    fn from(value: u64) -> Self {
        Self(value)
    }
}

impl From<NodeId> for u64 {
    fn from(value: NodeId) -> Self {
        value.0
    }
}

/// Tensor data type.
///
/// Mirrors `nn-core`'s `DType` (all nine variants). The translator maps the
/// float/int subset that NY's `ny_build::DataType` supports and rejects the
/// rest with an explicit error (sound by construction).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DType {
    /// 32-bit IEEE-754 float.
    F32,
    /// 16-bit IEEE-754 float.
    F16,
    /// 16-bit brain float.
    Bf16,
    /// 64-bit IEEE-754 float.
    F64,
    /// 32-bit signed integer.
    I32,
    /// 64-bit signed integer.
    I64,
    /// 32-bit unsigned integer.
    U32,
    /// 8-bit unsigned integer.
    U8,
    /// Boolean.
    Bool,
}

impl DType {
    /// Size of a single element in bytes.
    #[must_use]
    pub const fn size_bytes(self) -> usize {
        match self {
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::F16 | Self::Bf16 => 2,
            Self::F64 | Self::I64 => 8,
            Self::U8 | Self::Bool => 1,
        }
    }

    /// Returns `true` for the floating-point dtypes.
    #[must_use]
    pub const fn is_float(self) -> bool {
        matches!(self, Self::F32 | Self::F16 | Self::Bf16 | Self::F64)
    }

    /// Returns `true` for the signed/unsigned integer dtypes.
    #[must_use]
    pub const fn is_int(self) -> bool {
        matches!(self, Self::I32 | Self::I64 | Self::U32 | Self::U8)
    }
}

/// Dtype-tagged weight payload mirroring a traced weight tensor.
///
/// The source IR (`nn-core::WeightRef`) flattens weights to `Vec<f32>` + shape.
/// We preserve that exact representation in [`WeightData::F32`] for a faithful
/// round-trip of today's traces, while the dtype-tagged variants leave room for
/// lossless quantized / low-precision intake (roadmap P8). A shape-only
/// [`WeightData::Placeholder`] mirrors `WeightRef::from_shape` (the last-resort
/// fallback when data extraction fails).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WeightPayload {
    /// Original shape of the weight tensor.
    pub shape: Vec<usize>,
    /// Element data, tagged by its original dtype.
    pub data: WeightData,
}

impl WeightPayload {
    /// Construct an f32 weight payload (the representation NN emits today).
    #[must_use]
    pub fn f32(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self {
            shape,
            data: WeightData::F32(data),
        }
    }

    /// Construct a shape-only placeholder (no element data).
    ///
    /// Mirrors `nn-core::WeightRef::from_shape`: used when the source could not
    /// extract concrete weight values (e.g. an unsupported source dtype).
    #[must_use]
    pub fn placeholder(shape: Vec<usize>) -> Self {
        Self {
            shape,
            data: WeightData::Placeholder,
        }
    }

    /// Number of elements implied by the shape.
    #[must_use]
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Returns `true` when this carries no element data (shape-only).
    ///
    /// Mirrors the spirit of `nn-core::WeightRef::is_placeholder`: a non-empty
    /// shape with all-positive dims but no data.
    #[must_use]
    pub fn is_placeholder(&self) -> bool {
        matches!(self.data, WeightData::Placeholder)
            && !self.shape.is_empty()
            && self.shape.iter().all(|&d| d > 0)
    }
}

/// Dtype-tagged element storage for a [`WeightPayload`].
///
/// `F16`/`Bf16` use `half`'s serde-enabled types so low-precision weights
/// round-trip bit-exactly. The translator dequantizes as needed; the schema
/// itself stays lossless.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum WeightData {
    /// 32-bit float elements (the representation NN emits today).
    F32(Vec<f32>),
    /// 64-bit float elements.
    F64(Vec<f64>),
    /// 16-bit float elements.
    F16(Vec<half::f16>),
    /// 16-bit brain-float elements.
    Bf16(Vec<half::bf16>),
    /// 32-bit signed integer elements.
    I32(Vec<i32>),
    /// 64-bit signed integer elements.
    I64(Vec<i64>),
    /// Shape-only placeholder: data could not be captured at trace time.
    Placeholder,
}

// -- TraceOp support types ----------------------------------------------------

/// Element-wise comparison discriminant.
///
/// Mirrors `nn-core`'s `CompareOp`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CompareOp {
    /// Equal (`==`).
    Eq,
    /// Not equal (`!=`).
    Ne,
    /// Greater than or equal (`>=`).
    Ge,
    /// Greater than (`>`).
    Gt,
    /// Less than (`<`).
    Lt,
    /// Less than or equal (`<=`).
    Le,
}

/// Out-of-bounds padding mode for grid sampling.
///
/// Mirrors `nn-core`'s `GridSamplePaddingMode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GridSamplePaddingMode {
    /// Out-of-bounds positions return 0.
    Zeros,
    /// Out-of-bounds coordinates are clamped to the border pixel.
    Border,
}

/// Named activation for the generic [`TraceOp::Activation`] variant.
///
/// Mirrors `nn-core`'s `TraceActivation`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TraceActivation {
    /// Rectified linear unit.
    Relu,
    /// Gaussian error linear unit (tanh approximation).
    Gelu,
    /// Gaussian error linear unit (exact, erf-based).
    GeluErf,
    /// Sigmoid-weighted linear unit (`x * sigmoid(x)`).
    Silu,
    /// Logistic sigmoid.
    Sigmoid,
    /// Hyperbolic tangent.
    Tanh,
    /// Exponential.
    Exp,
    /// Natural logarithm.
    Log,
    /// Exponential linear unit.
    Elu,
    /// Leaky rectified linear unit.
    LeakyRelu,
    /// Mish (`x * tanh(softplus(x))`).
    Mish,
}

/// Upsampling interpolation mode for [`TraceOp::Upsample2d`].
///
/// Mirrors `nn-core`'s `TraceUpsampleMode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TraceUpsampleMode {
    /// Nearest-neighbor.
    Nearest,
    /// Bilinear interpolation.
    Bilinear,
    /// Bicubic interpolation.
    Bicubic,
}

/// Activation variant for [`KokoroFusedOp::FusedAdainResBlock`].
///
/// Mirrors `nn-core`'s `ResBlockActivation`.
// `Snake` carries two WeightPayloads vs `LeakyRelu`'s single f64 by design —
// mirrors the upstream `#[allow(clippy::large_enum_variant)]`.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ResBlockActivation {
    /// Snake activation: `x + (1/alpha) * sin²(alpha * x)` (Generator ResBlock).
    Snake {
        /// Per-channel alpha for the first activation, shape `[1, C, 1]`.
        alpha1: WeightPayload,
        /// Per-channel alpha for the second activation, shape `[1, C, 1]`.
        alpha2: WeightPayload,
    },
    /// Leaky ReLU activation with the given negative slope (F0 predictor).
    LeakyRelu {
        /// Negative slope (typically `0.2`).
        slope: f64,
    },
}

/// Kokoro-specific fused activation / normalization operations.
///
/// Mirrors `nn-core`'s `KokoroFusedOp`.
// `FusedAdainResBlock` carries eight WeightPayloads by design — mirrors the
// upstream `#[allow(clippy::large_enum_variant)]`.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum KokoroFusedOp {
    /// Per-channel Snake activation: `x + (1/alpha) * sin²(alpha * x)`.
    SnakeTensor {
        /// Per-channel alpha weight, typically shape `[1, C, 1]`.
        alpha: WeightPayload,
    },
    /// Fused AdaIN + Snake: `InstanceNorm → affine(gamma, beta) → Snake(alpha)`.
    AdainSnake {
        /// Per-channel Snake alpha.
        alpha: WeightPayload,
        /// InstanceNorm epsilon.
        eps: f64,
    },
    /// Fused AdaIN + LeakyRelu: `InstanceNorm → affine(gamma, beta) → LeakyRelu(slope)`.
    AdainLeakyRelu {
        /// InstanceNorm epsilon.
        eps: f64,
        /// LeakyRelu negative slope.
        slope: f64,
    },
    /// Fused Adaptive LayerNorm: `(1+gamma) * LayerNorm(x, w, b) + beta`.
    AdaLayerNorm {
        /// LayerNorm weight (scale).
        norm_weight: WeightPayload,
        /// LayerNorm bias (shift).
        norm_bias: WeightPayload,
        /// LayerNorm epsilon.
        eps: f64,
    },
    /// Fused AdaIN residual block: two `(AdaIN → activation → Conv1d)` pairs + residual.
    FusedAdainResBlock {
        /// Snake (Generator) or LeakyRelu (F0) activation.
        activation: ResBlockActivation,
        /// AdaIN 1: style projection weight `[2*C_in, S]`.
        adain1_weight: WeightPayload,
        /// AdaIN 1: style projection bias `[2*C_in]`.
        adain1_bias: WeightPayload,
        /// AdaIN 2: style projection weight `[2*C_out, S]`.
        adain2_weight: WeightPayload,
        /// AdaIN 2: style projection bias `[2*C_out]`.
        adain2_bias: WeightPayload,
        /// Conv1 weight `[C_out, C_in, K]`.
        conv1_weight: WeightPayload,
        /// Conv1 bias `[C_out]`.
        conv1_bias: WeightPayload,
        /// Conv1 dilation factor.
        conv1_dilation: usize,
        /// Conv1 padding.
        conv1_padding: usize,
        /// Conv2 weight `[C_out, C_out, K]`.
        conv2_weight: WeightPayload,
        /// Conv2 bias `[C_out]`.
        conv2_bias: WeightPayload,
        /// Conv2 padding.
        conv2_padding: usize,
        /// InstanceNorm epsilon.
        eps: f64,
        /// Residual scale factor (`1.0` for Generator, `1/√2` for F0).
        residual_scale: f64,
    },
}

// -- TraceOp ------------------------------------------------------------------

/// An operation recorded during tracing.
///
/// Faithful NY-owned mirror of `nn-core`'s `TraceOp` enum — all **123** variants
/// with their exact payloads. Variants are grouped to match the source layout
/// for easy cross-referencing. This is the central type of the wire contract;
/// the `tests` module pins the variant count so any source drift is caught.
// `KokoroFused` (and its `FusedAdainResBlock`) dominates the size by design —
// mirrors the upstream `#[allow(clippy::large_enum_variant)]`.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TraceOp {
    /// Network input placeholder.
    Input,
    /// Constant weight tensor auto-registered during tracing.
    ConstantWeight {
        /// The captured weight.
        weight: WeightPayload,
    },

    // -- Binary element-wise --
    /// Element-wise addition.
    Add,
    /// Element-wise subtraction.
    Sub,
    /// Element-wise multiplication.
    Mul,
    /// Element-wise division.
    Div,
    /// Element-wise maximum.
    Maximum,
    /// Element-wise minimum.
    Minimum,

    // -- Matrix multiply --
    /// Raw variable-variable matmul (compilable but not verifiable via NY).
    MatMul,

    // -- Unary element-wise --
    /// Rectified linear unit.
    Relu,
    /// Gaussian error linear unit (tanh approximation).
    Gelu,
    /// Gaussian error linear unit (exact, erf-based).
    GeluErf,
    /// Sigmoid-weighted linear unit.
    Silu,
    /// Hyperbolic tangent.
    Tanh,
    /// Logistic sigmoid.
    Sigmoid,
    /// Exponential.
    Exp,
    /// Natural logarithm.
    Log,
    /// Square root.
    Sqrt,
    /// Square (`x²`).
    Sqr,
    /// Absolute value.
    Abs,
    /// Negation.
    Neg,
    /// Reciprocal (`1/x`).
    Recip,
    /// Sine.
    Sin,
    /// Cosine.
    Cos,
    /// Tangent.
    Tan,
    /// Floor.
    Floor,
    /// Ceiling.
    Ceil,
    /// Round to nearest.
    Round,
    /// Sign (`-1`, `0`, `1`).
    Sign,
    /// Fractional part.
    Fract,

    // -- Reductions --
    /// Sum reduction along a dimension.
    ReduceSum {
        /// Dimension to reduce.
        dim: usize,
        /// Keep the reduced dimension as size 1.
        keepdim: bool,
    },
    /// Mean reduction along a dimension.
    ReduceMean {
        /// Dimension to reduce.
        dim: usize,
        /// Keep the reduced dimension as size 1.
        keepdim: bool,
    },
    /// Maximum reduction along a dimension.
    ReduceMax {
        /// Dimension to reduce.
        dim: usize,
        /// Keep the reduced dimension as size 1.
        keepdim: bool,
    },
    /// Minimum reduction along a dimension.
    ReduceMin {
        /// Dimension to reduce.
        dim: usize,
        /// Keep the reduced dimension as size 1.
        keepdim: bool,
    },

    // -- Shape operations --
    /// Reshape to a target shape.
    Reshape {
        /// Target shape.
        target_shape: Vec<usize>,
    },
    /// Swap two dimensions.
    Transpose {
        /// First dimension.
        dim0: usize,
        /// Second dimension.
        dim1: usize,
    },
    /// Narrow a dimension to a contiguous range.
    Narrow {
        /// Dimension to narrow.
        dim: usize,
        /// Start index.
        start: usize,
        /// Length of the slice.
        length: usize,
    },
    /// Insert a size-1 dimension.
    Unsqueeze {
        /// Dimension at which to insert.
        dim: usize,
    },
    /// Remove a size-1 dimension.
    Squeeze {
        /// Dimension to remove.
        dim: usize,
    },
    /// Permute dimensions by an axis ordering.
    Permute {
        /// New axis order.
        axes: Vec<usize>,
    },
    /// Concatenate inputs along a dimension.
    Cat {
        /// Concatenation dimension.
        dim: usize,
        /// Number of input tensors.
        num_inputs: usize,
    },

    // -- Normalization --
    /// Layer normalization with affine parameters.
    LayerNorm {
        /// Numerical-stability epsilon.
        eps: f64,
        /// Scale weight.
        weight: WeightPayload,
        /// Shift bias.
        bias: WeightPayload,
    },
    /// RMS normalization with a scale weight.
    RmsNorm {
        /// Numerical-stability epsilon.
        eps: f64,
        /// Scale weight.
        weight: WeightPayload,
    },
    /// Group normalization with affine parameters.
    GroupNorm {
        /// Number of groups.
        num_groups: usize,
        /// Numerical-stability epsilon.
        eps: f64,
        /// Scale weight.
        weight: WeightPayload,
        /// Shift bias.
        bias: WeightPayload,
    },
    /// Instance normalization.
    InstanceNorm {
        /// Numerical-stability epsilon.
        eps: f64,
    },
    /// Batch normalization with running statistics.
    BatchNorm {
        /// Numerical-stability epsilon.
        eps: f64,
        /// Scale weight.
        weight: WeightPayload,
        /// Shift bias.
        bias: WeightPayload,
        /// Running mean.
        running_mean: WeightPayload,
        /// Running variance.
        running_var: WeightPayload,
    },

    // -- Linear / Conv --
    /// Linear layer: `y = x @ weight^T + bias`.
    Linear {
        /// Weight matrix.
        weight: WeightPayload,
        /// Optional bias.
        bias: Option<WeightPayload>,
    },
    /// 1-D convolution.
    Conv1d {
        /// Kernel weight.
        weight: WeightPayload,
        /// Optional bias.
        bias: Option<WeightPayload>,
        /// Padding.
        padding: usize,
        /// Stride.
        stride: usize,
        /// Dilation.
        dilation: usize,
        /// Number of groups.
        groups: usize,
    },
    /// 2-D convolution.
    Conv2d {
        /// Kernel weight.
        weight: WeightPayload,
        /// Optional bias.
        bias: Option<WeightPayload>,
        /// Padding `[h, w]`.
        padding: [usize; 2],
        /// Stride `[h, w]`.
        stride: [usize; 2],
        /// Dilation `[h, w]`.
        dilation: [usize; 2],
        /// Number of groups.
        groups: usize,
    },
    /// 3-D convolution.
    Conv3d {
        /// Kernel weight.
        weight: WeightPayload,
        /// Optional bias.
        bias: Option<WeightPayload>,
        /// Padding `[d, h, w]`.
        padding: [usize; 3],
        /// Stride `[d, h, w]`.
        stride: [usize; 3],
        /// Dilation `[d, h, w]`.
        dilation: [usize; 3],
        /// Number of groups.
        groups: usize,
    },
    /// 1-D transposed convolution.
    ConvTranspose1d {
        /// Kernel weight.
        weight: WeightPayload,
        /// Optional bias.
        bias: Option<WeightPayload>,
        /// Padding.
        padding: usize,
        /// Output padding.
        output_padding: usize,
        /// Stride.
        stride: usize,
        /// Dilation.
        dilation: usize,
        /// Number of groups.
        groups: usize,
    },
    /// 2-D transposed convolution.
    ConvTranspose2d {
        /// Kernel weight.
        weight: WeightPayload,
        /// Optional bias.
        bias: Option<WeightPayload>,
        /// Padding `[h, w]`.
        padding: [usize; 2],
        /// Output padding `[h, w]`.
        output_padding: [usize; 2],
        /// Stride `[h, w]`.
        stride: [usize; 2],
        /// Dilation `[h, w]`.
        dilation: [usize; 2],
        /// Number of groups.
        groups: usize,
    },

    // -- Attention --
    /// Softmax along a dimension.
    Softmax {
        /// Softmax dimension.
        dim: usize,
    },
    /// Log-softmax along a dimension.
    LogSoftmax {
        /// Softmax dimension.
        dim: usize,
    },
    /// Scaled dot-product attention: `softmax(Q @ Kᵀ * scale + mask) @ V`.
    Sdpa {
        /// Attention scale factor.
        scale: f64,
    },
    /// Causal SDPA (no explicit mask tensor); inputs are Q, K, V.
    SdpaCausal {
        /// Attention scale factor.
        scale: f64,
    },
    /// Rotary position embedding applied to a Q or K tensor.
    RotaryEmbedding {
        /// Per-head dimension.
        head_dim: usize,
        /// Position offset.
        offset: usize,
        /// Narrowed cos frequencies, shape `[seq_len, head_dim/2]`.
        cos_cache: WeightPayload,
        /// Narrowed sin frequencies, shape `[seq_len, head_dim/2]`.
        sin_cache: WeightPayload,
    },
    /// Multi-head attention composite: Q/K/V proj → SDPA → output proj.
    MultiHeadAttention {
        /// Number of query heads.
        num_heads: usize,
        /// Number of key/value heads (for grouped-query attention).
        num_kv_heads: usize,
        /// Per-head dimension.
        head_dim: usize,
    },
    /// Embedding lookup.
    Embedding {
        /// Embedding table weight.
        weight: WeightPayload,
    },

    // -- Recurrent --
    /// LSTM cell.
    Lstm {
        /// Input-to-hidden weight.
        weight_ih: WeightPayload,
        /// Hidden-to-hidden weight.
        weight_hh: WeightPayload,
        /// Optional input-to-hidden bias.
        bias_ih: Option<WeightPayload>,
        /// Optional hidden-to-hidden bias.
        bias_hh: Option<WeightPayload>,
        /// Hidden state size.
        hidden_size: usize,
        /// Optional initial hidden state node (`None` = zero-initialized).
        initial_hidden: Option<NodeId>,
        /// Optional initial cell state node (`None` = zero-initialized).
        initial_cell: Option<NodeId>,
    },

    // -- Pooling --
    /// 1-D max pooling.
    MaxPool1d {
        /// Kernel size.
        kernel_size: usize,
        /// Stride.
        stride: usize,
        /// Padding.
        padding: usize,
    },
    /// 2-D average pooling.
    AvgPool2d {
        /// Kernel size `[h, w]`.
        kernel_size: [usize; 2],
        /// Stride `[h, w]`.
        stride: [usize; 2],
        /// Padding `[h, w]`.
        padding: [usize; 2],
    },
    /// 2-D max pooling.
    MaxPool2d {
        /// Kernel size `[h, w]`.
        kernel_size: [usize; 2],
        /// Stride `[h, w]`.
        stride: [usize; 2],
        /// Padding `[h, w]`.
        padding: [usize; 2],
    },
    /// Adaptive 2-D average pooling to a fixed output size.
    AdaptiveAvgPool2d {
        /// Output spatial size `[h, w]`.
        output_size: [usize; 2],
    },
    /// 1-D average pooling.
    AvgPool1d {
        /// Kernel size.
        kernel_size: usize,
        /// Stride.
        stride: usize,
        /// Padding.
        padding: usize,
    },
    /// Adaptive 1-D average pooling to a fixed output size.
    AdaptiveAvgPool1d {
        /// Output spatial size.
        output_size: usize,
    },
    /// Adaptive 2-D max pooling to a fixed output size.
    AdaptiveMaxPool2d {
        /// Output spatial size `[h, w]`.
        output_size: [usize; 2],
    },

    // -- Activation --
    /// Named activation function.
    Activation {
        /// Which activation.
        kind: TraceActivation,
    },
    /// Exponential linear unit.
    Elu {
        /// Alpha parameter.
        alpha: f64,
    },
    /// Leaky rectified linear unit.
    LeakyRelu {
        /// Negative slope.
        slope: f64,
    },
    /// Softplus: `log(1 + exp(x))`.
    Softplus,
    /// Scaled ELU.
    Selu,
    /// Continuous ELU.
    Celu {
        /// Alpha parameter.
        alpha: f64,
    },
    /// Mish: `x * tanh(softplus(x))`.
    Mish,
    /// Hard sigmoid.
    HardSigmoid,
    /// Hard swish.
    HardSwish,
    /// Softsign: `x / (1 + |x|)`.
    Softsign,
    /// Parametric ReLU.
    PRelu {
        /// Per-channel negative slope.
        slope: WeightPayload,
    },
    /// Kokoro-specific fused activation / normalization op.
    KokoroFused(KokoroFusedOp),
    /// SwiGLU gated feed-forward: `w_down(silu(w_gate(x)) * w_up(x))`.
    SwiGlu,
    /// Dropout (identity at inference).
    Dropout,

    // -- Vision --
    /// PixelShuffle: `[B, C*r², H, W] → [B, C, H*r, W*r]`.
    PixelShuffle {
        /// Upscale factor `r`.
        upscale_factor: usize,
    },
    /// PixelUnshuffle: `[B, C, H*r, W*r] → [B, C*r², H, W]`.
    PixelUnshuffle {
        /// Downscale factor `r`.
        downscale_factor: usize,
    },
    /// 1-D nearest-neighbor upsampling.
    Upsample1d {
        /// Integer scale factor.
        factor: usize,
    },
    /// 2-D upsampling (nearest or bilinear).
    Upsample2d {
        /// Interpolation mode.
        mode: TraceUpsampleMode,
        /// Height scale factor.
        scale_h: f64,
        /// Width scale factor.
        scale_w: f64,
    },
    /// Bilinear resize to absolute target dimensions.
    ResizeBilinear {
        /// Target height.
        target_h: usize,
        /// Target width.
        target_w: usize,
    },

    // -- Spatial mask / sampling --
    /// Upper-triangular mask.
    Triu {
        /// Diagonal offset.
        diagonal: i64,
    },
    /// Lower-triangular mask.
    Tril {
        /// Diagonal offset.
        diagonal: i64,
    },
    /// Bilinear grid sampling at arbitrary 2-D coordinates.
    GridSample {
        /// Out-of-bounds padding mode.
        padding_mode: GridSamplePaddingMode,
        /// Align-corners flag.
        align_corners: bool,
    },
    /// Quantized linear layer.
    QLinear {
        /// Quantized weight.
        weight: WeightPayload,
        /// Optional bias.
        bias: Option<WeightPayload>,
    },

    // -- Selection / indexing --
    /// Top-k values and indices along a dimension.
    Topk {
        /// Number of elements to select.
        k: usize,
        /// Dimension to select along.
        dim: usize,
    },
    /// Index of the maximum value along a dimension.
    Argmax {
        /// Dimension to reduce.
        dim: usize,
    },
    /// Index of the minimum value along a dimension.
    Argmin {
        /// Dimension to reduce.
        dim: usize,
    },
    /// Indices that would sort along a dimension.
    ArgSort {
        /// Dimension to sort along.
        dim: usize,
        /// Sort in descending order.
        descending: bool,
    },
    /// Sorted values (and indices) along a dimension.
    Sort {
        /// Dimension to sort along.
        dim: usize,
        /// Sort in descending order.
        descending: bool,
    },
    /// Select elements along `dim` using a 1-D index tensor.
    IndexSelect {
        /// Dimension to index.
        dim: usize,
    },
    /// Gather elements using an N-D index tensor along `dim`.
    Gather {
        /// Dimension to gather along.
        dim: usize,
    },
    /// Element-wise conditional select (ternary).
    WhereCond,
    /// Broadcast-expand a tensor to a larger shape.
    Expand {
        /// Target shape.
        target_shape: Vec<usize>,
    },
    /// Element-wise scalar comparison producing a mask.
    Compare {
        /// Comparison operator.
        op: CompareOp,
        /// Scalar compared against.
        value: f64,
    },
    /// Element-wise tensor-vs-tensor comparison producing a mask.
    CompareTensor {
        /// Comparison operator.
        op: CompareOp,
    },
    /// Cumulative sum along a dimension.
    Cumsum {
        /// Dimension to accumulate along.
        dim: usize,
    },
    /// Repeat each element along `dim` by variable counts.
    RepeatInterleave {
        /// Dimension to repeat along.
        dim: usize,
    },
    /// Element-wise power: `x^exponent`.
    Powf {
        /// Exponent.
        exponent: f64,
    },
    /// Convert a tensor to a different dtype.
    ToDtype {
        /// Target dtype.
        target_dtype: DType,
    },

    // -- Shape operations (extended) --
    /// Reverse elements along a dimension.
    Flip {
        /// Dimension to flip.
        dim: usize,
    },
    /// Circular shift along specified dimensions.
    Roll {
        /// Shift amounts.
        shifts: Vec<i64>,
        /// Dimensions to shift.
        dims: Vec<usize>,
    },
    /// Sliding-window extraction.
    Unfold {
        /// Dimension to unfold.
        dim: usize,
        /// Window size.
        size: usize,
        /// Window step.
        step: usize,
    },
    /// Write `src` into `self` at a slice along `dim` (KV-cache updates).
    SliceSet {
        /// Dimension written.
        dim: usize,
        /// Start index.
        start: usize,
    },
    /// Scatter `src` into `self` along `dim` using `index` (overwrite).
    Scatter {
        /// Scatter dimension.
        dim: usize,
    },
    /// Scatter-add `src` into `self` along `dim` using `index`.
    ScatterAdd {
        /// Scatter dimension.
        dim: usize,
    },
    /// Index-add `src` into `self` along `dim` using `index`.
    IndexAdd {
        /// Index dimension.
        dim: usize,
    },
    /// Non-mutating index-put: write `src` into `self` along `dim`.
    IndexPut {
        /// Index dimension.
        dim: usize,
    },
    /// Clamp values to a `[min, max]` range.
    Clamp {
        /// Lower bound (`None` = unbounded).
        min: Option<f64>,
        /// Upper bound (`None` = unbounded).
        max: Option<f64>,
    },
    /// Constant scalar (or filled tensor) injected during tracing.
    Constant {
        /// Scalar fill value.
        value: f64,
    },

    // -- Padding --
    /// 1-D reflection padding.
    ReflectionPad1d {
        /// Left pad.
        pad_left: usize,
        /// Right pad.
        pad_right: usize,
    },
    /// 2-D reflection padding.
    ReflectionPad2d {
        /// Left pad.
        pad_left: usize,
        /// Right pad.
        pad_right: usize,
        /// Top pad.
        pad_top: usize,
        /// Bottom pad.
        pad_bottom: usize,
    },
    /// N-D constant padding.
    ConstantPadNd {
        /// Per-dimension padding amounts.
        padding: Vec<usize>,
        /// Constant fill value.
        value: f64,
    },

    /// Two-argument arctangent: `atan2(y, x)` (inputs `y, x`).
    Atan2,

    // -- Tensor creation --
    /// Monotonic range `[start, end)` with `step`.
    Arange {
        /// Range start.
        start: f64,
        /// Range end (exclusive).
        end: f64,
        /// Step size.
        step: f64,
    },

    /// Pipeline segment boundary marker for data-dependent ops.
    SegmentBoundary {
        /// Human-readable reason (e.g. `"length_regulate"`).
        reason: String,
        /// Optional `(lower, upper)` bounds hint for the segment output.
        input_bounds: Option<(f32, f32)>,
    },

    /// MoE gating: softmax + top-k expert routing.
    MoeGating {
        /// Total number of experts.
        num_experts: usize,
        /// Number of experts selected per token.
        top_k: usize,
    },

    /// Custom / unknown operation (extensibility escape hatch).
    Custom {
        /// Operation name.
        name: String,
    },
}

// -- Graph types --------------------------------------------------------------

/// A node in a traced computation graph.
///
/// Mirrors `nn-core`'s `TraceNode`. Unlike the source — which keeps its fields
/// `pub(super)` behind accessors — these are `pub` because this type *is* the
/// wire format; consumers build it directly when serializing their trace.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TraceNode {
    /// Unique node identifier.
    pub id: NodeId,
    /// Human-readable name (e.g. `"linear_0"`).
    pub name: String,
    /// The operation this node performs.
    pub op: TraceOp,
    /// IDs of input nodes (must reference earlier nodes; see
    /// [`ComputationGraph::validate_topology`]).
    pub inputs: Vec<NodeId>,
    /// Output tensor shape.
    pub output_shape: Vec<usize>,
    /// Output tensor dtype.
    pub output_dtype: DType,
}

impl TraceNode {
    /// Construct a trace node from explicit fields.
    #[must_use]
    pub fn new(
        id: NodeId,
        name: impl Into<String>,
        op: TraceOp,
        inputs: Vec<NodeId>,
        output_shape: Vec<usize>,
        output_dtype: DType,
    ) -> Self {
        Self {
            id,
            name: name.into(),
            op,
            inputs,
            output_shape,
            output_dtype,
        }
    }
}

/// Topology validation error returned by [`ComputationGraph::validate_topology`].
///
/// Mirrors the failure case of `nn-core`'s `ComputationGraph::validate_topology`
/// (which returns the framework's `TensorError::TopologyError`). Owned here so
/// the schema has no framework dependency.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TopologyError {
    /// Name of the offending node.
    pub node_name: String,
    /// Index of the offending node in `nodes`.
    pub index: usize,
    /// The input ID that was referenced before it was defined.
    pub missing_input: NodeId,
}

impl std::fmt::Display for TopologyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "node '{}' at index {} references undefined input {}",
            self.node_name, self.index, self.missing_input.0
        )
    }
}

impl std::error::Error for TopologyError {}

/// A captured computation graph.
///
/// Mirrors `nn-core`'s `ComputationGraph`: nodes in topological order plus the
/// list of output node IDs. The `id → index` map the source caches is omitted
/// from the wire format (it is derivable) and rebuilt on demand by accessors.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComputationGraph {
    /// Nodes in topological (dependency-before-dependent) order.
    pub nodes: Vec<TraceNode>,
    /// Output node IDs, in the order they were marked. Single-output graphs
    /// have exactly one entry.
    pub output_nodes: Vec<NodeId>,
}

impl ComputationGraph {
    /// Build a graph from a pre-ordered slice of nodes.
    ///
    /// The last node (if any) becomes the sole output, matching
    /// `nn-core::ComputationGraph::from_nodes`. Callers must ensure topological
    /// ordering (validate with [`Self::validate_topology`]).
    #[must_use]
    pub fn from_nodes(nodes: Vec<TraceNode>) -> Self {
        let output_nodes = nodes.last().map(|n| n.id).into_iter().collect();
        Self {
            nodes,
            output_nodes,
        }
    }

    /// Returns the number of nodes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns `true` if the graph has no nodes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Returns a node by ID (linear scan; the wire format has no index map).
    #[must_use]
    pub fn node(&self, id: NodeId) -> Option<&TraceNode> {
        self.nodes.iter().find(|n| n.id == id)
    }

    /// Returns the primary (last-marked) output node, if any.
    #[must_use]
    pub fn output_node(&self) -> Option<&TraceNode> {
        self.output_nodes.last().and_then(|&id| self.node(id))
    }

    /// Returns all nodes whose op is [`TraceOp::Input`].
    #[must_use]
    pub fn input_nodes(&self) -> Vec<&TraceNode> {
        self.nodes
            .iter()
            .filter(|n| matches!(n.op, TraceOp::Input))
            .collect()
    }

    /// Returns indices of nodes that are [`TraceOp::SegmentBoundary`] markers.
    #[must_use]
    pub fn segment_boundaries(&self) -> Vec<usize> {
        self.nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| matches!(n.op, TraceOp::SegmentBoundary { .. }))
            .map(|(i, _)| i)
            .collect()
    }

    /// Returns `true` if the graph contains any [`TraceOp::SegmentBoundary`].
    #[must_use]
    pub fn has_segment_boundaries(&self) -> bool {
        self.nodes
            .iter()
            .any(|n| matches!(n.op, TraceOp::SegmentBoundary { .. }))
    }

    /// Validate that every node's inputs reference earlier nodes.
    ///
    /// Returns the first out-of-order reference as a [`TopologyError`]. Mirrors
    /// `nn-core::ComputationGraph::validate_topology`.
    ///
    /// # Errors
    ///
    /// Returns [`TopologyError`] if a node references an input ID that has not
    /// been defined by an earlier node.
    pub fn validate_topology(&self) -> Result<(), TopologyError> {
        use std::collections::HashSet;
        let mut seen: HashSet<NodeId> = HashSet::with_capacity(self.nodes.len());
        for (index, node) in self.nodes.iter().enumerate() {
            for &input_id in &node.inputs {
                if !seen.contains(&input_id) {
                    return Err(TopologyError {
                        node_name: node.name.clone(),
                        index,
                        missing_input: input_id,
                    });
                }
            }
            seen.insert(node.id);
        }
        Ok(())
    }

    /// Split this graph at [`TraceOp::SegmentBoundary`] markers.
    ///
    /// Each segment is a self-contained [`ComputationGraph`] (the boundary node
    /// itself is dropped). With no boundaries, returns the whole graph as a
    /// single segment. Mirrors
    /// `nn-core::ComputationGraph::split_at_segment_boundaries`.
    #[must_use]
    pub fn split_at_segment_boundaries(&self) -> SegmentedGraph {
        let boundary_indices = self.segment_boundaries();
        if boundary_indices.is_empty() {
            return SegmentedGraph {
                segments: vec![GraphSegment {
                    graph: self.clone(),
                    boundary_reason: None,
                    boundary_bounds: None,
                }],
            };
        }

        let mut segments = Vec::new();
        let mut seg_start = 0;
        for &boundary_idx in &boundary_indices {
            let seg_nodes = self.nodes[seg_start..boundary_idx].to_vec();
            let (reason, bounds) = match &self.nodes[boundary_idx].op {
                TraceOp::SegmentBoundary {
                    reason,
                    input_bounds,
                } => (Some(reason.clone()), *input_bounds),
                _ => (None, None),
            };
            if !seg_nodes.is_empty() {
                segments.push(GraphSegment {
                    graph: Self::from_nodes(seg_nodes),
                    boundary_reason: reason,
                    boundary_bounds: bounds,
                });
            }
            seg_start = boundary_idx + 1;
        }
        if seg_start < self.nodes.len() {
            let seg_nodes = self.nodes[seg_start..].to_vec();
            if !seg_nodes.is_empty() {
                segments.push(GraphSegment {
                    graph: Self::from_nodes(seg_nodes),
                    boundary_reason: None,
                    boundary_bounds: None,
                });
            }
        }
        SegmentedGraph { segments }
    }
}

/// A single segment of a computation graph split at boundary markers.
///
/// Mirrors `nn-core`'s `GraphSegment`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphSegment {
    /// The sub-graph for this segment.
    pub graph: ComputationGraph,
    /// Reason for the preceding boundary (`None` for the first/tail segment).
    pub boundary_reason: Option<String>,
    /// Optional `(lower, upper)` bounds hint from the preceding boundary.
    pub boundary_bounds: Option<(f32, f32)>,
}

/// A computation graph split at data-dependent operation boundaries.
///
/// Mirrors `nn-core`'s `SegmentedGraph`. Output bounds from segment `N` feed as
/// input bounds to segment `N+1` during verification.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SegmentedGraph {
    /// Segments in order; the first starts with the original inputs.
    pub segments: Vec<GraphSegment>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The number of `TraceOp` variants mirrored from `nn-core`.
    ///
    /// Pinned so a source-side addition that is not mirrored here fails the
    /// build. Keep in sync with [`all_trace_ops`].
    const EXPECTED_TRACE_OP_VARIANTS: usize = 123;

    /// One sample value of every `TraceOp` variant.
    ///
    /// This is the exhaustiveness witness: the compiler forces the `match` in
    /// [`trace_op_discriminant`] to cover every variant, and the list length
    /// pins the count.
    fn all_trace_ops() -> Vec<TraceOp> {
        let w = || WeightPayload::f32(vec![0.0], vec![1]);
        let ow = || Some(WeightPayload::f32(vec![0.0], vec![1]));
        vec![
            TraceOp::Input,
            TraceOp::ConstantWeight { weight: w() },
            TraceOp::Add,
            TraceOp::Sub,
            TraceOp::Mul,
            TraceOp::Div,
            TraceOp::Maximum,
            TraceOp::Minimum,
            TraceOp::MatMul,
            TraceOp::Relu,
            TraceOp::Gelu,
            TraceOp::GeluErf,
            TraceOp::Silu,
            TraceOp::Tanh,
            TraceOp::Sigmoid,
            TraceOp::Exp,
            TraceOp::Log,
            TraceOp::Sqrt,
            TraceOp::Sqr,
            TraceOp::Abs,
            TraceOp::Neg,
            TraceOp::Recip,
            TraceOp::Sin,
            TraceOp::Cos,
            TraceOp::Tan,
            TraceOp::Floor,
            TraceOp::Ceil,
            TraceOp::Round,
            TraceOp::Sign,
            TraceOp::Fract,
            TraceOp::ReduceSum {
                dim: 0,
                keepdim: false,
            },
            TraceOp::ReduceMean {
                dim: 0,
                keepdim: false,
            },
            TraceOp::ReduceMax {
                dim: 0,
                keepdim: false,
            },
            TraceOp::ReduceMin {
                dim: 0,
                keepdim: false,
            },
            TraceOp::Reshape {
                target_shape: vec![1],
            },
            TraceOp::Transpose { dim0: 0, dim1: 1 },
            TraceOp::Narrow {
                dim: 0,
                start: 0,
                length: 1,
            },
            TraceOp::Unsqueeze { dim: 0 },
            TraceOp::Squeeze { dim: 0 },
            TraceOp::Permute { axes: vec![0] },
            TraceOp::Cat {
                dim: 0,
                num_inputs: 2,
            },
            TraceOp::LayerNorm {
                eps: 1e-5,
                weight: w(),
                bias: w(),
            },
            TraceOp::RmsNorm {
                eps: 1e-5,
                weight: w(),
            },
            TraceOp::GroupNorm {
                num_groups: 1,
                eps: 1e-5,
                weight: w(),
                bias: w(),
            },
            TraceOp::InstanceNorm { eps: 1e-5 },
            TraceOp::BatchNorm {
                eps: 1e-5,
                weight: w(),
                bias: w(),
                running_mean: w(),
                running_var: w(),
            },
            TraceOp::Linear {
                weight: w(),
                bias: ow(),
            },
            TraceOp::Conv1d {
                weight: w(),
                bias: ow(),
                padding: 0,
                stride: 1,
                dilation: 1,
                groups: 1,
            },
            TraceOp::Conv2d {
                weight: w(),
                bias: ow(),
                padding: [0, 0],
                stride: [1, 1],
                dilation: [1, 1],
                groups: 1,
            },
            TraceOp::Conv3d {
                weight: w(),
                bias: ow(),
                padding: [0, 0, 0],
                stride: [1, 1, 1],
                dilation: [1, 1, 1],
                groups: 1,
            },
            TraceOp::ConvTranspose1d {
                weight: w(),
                bias: ow(),
                padding: 0,
                output_padding: 0,
                stride: 1,
                dilation: 1,
                groups: 1,
            },
            TraceOp::ConvTranspose2d {
                weight: w(),
                bias: ow(),
                padding: [0, 0],
                output_padding: [0, 0],
                stride: [1, 1],
                dilation: [1, 1],
                groups: 1,
            },
            TraceOp::Softmax { dim: 0 },
            TraceOp::LogSoftmax { dim: 0 },
            TraceOp::Sdpa { scale: 1.0 },
            TraceOp::SdpaCausal { scale: 1.0 },
            TraceOp::RotaryEmbedding {
                head_dim: 8,
                offset: 0,
                cos_cache: w(),
                sin_cache: w(),
            },
            TraceOp::MultiHeadAttention {
                num_heads: 1,
                num_kv_heads: 1,
                head_dim: 8,
            },
            TraceOp::Embedding { weight: w() },
            TraceOp::Lstm {
                weight_ih: w(),
                weight_hh: w(),
                bias_ih: ow(),
                bias_hh: ow(),
                hidden_size: 1,
                initial_hidden: None,
                initial_cell: None,
            },
            TraceOp::MaxPool1d {
                kernel_size: 1,
                stride: 1,
                padding: 0,
            },
            TraceOp::AvgPool2d {
                kernel_size: [1, 1],
                stride: [1, 1],
                padding: [0, 0],
            },
            TraceOp::MaxPool2d {
                kernel_size: [1, 1],
                stride: [1, 1],
                padding: [0, 0],
            },
            TraceOp::AdaptiveAvgPool2d {
                output_size: [1, 1],
            },
            TraceOp::AvgPool1d {
                kernel_size: 1,
                stride: 1,
                padding: 0,
            },
            TraceOp::AdaptiveAvgPool1d { output_size: 1 },
            TraceOp::AdaptiveMaxPool2d {
                output_size: [1, 1],
            },
            TraceOp::Activation {
                kind: TraceActivation::Relu,
            },
            TraceOp::Elu { alpha: 1.0 },
            TraceOp::LeakyRelu { slope: 0.2 },
            TraceOp::Softplus,
            TraceOp::Selu,
            TraceOp::Celu { alpha: 1.0 },
            TraceOp::Mish,
            TraceOp::HardSigmoid,
            TraceOp::HardSwish,
            TraceOp::Softsign,
            TraceOp::PRelu { slope: w() },
            TraceOp::KokoroFused(KokoroFusedOp::SnakeTensor { alpha: w() }),
            TraceOp::SwiGlu,
            TraceOp::Dropout,
            TraceOp::PixelShuffle { upscale_factor: 2 },
            TraceOp::PixelUnshuffle {
                downscale_factor: 2,
            },
            TraceOp::Upsample1d { factor: 2 },
            TraceOp::Upsample2d {
                mode: TraceUpsampleMode::Nearest,
                scale_h: 2.0,
                scale_w: 2.0,
            },
            TraceOp::ResizeBilinear {
                target_h: 4,
                target_w: 4,
            },
            TraceOp::Triu { diagonal: 0 },
            TraceOp::Tril { diagonal: 0 },
            TraceOp::GridSample {
                padding_mode: GridSamplePaddingMode::Zeros,
                align_corners: false,
            },
            TraceOp::QLinear {
                weight: w(),
                bias: ow(),
            },
            TraceOp::Topk { k: 1, dim: 0 },
            TraceOp::Argmax { dim: 0 },
            TraceOp::Argmin { dim: 0 },
            TraceOp::ArgSort {
                dim: 0,
                descending: false,
            },
            TraceOp::Sort {
                dim: 0,
                descending: false,
            },
            TraceOp::IndexSelect { dim: 0 },
            TraceOp::Gather { dim: 0 },
            TraceOp::WhereCond,
            TraceOp::Expand {
                target_shape: vec![1],
            },
            TraceOp::Compare {
                op: CompareOp::Eq,
                value: 0.0,
            },
            TraceOp::CompareTensor { op: CompareOp::Eq },
            TraceOp::Cumsum { dim: 0 },
            TraceOp::RepeatInterleave { dim: 0 },
            TraceOp::Powf { exponent: 2.0 },
            TraceOp::ToDtype {
                target_dtype: DType::F32,
            },
            TraceOp::Flip { dim: 0 },
            TraceOp::Roll {
                shifts: vec![1],
                dims: vec![0],
            },
            TraceOp::Unfold {
                dim: 0,
                size: 1,
                step: 1,
            },
            TraceOp::SliceSet { dim: 0, start: 0 },
            TraceOp::Scatter { dim: 0 },
            TraceOp::ScatterAdd { dim: 0 },
            TraceOp::IndexAdd { dim: 0 },
            TraceOp::IndexPut { dim: 0 },
            TraceOp::Clamp {
                min: Some(0.0),
                max: Some(1.0),
            },
            TraceOp::Constant { value: 0.0 },
            TraceOp::ReflectionPad1d {
                pad_left: 1,
                pad_right: 1,
            },
            TraceOp::ReflectionPad2d {
                pad_left: 1,
                pad_right: 1,
                pad_top: 1,
                pad_bottom: 1,
            },
            TraceOp::ConstantPadNd {
                padding: vec![1, 1],
                value: 0.0,
            },
            TraceOp::Atan2,
            TraceOp::Arange {
                start: 0.0,
                end: 1.0,
                step: 1.0,
            },
            TraceOp::SegmentBoundary {
                reason: "x".into(),
                input_bounds: None,
            },
            TraceOp::MoeGating {
                num_experts: 1,
                top_k: 1,
            },
            TraceOp::Custom { name: "x".into() },
        ]
    }

    /// Exhaustive discriminant index over every `TraceOp` variant.
    ///
    /// The `match` is total (no wildcard arm), so adding a source variant to
    /// the enum without updating this list is a compile error — keeping the
    /// mirror honest.
    fn trace_op_discriminant(op: &TraceOp) -> usize {
        match op {
            TraceOp::Input => 0,
            TraceOp::ConstantWeight { .. } => 1,
            TraceOp::Add => 2,
            TraceOp::Sub => 3,
            TraceOp::Mul => 4,
            TraceOp::Div => 5,
            TraceOp::Maximum => 6,
            TraceOp::Minimum => 7,
            TraceOp::MatMul => 8,
            TraceOp::Relu => 9,
            TraceOp::Gelu => 10,
            TraceOp::GeluErf => 11,
            TraceOp::Silu => 12,
            TraceOp::Tanh => 13,
            TraceOp::Sigmoid => 14,
            TraceOp::Exp => 15,
            TraceOp::Log => 16,
            TraceOp::Sqrt => 17,
            TraceOp::Sqr => 18,
            TraceOp::Abs => 19,
            TraceOp::Neg => 20,
            TraceOp::Recip => 21,
            TraceOp::Sin => 22,
            TraceOp::Cos => 23,
            TraceOp::Tan => 24,
            TraceOp::Floor => 25,
            TraceOp::Ceil => 26,
            TraceOp::Round => 27,
            TraceOp::Sign => 28,
            TraceOp::Fract => 29,
            TraceOp::ReduceSum { .. } => 30,
            TraceOp::ReduceMean { .. } => 31,
            TraceOp::ReduceMax { .. } => 32,
            TraceOp::ReduceMin { .. } => 33,
            TraceOp::Reshape { .. } => 34,
            TraceOp::Transpose { .. } => 35,
            TraceOp::Narrow { .. } => 36,
            TraceOp::Unsqueeze { .. } => 37,
            TraceOp::Squeeze { .. } => 38,
            TraceOp::Permute { .. } => 39,
            TraceOp::Cat { .. } => 40,
            TraceOp::LayerNorm { .. } => 41,
            TraceOp::RmsNorm { .. } => 42,
            TraceOp::GroupNorm { .. } => 43,
            TraceOp::InstanceNorm { .. } => 44,
            TraceOp::BatchNorm { .. } => 45,
            TraceOp::Linear { .. } => 46,
            TraceOp::Conv1d { .. } => 47,
            TraceOp::Conv2d { .. } => 48,
            TraceOp::Conv3d { .. } => 49,
            TraceOp::ConvTranspose1d { .. } => 50,
            TraceOp::ConvTranspose2d { .. } => 51,
            TraceOp::Softmax { .. } => 52,
            TraceOp::LogSoftmax { .. } => 53,
            TraceOp::Sdpa { .. } => 54,
            TraceOp::SdpaCausal { .. } => 55,
            TraceOp::RotaryEmbedding { .. } => 56,
            TraceOp::MultiHeadAttention { .. } => 57,
            TraceOp::Embedding { .. } => 58,
            TraceOp::Lstm { .. } => 59,
            TraceOp::MaxPool1d { .. } => 60,
            TraceOp::AvgPool2d { .. } => 61,
            TraceOp::MaxPool2d { .. } => 62,
            TraceOp::AdaptiveAvgPool2d { .. } => 63,
            TraceOp::AvgPool1d { .. } => 64,
            TraceOp::AdaptiveAvgPool1d { .. } => 65,
            TraceOp::AdaptiveMaxPool2d { .. } => 66,
            TraceOp::Activation { .. } => 67,
            TraceOp::Elu { .. } => 68,
            TraceOp::LeakyRelu { .. } => 69,
            TraceOp::Softplus => 70,
            TraceOp::Selu => 71,
            TraceOp::Celu { .. } => 72,
            TraceOp::Mish => 73,
            TraceOp::HardSigmoid => 74,
            TraceOp::HardSwish => 75,
            TraceOp::Softsign => 76,
            TraceOp::PRelu { .. } => 77,
            TraceOp::KokoroFused(_) => 78,
            TraceOp::SwiGlu => 79,
            TraceOp::Dropout => 80,
            TraceOp::PixelShuffle { .. } => 81,
            TraceOp::PixelUnshuffle { .. } => 82,
            TraceOp::Upsample1d { .. } => 83,
            TraceOp::Upsample2d { .. } => 84,
            TraceOp::ResizeBilinear { .. } => 85,
            TraceOp::Triu { .. } => 86,
            TraceOp::Tril { .. } => 87,
            TraceOp::GridSample { .. } => 88,
            TraceOp::QLinear { .. } => 89,
            TraceOp::Topk { .. } => 90,
            TraceOp::Argmax { .. } => 91,
            TraceOp::Argmin { .. } => 92,
            TraceOp::ArgSort { .. } => 93,
            TraceOp::Sort { .. } => 94,
            TraceOp::IndexSelect { .. } => 95,
            TraceOp::Gather { .. } => 96,
            TraceOp::WhereCond => 97,
            TraceOp::Expand { .. } => 98,
            TraceOp::Compare { .. } => 99,
            TraceOp::CompareTensor { .. } => 100,
            TraceOp::Cumsum { .. } => 101,
            TraceOp::RepeatInterleave { .. } => 102,
            TraceOp::Powf { .. } => 103,
            TraceOp::ToDtype { .. } => 104,
            TraceOp::Flip { .. } => 105,
            TraceOp::Roll { .. } => 106,
            TraceOp::Unfold { .. } => 107,
            TraceOp::SliceSet { .. } => 108,
            TraceOp::Scatter { .. } => 109,
            TraceOp::ScatterAdd { .. } => 110,
            TraceOp::IndexAdd { .. } => 111,
            TraceOp::IndexPut { .. } => 112,
            TraceOp::Clamp { .. } => 113,
            TraceOp::Constant { .. } => 114,
            TraceOp::ReflectionPad1d { .. } => 115,
            TraceOp::ReflectionPad2d { .. } => 116,
            TraceOp::ConstantPadNd { .. } => 117,
            TraceOp::Atan2 => 118,
            TraceOp::Arange { .. } => 119,
            TraceOp::SegmentBoundary { .. } => 120,
            TraceOp::MoeGating { .. } => 121,
            TraceOp::Custom { .. } => 122,
        }
    }

    #[test]
    fn trace_op_variant_count_matches_source() {
        assert_eq!(all_trace_ops().len(), EXPECTED_TRACE_OP_VARIANTS);
    }

    #[test]
    fn trace_op_discriminants_are_unique_and_dense() {
        use std::collections::BTreeSet;
        let discs: Vec<usize> = all_trace_ops().iter().map(trace_op_discriminant).collect();
        let unique: BTreeSet<usize> = discs.iter().copied().collect();
        assert_eq!(
            unique.len(),
            EXPECTED_TRACE_OP_VARIANTS,
            "every TraceOp sample must have a distinct discriminant"
        );
        // Dense range 0..N: confirms the witness list and the match agree.
        assert_eq!(*unique.iter().next().unwrap(), 0);
        assert_eq!(
            *unique.iter().next_back().unwrap(),
            EXPECTED_TRACE_OP_VARIANTS - 1
        );
    }

    #[test]
    fn every_trace_op_round_trips_json() {
        for op in all_trace_ops() {
            let json = serde_json::to_string(&op).expect("serialize TraceOp");
            let back: TraceOp = serde_json::from_str(&json).expect("deserialize TraceOp");
            assert_eq!(op, back, "round-trip mismatch for {op:?}");
        }
    }

    #[test]
    fn dtype_round_trips_and_metadata() {
        let all = [
            DType::F32,
            DType::F16,
            DType::Bf16,
            DType::F64,
            DType::I32,
            DType::I64,
            DType::U32,
            DType::U8,
            DType::Bool,
        ];
        for dt in all {
            let json = serde_json::to_string(&dt).unwrap();
            let back: DType = serde_json::from_str(&json).unwrap();
            assert_eq!(dt, back);
            assert!(dt.size_bytes() > 0);
        }
        assert!(DType::F32.is_float());
        assert!(!DType::F32.is_int());
        assert!(DType::I64.is_int());
        assert!(!DType::Bool.is_float() && !DType::Bool.is_int());
    }

    #[test]
    fn weight_payload_dtypes_round_trip() {
        let payloads = vec![
            WeightPayload::f32(vec![1.0, 2.0], vec![2]),
            WeightPayload {
                shape: vec![2],
                data: WeightData::F64(vec![1.0, 2.0]),
            },
            WeightPayload {
                shape: vec![2],
                data: WeightData::F16(vec![half::f16::from_f32(1.5), half::f16::from_f32(2.5)]),
            },
            WeightPayload {
                shape: vec![2],
                data: WeightData::Bf16(vec![half::bf16::from_f32(1.5), half::bf16::from_f32(2.5)]),
            },
            WeightPayload {
                shape: vec![2],
                data: WeightData::I32(vec![1, 2]),
            },
            WeightPayload {
                shape: vec![2],
                data: WeightData::I64(vec![1, 2]),
            },
            WeightPayload::placeholder(vec![3, 4]),
        ];
        for p in payloads {
            let json = serde_json::to_string(&p).unwrap();
            let back: WeightPayload = serde_json::from_str(&json).unwrap();
            assert_eq!(p, back);
        }
        assert!(WeightPayload::placeholder(vec![3, 4]).is_placeholder());
        assert!(!WeightPayload::f32(vec![0.0], vec![1]).is_placeholder());
        assert_eq!(WeightPayload::placeholder(vec![3, 4]).numel(), 12);
    }

    #[test]
    fn node_id_serializes_transparently() {
        let id = NodeId(42);
        assert_eq!(serde_json::to_string(&id).unwrap(), "42");
        let back: NodeId = serde_json::from_str("42").unwrap();
        assert_eq!(id, back);
        assert_eq!(id.get(), 42);
        assert_eq!(u64::from(id), 42);
        assert_eq!(NodeId::from(7u64), NodeId(7));
    }

    /// Hand-build a small graph and assert full JSON round-trip and topology.
    #[test]
    fn computation_graph_round_trips_json() {
        let nodes = vec![
            TraceNode::new(
                NodeId(0),
                "x",
                TraceOp::Input,
                vec![],
                vec![1, 4],
                DType::F32,
            ),
            TraceNode::new(
                NodeId(1),
                "fc",
                TraceOp::Linear {
                    weight: WeightPayload::f32(vec![0.0; 8], vec![2, 4]),
                    bias: Some(WeightPayload::f32(vec![0.0, 0.0], vec![2])),
                },
                vec![NodeId(0)],
                vec![1, 2],
                DType::F32,
            ),
            TraceNode::new(
                NodeId(2),
                "act",
                TraceOp::Relu,
                vec![NodeId(1)],
                vec![1, 2],
                DType::F32,
            ),
        ];
        let graph = ComputationGraph::from_nodes(nodes);
        assert_eq!(graph.len(), 3);
        assert!(!graph.is_empty());
        assert_eq!(graph.output_nodes, vec![NodeId(2)]);
        assert_eq!(graph.output_node().unwrap().name, "act");
        assert_eq!(graph.input_nodes().len(), 1);
        graph.validate_topology().expect("graph is well-ordered");

        let json = serde_json::to_string_pretty(&graph).unwrap();
        let back: ComputationGraph = serde_json::from_str(&json).unwrap();
        assert_eq!(graph, back);
        // Lookup by id survives the round trip.
        assert_eq!(back.node(NodeId(1)).unwrap().name, "fc");
    }

    #[test]
    fn topology_validation_rejects_forward_reference() {
        let nodes = vec![TraceNode::new(
            NodeId(0),
            "bad",
            TraceOp::Relu,
            vec![NodeId(99)],
            vec![1],
            DType::F32,
        )];
        let graph = ComputationGraph::from_nodes(nodes);
        let err = graph.validate_topology().unwrap_err();
        assert_eq!(err.missing_input, NodeId(99));
        assert_eq!(err.index, 0);
        assert_eq!(err.node_name, "bad");
        assert!(err.to_string().contains("undefined input 99"));
    }

    #[test]
    fn segment_split_no_boundary_yields_single_segment() {
        let nodes = vec![
            TraceNode::new(NodeId(0), "x", TraceOp::Input, vec![], vec![1], DType::F32),
            TraceNode::new(
                NodeId(1),
                "r",
                TraceOp::Relu,
                vec![NodeId(0)],
                vec![1],
                DType::F32,
            ),
        ];
        let graph = ComputationGraph::from_nodes(nodes);
        assert!(!graph.has_segment_boundaries());
        let segmented = graph.split_at_segment_boundaries();
        assert_eq!(segmented.segments.len(), 1);
        assert_eq!(segmented.segments[0].graph.len(), 2);
    }

    #[test]
    fn segment_split_at_boundary_drops_marker() {
        let nodes = vec![
            TraceNode::new(NodeId(0), "x", TraceOp::Input, vec![], vec![1], DType::F32),
            TraceNode::new(
                NodeId(1),
                "boundary",
                TraceOp::SegmentBoundary {
                    reason: "length_regulate".into(),
                    input_bounds: Some((-1.0, 1.0)),
                },
                vec![NodeId(0)],
                vec![1],
                DType::F32,
            ),
            TraceNode::new(
                NodeId(2),
                "r",
                TraceOp::Relu,
                vec![NodeId(1)],
                vec![1],
                DType::F32,
            ),
        ];
        let graph = ComputationGraph::from_nodes(nodes);
        assert!(graph.has_segment_boundaries());
        assert_eq!(graph.segment_boundaries(), vec![1]);
        let segmented = graph.split_at_segment_boundaries();
        assert_eq!(segmented.segments.len(), 2);
        // Mirrors nn-core: the boundary's metadata is attached to the segment
        // *built at* that boundary index — i.e. the nodes preceding it. The
        // trailing segment (nodes after the last boundary) carries None.
        assert_eq!(segmented.segments[0].graph.len(), 1);
        assert_eq!(
            segmented.segments[0].boundary_reason.as_deref(),
            Some("length_regulate")
        );
        assert_eq!(segmented.segments[0].boundary_bounds, Some((-1.0, 1.0)));
        // Trailing segment: nodes after the boundary, no attached metadata.
        assert_eq!(segmented.segments[1].graph.len(), 1);
        assert_eq!(segmented.segments[1].boundary_reason, None);
        // The boundary marker node itself is dropped from both segments.
        let total_nodes: usize = segmented.segments.iter().map(|s| s.graph.len()).sum();
        assert_eq!(total_nodes, 2);

        // SegmentedGraph round-trips as JSON too.
        let json = serde_json::to_string(&segmented).unwrap();
        let back: SegmentedGraph = serde_json::from_str(&json).unwrap();
        assert_eq!(segmented, back);
    }
}
