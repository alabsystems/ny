// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Build-time soundness-coverage gate for the trace bridge.
//!
//! This module is the single NY-owned source of truth for *how soundly* each
//! [`TraceOp`] *could* be lowered into a verifier graph — a bound-propagation
//! soundness taxonomy, not an implementation-status registry for
//! [`crate::translate`]. It exists so the never-translatable op set is
//! **explicit and compiler-checked** rather than discovered at runtime:
//! [`trace_op_soundness`] is an exhaustive `match` over all **123** `TraceOp`
//! variants with *no wildcard arm*, so adding a variant to
//! [`crate::schema::TraceOp`] later fails to compile here until it is
//! deliberately classified.
//!
//! [`crate::translate`] implements a narrower core set than the
//! non-[`Unsupported`](BridgeSoundness::Unsupported) taxonomy: any
//! classified-but-unimplemented op is refused with an explicit
//! `UnsupportedOp` error (fail-closed), never lowered unsoundly. Consult
//! `translate` itself, not this taxonomy, for current translator coverage.
//!
//! ## Classification
//!
//! Each op is assigned a [`BridgeSoundness`] by its bound-propagation behaviour:
//!
//! - [`BridgeSoundness::Exact`] — the bridge reproduces the op's semantics
//!   exactly on interval bounds (pure reshape / reindex / sign-aware affine; no
//!   over-approximation is introduced).
//! - [`BridgeSoundness::Sound`] — bounds are a *correct* over-approximation that
//!   is tight in practice (monotone activations, ReLU-family, clamps).
//! - [`BridgeSoundness::SoundButLoose`] — bounds remain a correct
//!   over-approximation but are *known to be loose*: normalisation, softmax,
//!   attention, cross-element reductions, and transcendentals
//!   (`exp`/`log`/`sqrt`/`div`/resize/grid-sample) whose interval relaxations
//!   widen quickly. Still safe to verify with; just weaker. The extreme case
//!   is `Custom`, the trace format's explicit opaque escape hatch: lowered to
//!   the graph builder's `OpaqueSkip` substitution, whose `[-inf, +inf]`
//!   bounds are a *vacuous but genuinely sound* over-approximation of any op
//!   (never an identity) — verification succeeds only if nothing downstream
//!   depends on that value.
//! - [`BridgeSoundness::Unsupported`] — the op's output *shape or routing*
//!   depends on tensor **values** (`Topk`, `Argmax`/`Argmin`, `ArgSort`,
//!   `Scatter*`, `IndexPut`, `MoeGating`, value comparisons, `WhereCond`).
//!   The bridge must **refuse** these rather than emit a vacuous/incorrect
//!   layer (sound by construction); they are handled by graph segmentation
//!   upstream, not by direct translation.
//!
//! [`coverage`] reports the four buckets over a canonical const list of every
//! variant's name, and the `tests` module asserts that list partitions all 123
//! variants — pinning the catalogue against silent drift from the enum.

use serde::{Deserialize, Serialize};

use crate::schema::{KokoroFusedOp, TraceOp};

/// How soundly the bridge could lower a given [`TraceOp`] into a verifier
/// graph.
///
/// See the [module docs](self) for the precise meaning of each level. The
/// ordering is from most to least faithful; [`Self::Unsupported`] is the only
/// level for which [`is_translatable`] returns `false`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BridgeSoundness {
    /// Bounds reproduce the op exactly (reshape / reindex / sign-aware affine).
    Exact,
    /// Bounds are a correct, practically tight over-approximation.
    Sound,
    /// Bounds are a correct but knowingly loose over-approximation.
    SoundButLoose,
    /// Output shape/routing is data-dependent (or opaque); cannot be translated.
    Unsupported,
}

impl BridgeSoundness {
    /// Returns `true` for every level except [`Self::Unsupported`].
    ///
    /// `true` means a sound lowering *exists in principle*; it does not mean
    /// [`crate::translate`] implements one today. `translate` additionally
    /// refuses any op outside its implemented core set (fail-closed), whereas
    /// [`Self::Unsupported`] ops must *always* be refused.
    #[must_use]
    pub const fn is_translatable(self) -> bool {
        !matches!(self, Self::Unsupported)
    }

    /// Short, stable lowercase label (matches the serde representation).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Exact => "exact",
            Self::Sound => "sound",
            Self::SoundButLoose => "sound_but_loose",
            Self::Unsupported => "unsupported",
        }
    }
}

/// Classify a single [`TraceOp`] by its bridge soundness.
///
/// This is an **exhaustive** match over all 123 `TraceOp` variants with no
/// wildcard arm: introducing a new variant in [`crate::schema::TraceOp`] is a
/// compile error here until it is explicitly classified. That is the whole point
/// of the gate — the unsupported / lossy set can never silently grow.
#[must_use]
pub fn trace_op_soundness(op: &TraceOp) -> BridgeSoundness {
    use BridgeSoundness::{Exact, Sound, SoundButLoose, Unsupported};
    match op {
        // -- Exact: pure structural / reindex / identity ----------------------
        // No values change and no over-approximation is introduced.
        TraceOp::Input
        | TraceOp::ConstantWeight { .. }
        | TraceOp::Constant { .. }
        | TraceOp::Reshape { .. }
        | TraceOp::Transpose { .. }
        | TraceOp::Narrow { .. }
        | TraceOp::Unsqueeze { .. }
        | TraceOp::Squeeze { .. }
        | TraceOp::Permute { .. }
        | TraceOp::Cat { .. }
        | TraceOp::Expand { .. }
        | TraceOp::Flip { .. }
        | TraceOp::Roll { .. }
        | TraceOp::Unfold { .. }
        | TraceOp::SliceSet { .. }
        | TraceOp::Triu { .. }
        | TraceOp::Tril { .. }
        | TraceOp::PixelShuffle { .. }
        | TraceOp::PixelUnshuffle { .. }
        | TraceOp::Upsample1d { .. }
        | TraceOp::IndexSelect { .. }
        | TraceOp::Gather { .. }
        | TraceOp::RepeatInterleave { .. }
        | TraceOp::ReflectionPad1d { .. }
        | TraceOp::ReflectionPad2d { .. }
        | TraceOp::ConstantPadNd { .. }
        | TraceOp::Arange { .. }
        | TraceOp::Dropout
        | TraceOp::Neg => Exact,

        // -- Exact: sign-aware affine arithmetic ------------------------------
        // Interval arithmetic is exact for ±/scalar-affine and min/max; these
        // introduce no relaxation error on bounds.
        TraceOp::Add
        | TraceOp::Sub
        | TraceOp::Maximum
        | TraceOp::Minimum
        | TraceOp::Abs
        | TraceOp::Sign
        | TraceOp::Floor
        | TraceOp::Ceil
        | TraceOp::Round
        | TraceOp::Linear { .. }
        | TraceOp::Conv1d { .. }
        | TraceOp::Conv2d { .. }
        | TraceOp::Conv3d { .. }
        | TraceOp::ConvTranspose1d { .. }
        | TraceOp::ConvTranspose2d { .. }
        | TraceOp::Embedding { .. }
        | TraceOp::QLinear { .. }
        | TraceOp::Cumsum { .. } => Exact,

        // -- Sound: monotone / piecewise activations & relaxed multiplies -----
        // Correct, practically tight relaxations (ReLU family, sigmoid/tanh
        // family, clamps, dtype/cast). `Mul`/`Sqr` use an exact-on-intervals
        // product rule; the activations use standard tight convex relaxations.
        TraceOp::Mul
        | TraceOp::Sqr
        | TraceOp::Relu
        | TraceOp::Gelu
        | TraceOp::GeluErf
        | TraceOp::Silu
        | TraceOp::Tanh
        | TraceOp::Sigmoid
        | TraceOp::Sin
        | TraceOp::Cos
        | TraceOp::Tan
        | TraceOp::Fract
        | TraceOp::Atan2
        | TraceOp::Activation { .. }
        | TraceOp::Elu { .. }
        | TraceOp::LeakyRelu { .. }
        | TraceOp::Softplus
        | TraceOp::Selu
        | TraceOp::Celu { .. }
        | TraceOp::Mish
        | TraceOp::HardSigmoid
        | TraceOp::HardSwish
        | TraceOp::Softsign
        | TraceOp::PRelu { .. }
        | TraceOp::SwiGlu
        | TraceOp::Powf { .. }
        | TraceOp::Clamp { .. }
        | TraceOp::ToDtype { .. }
        | TraceOp::MaxPool1d { .. }
        | TraceOp::MaxPool2d { .. }
        | TraceOp::AdaptiveMaxPool2d { .. } => Sound,

        // Comparisons: lowered to Compare/CompareTensor layers whose
        // {0,1}-interval IBP is exact off the threshold and the [0,1] hull
        // across it — sound, and as tight as intervals allow (IBP-only; a
        // step has no useful CROWN linear relaxation).
        TraceOp::Compare { .. } | TraceOp::CompareTensor { .. } => Sound,

        // -- SoundButLoose: normalisation, attention, reductions, transcend. --
        // Correct over-approximations whose interval relaxations widen quickly:
        // anything that divides by a data-dependent denominator, exponentiates,
        // takes logs/roots, normalises across elements, or resamples spatially.
        TraceOp::Div
        | TraceOp::Recip
        | TraceOp::Exp
        | TraceOp::Log
        | TraceOp::Sqrt
        | TraceOp::MatMul
        | TraceOp::ReduceSum { .. }
        | TraceOp::ReduceMean { .. }
        | TraceOp::ReduceMax { .. }
        | TraceOp::ReduceMin { .. }
        | TraceOp::LayerNorm { .. }
        | TraceOp::RmsNorm { .. }
        | TraceOp::GroupNorm { .. }
        | TraceOp::InstanceNorm { .. }
        | TraceOp::BatchNorm { .. }
        | TraceOp::Softmax { .. }
        | TraceOp::LogSoftmax { .. }
        | TraceOp::Sdpa { .. }
        | TraceOp::SdpaCausal { .. }
        | TraceOp::RotaryEmbedding { .. }
        | TraceOp::MultiHeadAttention { .. }
        | TraceOp::Lstm { .. }
        | TraceOp::AvgPool1d { .. }
        | TraceOp::AvgPool2d { .. }
        | TraceOp::AdaptiveAvgPool1d { .. }
        | TraceOp::AdaptiveAvgPool2d { .. }
        | TraceOp::KokoroFused(_)
        | TraceOp::Upsample2d { .. }
        | TraceOp::ResizeBilinear { .. }
        | TraceOp::GridSample { .. } => SoundButLoose,

        // `Custom` — the explicit opaque escape hatch — is the maximally
        // loose case: lowered to the graph builder's OpaqueSkip substitution,
        // a vacuous-but-sound `[-inf, +inf]` over-approximation of any op
        // (INC-FINAL reconciliation; see translate/mod.rs Custom arm and
        // ny-propagate layers/misc/skip_merge.rs for the verified rule).
        TraceOp::Custom { .. } => SoundButLoose,

        // -- Unsupported: data-dependent shape/routing ------------------------
        // Output shape, selection, or routing depends on tensor *values* — the
        // bridge must refuse (sound by construction) and let upstream graph
        // segmentation handle the boundary.
        TraceOp::Topk { .. }
        | TraceOp::Argmax { .. }
        | TraceOp::Argmin { .. }
        | TraceOp::ArgSort { .. }
        | TraceOp::Sort { .. }
        | TraceOp::Scatter { .. }
        | TraceOp::ScatterAdd { .. }
        | TraceOp::IndexAdd { .. }
        | TraceOp::IndexPut { .. }
        | TraceOp::MoeGating { .. }
        | TraceOp::WhereCond
        | TraceOp::SegmentBoundary { .. } => Unsupported,
    }
}

/// Convenience predicate: does a sound lowering of this op exist in principle?
///
/// Equivalent to `trace_op_soundness(op).is_translatable()` — i.e. `true` for
/// every classification except [`BridgeSoundness::Unsupported`]. This does not
/// imply [`crate::translate`] implements the lowering today: `translate`
/// refuses ops outside its implemented core set with an explicit error.
#[must_use]
pub fn is_translatable(op: &TraceOp) -> bool {
    trace_op_soundness(op).is_translatable()
}

/// A bucketed summary of every [`TraceOp`]'s bridge soundness.
///
/// Each field holds the *names* of the variants in that bucket, in canonical
/// order. Produced by [`coverage`]; the union of the four buckets is exactly the
/// 123-variant catalogue (asserted in tests).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CoverageReport {
    /// Variants classified [`BridgeSoundness::Exact`].
    pub exact: Vec<String>,
    /// Variants classified [`BridgeSoundness::Sound`].
    pub sound: Vec<String>,
    /// Variants classified [`BridgeSoundness::SoundButLoose`].
    pub loose: Vec<String>,
    /// Variants classified [`BridgeSoundness::Unsupported`].
    pub unsupported: Vec<String>,
}

impl CoverageReport {
    /// Total number of classified variants across all four buckets.
    #[must_use]
    pub fn total(&self) -> usize {
        self.exact.len() + self.sound.len() + self.loose.len() + self.unsupported.len()
    }

    /// Number of *translatable* variants (everything but `unsupported`).
    #[must_use]
    pub fn translatable(&self) -> usize {
        self.exact.len() + self.sound.len() + self.loose.len()
    }
}

/// Canonical catalogue of every `TraceOp` variant: `(name, sample op)`.
///
/// The list length is pinned to the enum's variant count by the schema's own
/// test and by [`tests::canonical_catalogue_has_123_entries`] here, so it cannot
/// silently drift from [`crate::schema::TraceOp`]. The sample value carries a
/// representative payload so [`trace_op_soundness`] can classify it; only the
/// discriminant matters.
///
/// Order mirrors the source enum declaration (and the schema's discriminant
/// table) for easy cross-referencing.
fn canonical_trace_ops() -> Vec<(&'static str, TraceOp)> {
    use crate::schema::{
        CompareOp, DType, GridSamplePaddingMode, TraceActivation, TraceUpsampleMode, WeightPayload,
    };
    let w = || WeightPayload::f32(vec![0.0], vec![1]);
    let ow = || Some(WeightPayload::f32(vec![0.0], vec![1]));
    vec![
        ("Input", TraceOp::Input),
        ("ConstantWeight", TraceOp::ConstantWeight { weight: w() }),
        ("Add", TraceOp::Add),
        ("Sub", TraceOp::Sub),
        ("Mul", TraceOp::Mul),
        ("Div", TraceOp::Div),
        ("Maximum", TraceOp::Maximum),
        ("Minimum", TraceOp::Minimum),
        ("MatMul", TraceOp::MatMul),
        ("Relu", TraceOp::Relu),
        ("Gelu", TraceOp::Gelu),
        ("GeluErf", TraceOp::GeluErf),
        ("Silu", TraceOp::Silu),
        ("Tanh", TraceOp::Tanh),
        ("Sigmoid", TraceOp::Sigmoid),
        ("Exp", TraceOp::Exp),
        ("Log", TraceOp::Log),
        ("Sqrt", TraceOp::Sqrt),
        ("Sqr", TraceOp::Sqr),
        ("Abs", TraceOp::Abs),
        ("Neg", TraceOp::Neg),
        ("Recip", TraceOp::Recip),
        ("Sin", TraceOp::Sin),
        ("Cos", TraceOp::Cos),
        ("Tan", TraceOp::Tan),
        ("Floor", TraceOp::Floor),
        ("Ceil", TraceOp::Ceil),
        ("Round", TraceOp::Round),
        ("Sign", TraceOp::Sign),
        ("Fract", TraceOp::Fract),
        (
            "ReduceSum",
            TraceOp::ReduceSum {
                dim: 0,
                keepdim: false,
            },
        ),
        (
            "ReduceMean",
            TraceOp::ReduceMean {
                dim: 0,
                keepdim: false,
            },
        ),
        (
            "ReduceMax",
            TraceOp::ReduceMax {
                dim: 0,
                keepdim: false,
            },
        ),
        (
            "ReduceMin",
            TraceOp::ReduceMin {
                dim: 0,
                keepdim: false,
            },
        ),
        (
            "Reshape",
            TraceOp::Reshape {
                target_shape: vec![1],
            },
        ),
        ("Transpose", TraceOp::Transpose { dim0: 0, dim1: 1 }),
        (
            "Narrow",
            TraceOp::Narrow {
                dim: 0,
                start: 0,
                length: 1,
            },
        ),
        ("Unsqueeze", TraceOp::Unsqueeze { dim: 0 }),
        ("Squeeze", TraceOp::Squeeze { dim: 0 }),
        ("Permute", TraceOp::Permute { axes: vec![0] }),
        (
            "Cat",
            TraceOp::Cat {
                dim: 0,
                num_inputs: 2,
            },
        ),
        (
            "LayerNorm",
            TraceOp::LayerNorm {
                eps: 1e-5,
                weight: w(),
                bias: w(),
            },
        ),
        (
            "RmsNorm",
            TraceOp::RmsNorm {
                eps: 1e-5,
                weight: w(),
            },
        ),
        (
            "GroupNorm",
            TraceOp::GroupNorm {
                num_groups: 1,
                eps: 1e-5,
                weight: w(),
                bias: w(),
            },
        ),
        ("InstanceNorm", TraceOp::InstanceNorm { eps: 1e-5 }),
        (
            "BatchNorm",
            TraceOp::BatchNorm {
                eps: 1e-5,
                weight: w(),
                bias: w(),
                running_mean: w(),
                running_var: w(),
            },
        ),
        (
            "Linear",
            TraceOp::Linear {
                weight: w(),
                bias: ow(),
            },
        ),
        (
            "Conv1d",
            TraceOp::Conv1d {
                weight: w(),
                bias: ow(),
                padding: 0,
                stride: 1,
                dilation: 1,
                groups: 1,
            },
        ),
        (
            "Conv2d",
            TraceOp::Conv2d {
                weight: w(),
                bias: ow(),
                padding: [0, 0],
                stride: [1, 1],
                dilation: [1, 1],
                groups: 1,
            },
        ),
        (
            "Conv3d",
            TraceOp::Conv3d {
                weight: w(),
                bias: ow(),
                padding: [0, 0, 0],
                stride: [1, 1, 1],
                dilation: [1, 1, 1],
                groups: 1,
            },
        ),
        (
            "ConvTranspose1d",
            TraceOp::ConvTranspose1d {
                weight: w(),
                bias: ow(),
                padding: 0,
                output_padding: 0,
                stride: 1,
                dilation: 1,
                groups: 1,
            },
        ),
        (
            "ConvTranspose2d",
            TraceOp::ConvTranspose2d {
                weight: w(),
                bias: ow(),
                padding: [0, 0],
                output_padding: [0, 0],
                stride: [1, 1],
                dilation: [1, 1],
                groups: 1,
            },
        ),
        ("Softmax", TraceOp::Softmax { dim: 0 }),
        ("LogSoftmax", TraceOp::LogSoftmax { dim: 0 }),
        ("Sdpa", TraceOp::Sdpa { scale: 1.0 }),
        ("SdpaCausal", TraceOp::SdpaCausal { scale: 1.0 }),
        (
            "RotaryEmbedding",
            TraceOp::RotaryEmbedding {
                head_dim: 8,
                offset: 0,
                cos_cache: w(),
                sin_cache: w(),
            },
        ),
        (
            "MultiHeadAttention",
            TraceOp::MultiHeadAttention {
                num_heads: 1,
                num_kv_heads: 1,
                head_dim: 8,
            },
        ),
        ("Embedding", TraceOp::Embedding { weight: w() }),
        (
            "Lstm",
            TraceOp::Lstm {
                weight_ih: w(),
                weight_hh: w(),
                bias_ih: ow(),
                bias_hh: ow(),
                hidden_size: 1,
                initial_hidden: None,
                initial_cell: None,
            },
        ),
        (
            "MaxPool1d",
            TraceOp::MaxPool1d {
                kernel_size: 1,
                stride: 1,
                padding: 0,
            },
        ),
        (
            "AvgPool2d",
            TraceOp::AvgPool2d {
                kernel_size: [1, 1],
                stride: [1, 1],
                padding: [0, 0],
            },
        ),
        (
            "MaxPool2d",
            TraceOp::MaxPool2d {
                kernel_size: [1, 1],
                stride: [1, 1],
                padding: [0, 0],
            },
        ),
        (
            "AdaptiveAvgPool2d",
            TraceOp::AdaptiveAvgPool2d {
                output_size: [1, 1],
            },
        ),
        (
            "AvgPool1d",
            TraceOp::AvgPool1d {
                kernel_size: 1,
                stride: 1,
                padding: 0,
            },
        ),
        (
            "AdaptiveAvgPool1d",
            TraceOp::AdaptiveAvgPool1d { output_size: 1 },
        ),
        (
            "AdaptiveMaxPool2d",
            TraceOp::AdaptiveMaxPool2d {
                output_size: [1, 1],
            },
        ),
        (
            "Activation",
            TraceOp::Activation {
                kind: TraceActivation::Relu,
            },
        ),
        ("Elu", TraceOp::Elu { alpha: 1.0 }),
        ("LeakyRelu", TraceOp::LeakyRelu { slope: 0.2 }),
        ("Softplus", TraceOp::Softplus),
        ("Selu", TraceOp::Selu),
        ("Celu", TraceOp::Celu { alpha: 1.0 }),
        ("Mish", TraceOp::Mish),
        ("HardSigmoid", TraceOp::HardSigmoid),
        ("HardSwish", TraceOp::HardSwish),
        ("Softsign", TraceOp::Softsign),
        ("PRelu", TraceOp::PRelu { slope: w() }),
        (
            "KokoroFused",
            TraceOp::KokoroFused(KokoroFusedOp::SnakeTensor { alpha: w() }),
        ),
        ("SwiGlu", TraceOp::SwiGlu),
        ("Dropout", TraceOp::Dropout),
        ("PixelShuffle", TraceOp::PixelShuffle { upscale_factor: 2 }),
        (
            "PixelUnshuffle",
            TraceOp::PixelUnshuffle {
                downscale_factor: 2,
            },
        ),
        ("Upsample1d", TraceOp::Upsample1d { factor: 2 }),
        (
            "Upsample2d",
            TraceOp::Upsample2d {
                mode: TraceUpsampleMode::Nearest,
                scale_h: 2.0,
                scale_w: 2.0,
            },
        ),
        (
            "ResizeBilinear",
            TraceOp::ResizeBilinear {
                target_h: 4,
                target_w: 4,
            },
        ),
        ("Triu", TraceOp::Triu { diagonal: 0 }),
        ("Tril", TraceOp::Tril { diagonal: 0 }),
        (
            "GridSample",
            TraceOp::GridSample {
                padding_mode: GridSamplePaddingMode::Zeros,
                align_corners: false,
            },
        ),
        (
            "QLinear",
            TraceOp::QLinear {
                weight: w(),
                bias: ow(),
            },
        ),
        ("Topk", TraceOp::Topk { k: 1, dim: 0 }),
        ("Argmax", TraceOp::Argmax { dim: 0 }),
        ("Argmin", TraceOp::Argmin { dim: 0 }),
        (
            "ArgSort",
            TraceOp::ArgSort {
                dim: 0,
                descending: false,
            },
        ),
        (
            "Sort",
            TraceOp::Sort {
                dim: 0,
                descending: false,
            },
        ),
        ("IndexSelect", TraceOp::IndexSelect { dim: 0 }),
        ("Gather", TraceOp::Gather { dim: 0 }),
        ("WhereCond", TraceOp::WhereCond),
        (
            "Expand",
            TraceOp::Expand {
                target_shape: vec![1],
            },
        ),
        (
            "Compare",
            TraceOp::Compare {
                op: CompareOp::Eq,
                value: 0.0,
            },
        ),
        (
            "CompareTensor",
            TraceOp::CompareTensor { op: CompareOp::Eq },
        ),
        ("Cumsum", TraceOp::Cumsum { dim: 0 }),
        ("RepeatInterleave", TraceOp::RepeatInterleave { dim: 0 }),
        ("Powf", TraceOp::Powf { exponent: 2.0 }),
        (
            "ToDtype",
            TraceOp::ToDtype {
                target_dtype: DType::F32,
            },
        ),
        ("Flip", TraceOp::Flip { dim: 0 }),
        (
            "Roll",
            TraceOp::Roll {
                shifts: vec![1],
                dims: vec![0],
            },
        ),
        (
            "Unfold",
            TraceOp::Unfold {
                dim: 0,
                size: 1,
                step: 1,
            },
        ),
        ("SliceSet", TraceOp::SliceSet { dim: 0, start: 0 }),
        ("Scatter", TraceOp::Scatter { dim: 0 }),
        ("ScatterAdd", TraceOp::ScatterAdd { dim: 0 }),
        ("IndexAdd", TraceOp::IndexAdd { dim: 0 }),
        ("IndexPut", TraceOp::IndexPut { dim: 0 }),
        (
            "Clamp",
            TraceOp::Clamp {
                min: Some(0.0),
                max: Some(1.0),
            },
        ),
        ("Constant", TraceOp::Constant { value: 0.0 }),
        (
            "ReflectionPad1d",
            TraceOp::ReflectionPad1d {
                pad_left: 1,
                pad_right: 1,
            },
        ),
        (
            "ReflectionPad2d",
            TraceOp::ReflectionPad2d {
                pad_left: 1,
                pad_right: 1,
                pad_top: 1,
                pad_bottom: 1,
            },
        ),
        (
            "ConstantPadNd",
            TraceOp::ConstantPadNd {
                padding: vec![1, 1],
                value: 0.0,
            },
        ),
        ("Atan2", TraceOp::Atan2),
        (
            "Arange",
            TraceOp::Arange {
                start: 0.0,
                end: 1.0,
                step: 1.0,
            },
        ),
        (
            "SegmentBoundary",
            TraceOp::SegmentBoundary {
                reason: "x".into(),
                input_bounds: None,
            },
        ),
        (
            "MoeGating",
            TraceOp::MoeGating {
                num_experts: 1,
                top_k: 1,
            },
        ),
        ("Custom", TraceOp::Custom { name: "x".into() }),
    ]
}

/// Build the bucketed [`CoverageReport`] over the canonical variant catalogue.
///
/// Walks [`canonical_trace_ops`], classifies each with [`trace_op_soundness`],
/// and groups the variant *names* by bucket. The result is deterministic and the
/// union of all four buckets is exactly the 123-variant catalogue.
#[must_use]
pub fn coverage() -> CoverageReport {
    let mut report = CoverageReport {
        exact: Vec::new(),
        sound: Vec::new(),
        loose: Vec::new(),
        unsupported: Vec::new(),
    };
    for (name, op) in canonical_trace_ops() {
        let bucket = match trace_op_soundness(&op) {
            BridgeSoundness::Exact => &mut report.exact,
            BridgeSoundness::Sound => &mut report.sound,
            BridgeSoundness::SoundButLoose => &mut report.loose,
            BridgeSoundness::Unsupported => &mut report.unsupported,
        };
        bucket.push(name.to_owned());
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// The number of `TraceOp` variants, pinned identically to `schema`.
    const EXPECTED_TRACE_OP_VARIANTS: usize = 123;

    #[test]
    fn canonical_catalogue_has_123_entries() {
        // If this fails, the catalogue drifted from the enum: add the new
        // variant here *and* classify it in `trace_op_soundness` (which will
        // not compile until you do).
        assert_eq!(canonical_trace_ops().len(), EXPECTED_TRACE_OP_VARIANTS);
    }

    #[test]
    fn catalogue_names_are_unique() {
        let names: BTreeSet<&str> = canonical_trace_ops().iter().map(|(n, _)| *n).collect();
        assert_eq!(
            names.len(),
            EXPECTED_TRACE_OP_VARIANTS,
            "every catalogue entry must have a distinct name"
        );
    }

    #[test]
    fn buckets_partition_all_123_variants() {
        let report = coverage();
        // Sum of buckets == total catalogue (no variant dropped/duplicated).
        assert_eq!(report.total(), EXPECTED_TRACE_OP_VARIANTS);

        // The four buckets are pairwise disjoint and their union is the full
        // catalogue — i.e. they *partition* it.
        let mut union: BTreeSet<String> = BTreeSet::new();
        for name in report
            .exact
            .iter()
            .chain(&report.sound)
            .chain(&report.loose)
            .chain(&report.unsupported)
        {
            assert!(
                union.insert(name.clone()),
                "variant {name} appears in more than one bucket"
            );
        }
        let catalogue: BTreeSet<String> = canonical_trace_ops()
            .iter()
            .map(|(n, _)| (*n).to_owned())
            .collect();
        assert_eq!(
            union, catalogue,
            "bucket union must equal the canonical catalogue"
        );
    }

    #[test]
    fn classification_is_total_and_agrees_with_buckets() {
        // Every catalogue op classifies, and its bucket membership in the
        // report agrees with the direct classifier — guards against the report
        // builder and the classifier diverging.
        let report = coverage();
        for (name, op) in canonical_trace_ops() {
            let s = trace_op_soundness(&op);
            let in_bucket = match s {
                BridgeSoundness::Exact => &report.exact,
                BridgeSoundness::Sound => &report.sound,
                BridgeSoundness::SoundButLoose => &report.loose,
                BridgeSoundness::Unsupported => &report.unsupported,
            };
            assert!(
                in_bucket.iter().any(|n| n == name),
                "{name} classified {s:?} but missing from that bucket"
            );
        }
    }

    #[test]
    fn is_translatable_matches_non_unsupported() {
        for (name, op) in canonical_trace_ops() {
            let s = trace_op_soundness(&op);
            assert_eq!(
                is_translatable(&op),
                s != BridgeSoundness::Unsupported,
                "is_translatable disagrees with soundness for {name}"
            );
            assert_eq!(is_translatable(&op), s.is_translatable());
        }
    }

    #[test]
    fn data_dependent_ops_are_unsupported() {
        // The roadmap-pinned data-dependent / opaque set MUST be Unsupported so
        // the bridge refuses them (sound by construction).
        let must_be_unsupported = [
            TraceOp::Topk { k: 1, dim: 0 },
            TraceOp::Argmax { dim: 0 },
            TraceOp::Argmin { dim: 0 },
            TraceOp::ArgSort {
                dim: 0,
                descending: false,
            },
            TraceOp::Scatter { dim: 0 },
            TraceOp::ScatterAdd { dim: 0 },
            TraceOp::IndexPut { dim: 0 },
            TraceOp::MoeGating {
                num_experts: 1,
                top_k: 1,
            },
            TraceOp::WhereCond,
        ];
        for op in must_be_unsupported {
            assert_eq!(
                trace_op_soundness(&op),
                BridgeSoundness::Unsupported,
                "{op:?} must be Unsupported"
            );
            assert!(!is_translatable(&op));
        }
    }

    /// `Custom` — the explicit opaque escape hatch — is translatable via the
    /// vacuous-but-sound OpaqueSkip lowering (INC-FINAL reconciliation), and
    /// classified maximally loose, never Exact/Sound.
    #[test]
    fn custom_is_sound_but_loose() {
        let op = TraceOp::Custom { name: "x".into() };
        assert_eq!(trace_op_soundness(&op), BridgeSoundness::SoundButLoose);
        assert!(is_translatable(&op));
    }

    #[test]
    fn structural_and_affine_ops_are_exact() {
        // Spot-check that pure reshape/reindex and affine ops stay Exact.
        for op in [
            TraceOp::Input,
            TraceOp::Reshape {
                target_shape: vec![1],
            },
            TraceOp::Transpose { dim0: 0, dim1: 1 },
            TraceOp::Add,
            TraceOp::Sub,
            TraceOp::Linear {
                weight: crate::schema::WeightPayload::f32(vec![0.0], vec![1]),
                bias: None,
            },
        ] {
            assert_eq!(trace_op_soundness(&op), BridgeSoundness::Exact);
        }
    }

    #[test]
    fn norms_and_attention_are_loose() {
        for op in [
            TraceOp::Softmax { dim: 0 },
            TraceOp::Sdpa { scale: 1.0 },
            TraceOp::LayerNorm {
                eps: 1e-5,
                weight: crate::schema::WeightPayload::f32(vec![0.0], vec![1]),
                bias: crate::schema::WeightPayload::f32(vec![0.0], vec![1]),
            },
            TraceOp::Exp,
            TraceOp::Div,
        ] {
            assert_eq!(trace_op_soundness(&op), BridgeSoundness::SoundButLoose);
        }
    }

    #[test]
    fn soundness_round_trips_json() {
        for s in [
            BridgeSoundness::Exact,
            BridgeSoundness::Sound,
            BridgeSoundness::SoundButLoose,
            BridgeSoundness::Unsupported,
        ] {
            let json = serde_json::to_string(&s).unwrap();
            let back: BridgeSoundness = serde_json::from_str(&json).unwrap();
            assert_eq!(s, back);
            assert_eq!(json.trim_matches('"'), s.as_str());
        }
    }

    #[test]
    fn coverage_report_round_trips_json() {
        let report = coverage();
        let json = serde_json::to_string(&report).unwrap();
        let back: CoverageReport = serde_json::from_str(&json).unwrap();
        assert_eq!(report, back);
        assert_eq!(back.translatable() + back.unsupported.len(), back.total());
    }
}
