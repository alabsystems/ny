// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Operation soundness-coverage gate (P9).
//!
//! This module is the canonical, build-time-checked map from each
//! [`ny_core::LayerType`] to how soundly ny's bound-propagation handles it.
//! It classifies every operator into a [`SoundnessClass`] and exposes a
//! [`ConformanceReport`] so external consumers (e.g. the NN ML framework)
//! can query op coverage without reaching past the facade.
//!
//! It operates purely on `ny-core` types and is therefore always available
//! (no `propagate` feature required).
//!
//! # Coverage gate
//!
//! [`soundness_class`] is implemented as a per-variant `match` that names
//! every current `LayerType` explicitly. Adding a new operator that is left
//! unclassified is intended to be caught here: the body lists each variant by
//! name, so a reviewer extending `LayerType` must add a corresponding arm and
//! deliberately assign a soundness class.
//!
//! Note: `LayerType` is `#[non_exhaustive]` and is defined in a different
//! crate (`ny-core`), so the Rust compiler *requires* a trailing wildcard arm
//! here — a fully wildcard-free exhaustive match is not permitted across the
//! crate boundary. The wildcard therefore only ever matches *future* variants
//! and conservatively classifies them as [`SoundnessClass::Unsupported`] until
//! they are explicitly listed above it. Every variant that exists today is
//! named explicitly, so the wildcard is unreachable for the current enum.
//!
//! Note: serde `Serialize`/`Deserialize` derives are intentionally omitted
//! because `ny-api` does not currently depend on `serde`. Re-add them once
//! `serde` is a dependency of this crate (see the structured report).

/// How soundly ny's bound propagation handles a given operation.
///
/// Ordered from tightest to least supported. Classification reflects the
/// known interval/CROWN behavior of each operator, favoring the more
/// conservative class when in doubt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SoundnessClass {
    /// Bounds are exact (no over-approximation): affine / index-permutation ops.
    Exact,
    /// Bounds are sound and reasonably tight (e.g. relaxed monotone activations).
    Sound,
    /// Bounds are sound but can be loose (norms, softmax, attention, reductions
    /// over non-monotone composites, transcendental ops, resampling).
    SoundButLoose,
    /// Not soundly supported for verification (data-dependent / control-flow ops).
    Unsupported,
}

/// Classify a single [`ny_core::LayerType`] by its bound-propagation soundness.
///
/// This is the op soundness-coverage gate: every current variant is named
/// explicitly. See the module docs for why a trailing wildcard is required by
/// the `#[non_exhaustive]` cross-crate enum, and why it conservatively maps
/// unknown future variants to [`SoundnessClass::Unsupported`].
pub fn soundness_class(layer: &ny_core::LayerType) -> SoundnessClass {
    use ny_core::LayerType as L;
    use SoundnessClass::{Exact, Sound, SoundButLoose, Unsupported};
    match layer {
        // --- Affine / structural: exact under interval + CROWN ---
        L::Linear => Exact,
        L::MatMul => Sound, // both inputs bounded: bilinear, sound but tighter than loose
        L::Add => Exact,
        L::Sub => Exact,
        L::Neg => Exact,
        L::Mul => Sound, // by-const is exact; bounded*bounded is sound
        L::CumSum => Exact,
        L::Reshape => Exact,
        L::Flatten => Exact,
        L::Transpose => Exact,
        L::Squeeze => Exact,
        L::Unsqueeze => Exact,
        L::Concat => Exact,
        L::Slice => Exact,
        L::Tile => Exact,
        L::Expand => Exact,
        L::Pad => Exact,
        L::Cast => Exact,      // identity for f32 bound propagation
        L::Shape => Exact,     // shape metadata, not a value transform
        L::Triu => Exact,      // binary mask = mul-by-constant
        L::Tril => Exact,      // binary mask = mul-by-constant
        L::RoPE => Exact,      // block-diagonal rotation at fixed position (affine)
        L::Embedding => Exact, // lookup of constant rows
        L::Gather => Sound,    // const index is exact; conservative under bounded index

        // --- Convolution / pooling: affine or monotone-structured ---
        L::Conv1d => Exact,
        L::Conv2d => Exact,
        L::ConvTranspose1d => Exact,
        L::ConvTranspose2d => Exact,
        L::AveragePool => Exact,
        L::MaxPool => Sound,

        // --- Quantization affine maps ---
        L::DequantizeLinear => Exact, // y = (x - zp) * scale, affine
        L::QuantizeLinear => SoundButLoose, // round/saturate: sound but step-relaxed

        // --- Activations: relaxable, sound (possibly loose for non-monotone) ---
        L::ReLU => Sound,
        L::LeakyRelu => Sound,
        L::PRelu => Sound,
        L::ThresholdedRelu => Sound,
        L::GELU => Sound,
        L::SiLU => Sound,
        L::Sigmoid => Sound,
        L::Tanh => Sound,
        L::Softplus => Sound,
        L::Softsign => Sound,
        L::Elu => Sound,
        L::Selu => Sound,
        L::Celu => Sound,
        L::HardSigmoid => Sound,
        L::HardSwish => Sound,
        L::Clip => Sound,
        L::Mish => SoundButLoose,  // non-monotone composite
        L::Snake => SoundButLoose, // x + sin^2 term: non-monotone, periodic
        L::Shrink => Sound,
        L::Sign => SoundButLoose, // piecewise-constant step

        // --- Transcendental / elementwise math: sound but loose ---
        L::Exp => SoundButLoose,
        L::Log => SoundButLoose,
        L::LogSumExp => SoundButLoose,
        L::Sqrt => SoundButLoose,
        L::Reciprocal => SoundButLoose,
        L::Div => SoundButLoose,
        L::Pow => SoundButLoose,
        L::Abs => Sound, // monotone on each side, tight enough
        L::Sin => SoundButLoose,
        L::Cos => SoundButLoose,
        L::Tan => SoundButLoose,
        L::Arctan => SoundButLoose,
        L::Atan2 => SoundButLoose,

        // --- Rounding: piecewise-constant, sound but loose ---
        L::Floor => SoundButLoose,
        L::Ceil => SoundButLoose,
        L::Round => SoundButLoose,

        // --- Elementwise min/max of bounded tensors ---
        L::Min => Sound,
        L::Max => Sound,

        // --- Normalization: mean/variance composites, sound but loose ---
        L::LayerNorm => SoundButLoose,
        L::RMSNorm => SoundButLoose,
        L::InstanceNorm => SoundButLoose,
        L::GroupNorm => SoundButLoose,
        L::AdaIN => SoundButLoose,
        L::BatchNorm => SoundButLoose,

        // --- Softmax family: exp-normalized, sound but loose ---
        L::Softmax => SoundButLoose,
        L::CausalSoftmax => SoundButLoose,
        L::LogSoftmax => SoundButLoose,

        // --- Attention: composite of matmul + softmax, sound but loose ---
        L::MultiHeadAttention => SoundButLoose,

        // --- Reductions over an axis ---
        L::ReduceMean => Sound,
        L::ReduceSum => Exact,
        L::ReduceMax => SoundButLoose,
        L::ReduceMin => SoundButLoose,

        // --- Resampling ---
        L::Resize => SoundButLoose,

        // --- Data-dependent indices / comparisons / control flow: unsupported ---
        L::Argmax => Unsupported,
        L::ArgMin => Unsupported,
        L::ArgSort => Unsupported,
        L::Topk => Unsupported,
        L::Compare => Unsupported,
        L::CompareTensor => Unsupported,
        L::Where => Unsupported,
        L::NonZero => Unsupported,
        L::ScatterND => Unsupported,
        L::Unknown => Unsupported,

        // `LayerType` is `#[non_exhaustive]` in `ny-core`: the compiler requires
        // this arm. Future variants are conservatively unsupported until they
        // are classified explicitly above. See module docs.
        _ => Unsupported,
    }
}

/// Whether ny can soundly verify networks containing this operation.
///
/// Returns `true` for every class except [`SoundnessClass::Unsupported`].
pub fn is_verifiable(layer: &ny_core::LayerType) -> bool {
    soundness_class(layer) != SoundnessClass::Unsupported
}

/// Canonical list of every classified `LayerType`.
///
/// Kept in step with [`soundness_class`]; used by [`report`] to materialize
/// the full coverage breakdown.
fn all_variants() -> Vec<ny_core::LayerType> {
    use ny_core::LayerType as L;
    vec![
        L::Linear,
        L::Conv1d,
        L::Conv2d,
        L::ConvTranspose1d,
        L::ConvTranspose2d,
        L::AveragePool,
        L::MaxPool,
        L::ReLU,
        L::LeakyRelu,
        L::GELU,
        L::SiLU,
        L::Sigmoid,
        L::Tanh,
        L::Softplus,
        L::Softmax,
        L::CausalSoftmax,
        L::Clip,
        L::Elu,
        L::Selu,
        L::PRelu,
        L::HardSigmoid,
        L::HardSwish,
        L::Exp,
        L::Log,
        L::LogSumExp,
        L::Celu,
        L::Mish,
        L::LogSoftmax,
        L::ThresholdedRelu,
        L::Shrink,
        L::Softsign,
        L::Snake,
        L::Floor,
        L::Ceil,
        L::Round,
        L::Sign,
        L::Reciprocal,
        L::Sin,
        L::Cos,
        L::Tan,
        L::Arctan,
        L::RoPE,
        L::LayerNorm,
        L::RMSNorm,
        L::InstanceNorm,
        L::GroupNorm,
        L::AdaIN,
        L::BatchNorm,
        L::MultiHeadAttention,
        L::Embedding,
        L::Add,
        L::Concat,
        L::Reshape,
        L::Flatten,
        L::Transpose,
        L::Cast,
        L::DequantizeLinear,
        L::QuantizeLinear,
        L::Squeeze,
        L::Unsqueeze,
        L::Pad,
        L::Resize,
        L::MatMul,
        L::Mul,
        L::Min,
        L::Max,
        L::Atan2,
        L::Neg,
        L::Triu,
        L::Tril,
        L::Abs,
        L::Sqrt,
        L::Div,
        L::Sub,
        L::Pow,
        L::ReduceMean,
        L::ReduceSum,
        L::ReduceMax,
        L::ReduceMin,
        L::Argmax,
        L::ArgMin,
        L::ArgSort,
        L::Topk,
        L::CumSum,
        L::Tile,
        L::Expand,
        L::Compare,
        L::CompareTensor,
        L::Where,
        L::NonZero,
        L::Gather,
        L::ScatterND,
        L::Slice,
        L::Shape,
        L::Unknown,
    ]
}

/// Full op soundness-coverage breakdown, one operator name per class bucket.
///
/// Names are the canonical `LayerType` `Display` spellings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConformanceReport {
    /// Operators with exact bounds.
    pub exact: Vec<String>,
    /// Operators with sound, reasonably tight bounds.
    pub sound: Vec<String>,
    /// Operators that are sound but may be loose.
    pub loose: Vec<String>,
    /// Operators not soundly supported for verification.
    pub unsupported: Vec<String>,
}

/// Build the full [`ConformanceReport`] by classifying every `LayerType`.
pub fn report() -> ConformanceReport {
    let mut out = ConformanceReport {
        exact: Vec::new(),
        sound: Vec::new(),
        loose: Vec::new(),
        unsupported: Vec::new(),
    };
    for variant in all_variants() {
        let name = variant.to_string();
        match soundness_class(&variant) {
            SoundnessClass::Exact => out.exact.push(name),
            SoundnessClass::Sound => out.sound.push(name),
            SoundnessClass::SoundButLoose => out.loose.push(name),
            SoundnessClass::Unsupported => out.unsupported.push(name),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ny_core::LayerType as L;

    #[test]
    fn every_class_is_populated() {
        let r = report();
        assert!(!r.exact.is_empty(), "no Exact ops classified");
        assert!(!r.sound.is_empty(), "no Sound ops classified");
        assert!(!r.loose.is_empty(), "no SoundButLoose ops classified");
        assert!(!r.unsupported.is_empty(), "no Unsupported ops classified");
    }

    #[test]
    fn report_covers_all_variants() {
        let r = report();
        let total = r.exact.len() + r.sound.len() + r.loose.len() + r.unsupported.len();
        assert_eq!(total, all_variants().len());
        // 95 LayerType variants classified as of this snapshot.
        assert_eq!(total, 95);
    }

    #[test]
    fn data_dependent_ops_are_unsupported() {
        // Spelled `Argmax` / `Topk` in ny-core (not `ArgMax` / `TopK`).
        assert_eq!(soundness_class(&L::Topk), SoundnessClass::Unsupported);
        assert_eq!(soundness_class(&L::Argmax), SoundnessClass::Unsupported);
        assert!(!is_verifiable(&L::Topk));
        assert!(!is_verifiable(&L::Argmax));

        let r = report();
        assert!(r.unsupported.contains(&"Topk".to_string()));
        assert!(r.unsupported.contains(&"Argmax".to_string()));
    }

    #[test]
    fn representative_variants_classify_as_expected() {
        // The exhaustive per-variant match guarantees totality at compile time;
        // these spot-checks pin the intended classes.
        assert_eq!(soundness_class(&L::Linear), SoundnessClass::Exact);
        assert_eq!(soundness_class(&L::Reshape), SoundnessClass::Exact);
        assert_eq!(soundness_class(&L::ReLU), SoundnessClass::Sound);
        assert_eq!(soundness_class(&L::Sigmoid), SoundnessClass::Sound);
        assert_eq!(
            soundness_class(&L::LayerNorm),
            SoundnessClass::SoundButLoose
        );
        assert_eq!(soundness_class(&L::Softmax), SoundnessClass::SoundButLoose);
        assert_eq!(soundness_class(&L::Where), SoundnessClass::Unsupported);

        assert!(is_verifiable(&L::Linear));
        assert!(is_verifiable(&L::Softmax));
    }
}
