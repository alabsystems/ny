// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Operator soundness classifications over [`ny_core::LayerType`], available
//! without the `propagate` feature through [`ConformanceReport`].
//!
//! `LayerType` is a cross-crate `#[non_exhaustive]` enum, so the required
//! wildcard conservatively returns [`SoundnessClass::Unsupported`]. Rust
//! cannot enforce exhaustive classification here. When adding a variant,
//! update both [`soundness_class`] and [`all_variants`], not just a test count.
//!
//! `tests::report_covers_all_variants` pins the expected catalogue size;
//! `tests::all_variants_is_a_duplicate_free_round_tripping_list` checks its
//! identities; `tests::unsupported_bucket_is_exactly_the_two_intended_ops`
//! pins unsupported names so a listed but unclassified variant fails visibly.
//! These checks complement the required manual review of enum additions.
//! Classifications do not certify the numeric soundness of an individual run.

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
///
/// Classification is deliberately asymmetric. Reporting an op *stricter* than
/// it is costs a spuriously refused verification; reporting it *looser* than it
/// is turns a downstream gate's `true` into a false proof. So every arm below
/// is justified against the layer that actually propagates it in
/// `ny-propagate`, and where one `LayerType` covers several lowered behaviours
/// (`Topk`, `ScatterND`) the arm takes the weakest of them.
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
        // Conv3d: the same affine map as its lower-rank siblings, and no longer
        // a classification that outruns the implementation —
        // `ny_propagate::Conv3dLayer` exists (layers/convolution/conv3d),
        // ny-build has the convert arm, and the trace bridge emits the spec.
        // Its ONLY interval forward is the certified directed-binary64
        // enclosure in `conv3d::ops_ibp_cert`, so the affine map is reproduced
        // exactly (outward rounding only, where an endpoint is not
        // representable). Its CROWN backward fails closed with `UnsupportedOp`,
        // which degrades the network to IBP at that node — an enclosure-
        // TIGHTNESS question, not a soundness one, exactly as
        // `coverage.rs` reasons when it classifies `TraceOp::Conv3d` `Exact`
        // beside Conv1d/Conv2d.
        L::Conv3d => Exact,
        L::ConvTranspose1d => Exact,
        L::ConvTranspose2d => Exact,
        L::AveragePool => Exact,
        L::MaxPool => Sound,

        // --- Quantization affine maps ---
        L::DequantizeLinear => Exact, // y = (x - zp) * scale, affine
        L::QuantizeLinear => SoundButLoose, // round/saturate: sound but step-relaxed

        // --- Certified additive widening ---
        // y = x + e, |e| <= eps: the interval image IS the declared op
        // semantics, so IBP is exact (outward-rounded only where an endpoint
        // is not representable).
        L::EpsilonExpand => Exact,

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
        // Erf: monotone increasing, odd, range (-1, 1); shares the exact
        // convexity-split relaxation machinery as Tanh/Sigmoid, so it belongs
        // in their class rather than with the transcendentals.
        L::Erf => Sound,

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
        // Trunc: rounds toward zero — the same piecewise-constant step family.
        // (Also the guarded lowering of Cast-to-int, which additionally carries
        // a fail-closed destination-domain obligation at every propagation
        // boundary; that obligation only ever makes it stricter, never looser.)
        L::Trunc => SoundButLoose,

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

        // --- Comparisons: sound {0, 1} step semantics ---
        // NOT data-dependent in the disqualifying sense: the ny-propagate
        // Compare / CompareTensor layers model the output as the exact
        // `{0, 1}` value set — IBP is exact wherever the operand intervals do
        // not straddle the threshold and the `[0, 1]` hull wherever they do,
        // with a zero-slope CROWN relaxation. They were classified
        // `Unsupported` here (making `is_verifiable` report `false`) long
        // after the layers and the trace bridge both treated them as sound,
        // which misreported every mask-using graph as unverifiable.
        L::Compare => Sound,
        L::CompareTensor => Sound,

        // --- Index-producing selection: statically shaped, sound indices ---
        // NOT data-dependent in the disqualifying sense. The output SHAPE is a
        // function of the input shape and the axis alone, and the output VALUES
        // are integer indices that ny-propagate's `reduction/selection.rs`
        // bounds EXACTLY wherever the input intervals are strictly separated —
        // the winner (or, for ArgSort, the whole permutation) is then determined
        // at every point of the box — and by the sound `[0, n-1]` index hull
        // wherever they overlap. Not `Exact`, precisely because of that overlap
        // fallback.
        //
        // Two details carry the soundness. First, ONNX's source-index tie rule
        // is honored by the domination test in `exact_argext_index`: a
        // competitor on the preferred side must be STRICTLY dominated, so a tie
        // can never fabricate a unique winner. Second, index values are modeled
        // as f32 integers widened OUTWARD by one endpoint above 2^24, where
        // consecutive integers stop being representable
        // (`selection.rs::index_interval`) — never snapped to a wrong integer.
        //
        // CROWN backward is an explicit `UnsupportedOp` for all three, which is
        // the drivers' documented fall-back-to-IBP signal (see
        // `network/alpha_crown_loop.rs`), not an unsoundness: the node degrades
        // the network to IBP at that point. This matches
        // `ny_trace_bridge::coverage::trace_op_soundness`, which classifies
        // `TraceOp::Argmax` / `Argmin` / `ArgSort` as `Sound`.
        L::Argmax => Sound,
        L::ArgMin => Sound,
        L::ArgSort => Sound,

        // Topk: `ny_propagate::TopkLayer` carries a `TopkOutputKind` and this
        // single `LayerType` names BOTH forms (ny-build's
        // `read_topk_output_kind` picks between them from the `output`
        // attribute, defaulting to values), so the arm must classify at the
        // weaker of the two:
        //
        //   * VALUES — the layer sorts each bound lane descending and keeps the
        //     first k. The j-th order statistic is monotone in every input, so
        //     those endpoints are attained: EXACT for the `largest && sorted`
        //     function, which is the only one that ever reaches this layer
        //     (`ny_trace_bridge::translate::ops_misc` refuses the other three
        //     flag combinations BY NAME rather than bounding them here, because
        //     the per-position interval is strictly narrower than they need).
        //     A non-finite input lane widens soundly to `[-inf, +inf]`.
        //   * INDICES — the unconditional `[0, n-1]` index hull. Unlike the
        //     Arg* layers above, this form never attempts the separated-input
        //     exact case, so it is ALWAYS loose.
        //
        // Sound either way, loose in the indices form => `SoundButLoose`.
        // `coverage.rs` can be sharper at the TraceOp level — it splits on the
        // serialized `sorted` flag and calls `Topk { sorted: true }` `Exact` —
        // because it classifies the traced FUNCTION, whereas the lowered
        // `LayerType` here cannot distinguish the two output kinds.
        L::Topk => SoundButLoose,

        // --- Conditional select: union hull, sound but loose ---
        // `WhereLayer::propagate_ibp_ternary` IGNORES its condition operand and
        // returns the union hull `[min(x_lo, y_lo), max(x_hi, y_hi)]`, using
        // NaN-propagating min/max so an upstream NaN is never absorbed into a
        // too-tight bound. That hull encloses BOTH branches at every point of
        // the box, hence is sound for any condition whatsoever — and
        // deliberately loose wherever the condition is in fact decided. Only
        // the embedded-constant path (`embedded_constant_select_output`)
        // tightens to the exact select, and only per element where the
        // condition's own interval is determined (one side of 0.5 for a
        // BOOL-valued condition; the trace bridge lowers nn's `mask != 0` to
        // that meaning with an explicit `Compare(Ne, 0)`). IBP-only: a select
        // has no useful CROWN linear relaxation.
        //
        // Sound but loose, therefore — NOT unsupported. Classifying `Where`
        // `Unsupported` reported a soundness hole on every masked graph
        // (attention masks, `Where`-built padding masks) that ny does not
        // actually have. Matches `coverage.rs`'s
        // `TraceOp::WhereCond => SoundButLoose`.
        L::Where => SoundButLoose,

        // --- Indexed writes ---
        // ScatterND overwrites `data` with `updates` at `indices`. Both index
        // regimes are enclosed; neither is exact in general:
        //   * static indices — scatter the lower bounds into the lower output
        //     and the upper into the upper, combining DUPLICATE destinations
        //     with min/max. Overwrite is last-writer-nondeterministic on a
        //     duplicate, and that min/max union IS the enclosure of every visit
        //     order.
        //   * dynamic indices — classify each output cell as definitely /
        //     possibly / never written from the indices' own interval bounds,
        //     degrading to the global min/max hull when those bounds are
        //     unusable (non-finite, wrong shape, over budget).
        // CROWN backward is an explicit `UnsupportedOp` (alpha-beta-CROWN has no
        // ScatterND either), so the node is IBP-only. Sound, knowingly loose.
        L::ScatterND => SoundButLoose,

        // ScatterAdd / IndexAdd: additive indexed accumulation
        // (`ny_propagate::layers::transform::accumulate`). `+=` is associative
        // and commutative over the reals, so — unlike the overwriting ScatterND
        // above — a cell named by several index entries receives the SUM under
        // every visit order: one value, no union needed. In f32 it is neither,
        // and a plain `+=` used for both endpoints would TRANSLATE the interval
        // off the true value rather than widen it around it, so every
        // accumulation here rounds OUTWARD for the endpoint it is building
        // (`add_down_f32` below, `add_up_f32` above; the dynamic hull folds in
        // f64 with directed adds and lands with one directed cast).
        //   * static index — the exact interval image, modulo that outward
        //     rounding.
        //   * dynamic index — the index never has to be read: `data` is widened
        //     by the total possible additive contribution of `src` (the sum of
        //     its negative lower bounds below, of its positive upper bounds
        //     above). Every index assignment routes each `src` element to
        //     exactly one cell, so every realizable output lies in that hull.
        // "Loose" names that dynamic case, plus the static case's own
        // decorrelation when `data` and `src` share upstream variables. Matches
        // `coverage.rs`'s `TraceOp::ScatterAdd` / `IndexAdd => SoundButLoose`.
        L::ScatterAdd => SoundButLoose,
        L::IndexAdd => SoundButLoose,

        // --- Genuinely unsupported ---
        // NonZero: output shape is `[rank, num_nonzero]`, and `num_nonzero` is a
        // function of the tensor's VALUES, not of its shape. `NonZeroLayer`
        // literally sizes its output from the count of elements whose interval
        // is not pinned to zero, so the tensor's shape moves with the bounds:
        // there is no static graph shape for a downstream layer to consume, and
        // CROWN is refused outright. This is the disqualifying kind of
        // data-dependence that the Arg* / Topk layers above do NOT have.
        L::NonZero => Unsupported,
        // Unknown: the explicit "op we could not identify" placeholder. Nothing
        // is known about its semantics, so nothing may be assumed about its
        // bounds.
        L::Unknown => Unsupported,

        // `LayerType` is `#[non_exhaustive]` in `ny-core` and this is a
        // different crate, so the compiler REQUIRES this arm: a wildcard-free
        // exhaustive match is not permitted across the crate boundary, and
        // removing it is a hard compile error (`E0004`), not a lint. Future
        // variants are therefore conservatively unsupported until they are
        // classified explicitly above, and the two test pins described in the
        // module docs — not the type system — are what surface them.
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
        L::Conv3d,
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
        L::Trunc,
        L::Erf,
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
        L::EpsilonExpand,
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
        L::ScatterAdd,
        L::IndexAdd,
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

    /// `all_variants()` must enumerate the whole `LayerType` enum.
    ///
    /// This is the first of the two pins standing in for the wildcard-free
    /// exhaustive match the `#[non_exhaustive]` cross-crate enum forbids: it
    /// catches a variant added to `ny-core` but never listed here (which is
    /// exactly what happened to `Trunc` and `Erf`, and again to `Conv3d`,
    /// `ScatterAdd` and `IndexAdd`). When this fails because `ny-core` grew a
    /// variant, add the match arm in `soundness_class` AND the `all_variants()`
    /// entry — do not just bump the number.
    ///
    /// The other direction — listed but never classified — is caught by
    /// [`unsupported_bucket_is_exactly_the_two_intended_ops`].
    #[test]
    fn report_covers_all_variants() {
        let r = report();
        let total = r.exact.len() + r.sound.len() + r.loose.len() + r.unsupported.len();
        assert_eq!(total, all_variants().len());
        // 101 LayerType variants in ny-core as of this snapshot.
        assert_eq!(total, 101);
    }

    /// No variant is listed twice, and every listed variant's `Display` name
    /// round-trips through `FromStr` back to the same variant — so a typo in
    /// `all_variants()` cannot masquerade as coverage.
    #[test]
    fn all_variants_is_a_duplicate_free_round_tripping_list() {
        use std::collections::BTreeSet;
        use std::str::FromStr;

        let variants = all_variants();
        let names: BTreeSet<String> = variants.iter().map(ToString::to_string).collect();
        assert_eq!(
            names.len(),
            variants.len(),
            "all_variants() contains a duplicate"
        );
        for variant in &variants {
            let name = variant.to_string();
            assert_eq!(
                L::from_str(&name).as_ref(),
                Ok(variant),
                "{name} does not round-trip through FromStr"
            );
        }
    }

    /// Only ops whose output SHAPE moves with tensor VALUES (plus the explicit
    /// `Unknown` placeholder) are unsupported.
    ///
    /// `NonZero` is the whole of the first category: `NonZeroLayer` sizes its
    /// `[rank, num_nonzero]` output from the count of elements whose interval is
    /// not pinned to zero, so no static graph shape exists for a consumer to
    /// read, and its CROWN backward is refused outright. The selection ops that
    /// used to sit beside it here (`Argmax` / `ArgMin` / `ArgSort` / `Topk`)
    /// are statically shaped and therefore do NOT belong in this bucket — see
    /// [`index_selection_ops_are_sound_not_unsupported`].
    #[test]
    fn data_dependent_shape_ops_are_unsupported() {
        assert_eq!(soundness_class(&L::NonZero), SoundnessClass::Unsupported);
        assert_eq!(soundness_class(&L::Unknown), SoundnessClass::Unsupported);
        assert!(!is_verifiable(&L::NonZero));
        assert!(!is_verifiable(&L::Unknown));

        let r = report();
        assert!(r.unsupported.contains(&"NonZero".to_string()));
        assert!(r.unsupported.contains(&"Unknown".to_string()));
    }

    /// The `Unsupported` bucket is pinned BY NAME, not just by size.
    ///
    /// This is the second of the two pins standing in for the exhaustive match
    /// (see the module docs). A `LayerType` that is listed in `all_variants()`
    /// but never classified in `soundness_class` falls into the required
    /// `#[non_exhaustive]` wildcard, lands in this bucket, and shows up here by
    /// name — the failure mode the bumpable variant COUNT cannot see, and the
    /// one that hid `Conv3d` / `ScatterAdd` / `IndexAdd`.
    ///
    /// If this fails with a new name in `unsupported`, the fix is to classify
    /// that variant in `soundness_class` against the ny-propagate layer that
    /// actually bounds it — not to add it to the list below. Add it below only
    /// when the layer genuinely has no sound propagation, with the reason.
    #[test]
    fn unsupported_bucket_is_exactly_the_two_intended_ops() {
        let r = report();
        assert_eq!(
            r.unsupported,
            vec!["NonZero".to_string(), "Unknown".to_string()],
            "unexpected membership in the Unsupported bucket — an unclassified \
             LayerType has fallen into the #[non_exhaustive] wildcard, or a \
             genuine Unsupported verdict was added/removed without updating \
             this pin"
        );
    }

    /// Index-producing selection is verifiable: `Argmax` / `ArgMin` / `ArgSort`
    /// have a STATIC output shape and sound index values.
    ///
    /// `ny_propagate::layers::reduction::selection` bounds the index exactly
    /// wherever the input intervals are strictly separated (with ONNX's
    /// source-index tie rule honored by a strict-domination test) and falls
    /// back to the `[0, n-1]` index hull wherever they overlap. Hence `Sound`,
    /// not `Exact` and emphatically not `Unsupported` — the classification these
    /// carried while `ny-propagate`, `ny-build` and
    /// `ny_trace_bridge::coverage` had all treated them as sound.
    #[test]
    fn index_selection_ops_are_sound_not_unsupported() {
        // Spelled `Argmax` / `Topk` in ny-core (not `ArgMax` / `TopK`).
        for layer in [L::Argmax, L::ArgMin, L::ArgSort] {
            assert_eq!(
                soundness_class(&layer),
                SoundnessClass::Sound,
                "{layer}: exact on separated intervals, [0, n-1] index hull \
                 otherwise — sound, and not exact only because of that fallback"
            );
            assert!(is_verifiable(&layer), "{layer} must be verifiable");
        }

        let r = report();
        for name in ["Argmax", "ArgMin", "ArgSort"] {
            assert!(r.sound.contains(&name.to_string()));
            assert!(!r.unsupported.contains(&name.to_string()));
        }
    }

    /// `Topk` is classified at the WEAKER of the two forms the one `LayerType`
    /// names.
    ///
    /// `TopkLayer`'s values output is exact for the `largest && sorted`
    /// function (the j-th order statistic is monotone in every input, so the
    /// per-lane sorted endpoints are attained), but its indices output is the
    /// unconditional `[0, n-1]` hull with no exact case at all. ny-build's
    /// `read_topk_output_kind` selects between them from a layer attribute that
    /// `LayerType::Topk` does not carry, so the arm must take the loose one.
    ///
    /// This is deliberately weaker than `ny_trace_bridge::coverage`, which sees
    /// the serialized `sorted` flag and can call `Topk { sorted: true }`
    /// `Exact`. Both remain verifiable; only the tightness claim differs.
    #[test]
    fn topk_is_loose_because_the_layer_type_covers_the_indices_output() {
        assert_eq!(soundness_class(&L::Topk), SoundnessClass::SoundButLoose);
        assert!(is_verifiable(&L::Topk));

        let r = report();
        assert!(r.loose.contains(&"Topk".to_string()));
        assert!(!r.unsupported.contains(&"Topk".to_string()));
        // Not Exact: the indices form never attempts the separated-input case.
        assert!(!r.exact.contains(&"Topk".to_string()));
    }

    /// `Where` is sound but LOOSE, not unsupported.
    ///
    /// `WhereLayer::propagate_ibp_ternary` ignores its condition entirely and
    /// returns the union hull of the two branches, which encloses both at every
    /// point of the box. That is a correct over-approximation for any condition
    /// whatsoever — and deliberately loose wherever the condition is in fact
    /// decided, which is why it is `SoundButLoose` rather than `Sound` or
    /// `Exact`. Calling it `Unsupported` reported a soundness hole on every
    /// masked graph that ny does not have.
    #[test]
    fn where_is_the_sound_union_hull_not_unsupported() {
        assert_eq!(soundness_class(&L::Where), SoundnessClass::SoundButLoose);
        assert!(is_verifiable(&L::Where), "Where must be verifiable");

        let r = report();
        assert!(r.loose.contains(&"Where".to_string()));
        assert!(!r.unsupported.contains(&"Where".to_string()));
        // The union hull is never exact when the condition is decided.
        assert!(!r.exact.contains(&"Where".to_string()));
    }

    /// The indexed-write family is verifiable, and split by COMBINE semantics.
    ///
    /// All three are `SoundButLoose`, for two different reasons that the
    /// classification does not need to distinguish but the comments do:
    /// `ScatterND` OVERWRITES, so duplicate destinations are enclosed by the
    /// min/max union of the candidate writes; `ScatterAdd` / `IndexAdd`
    /// ACCUMULATE with outward-directed rounding, so duplicates are
    /// deterministic and looseness comes from the dynamic-index hull instead.
    /// The additive pair had no arm at all and were being swallowed by the
    /// wildcard.
    #[test]
    fn indexed_write_family_is_loose_not_unsupported() {
        for layer in [L::ScatterND, L::ScatterAdd, L::IndexAdd] {
            assert_eq!(
                soundness_class(&layer),
                SoundnessClass::SoundButLoose,
                "{layer}: enclosed under both static and dynamic indices, but \
                 loose in the dynamic case"
            );
            assert!(is_verifiable(&layer), "{layer} must be verifiable");
        }

        let r = report();
        for name in ["ScatterND", "ScatterAdd", "IndexAdd"] {
            assert!(r.loose.contains(&name.to_string()));
            assert!(!r.unsupported.contains(&name.to_string()));
        }
    }

    /// `Conv3d` is exact like its lower-rank siblings.
    ///
    /// `Conv3dLayer`'s only interval forward is a certified directed-binary64
    /// enclosure of the affine map, so the map is reproduced exactly. Its CROWN
    /// backward fails closed, which degrades the node to IBP — a tightness
    /// consequence, not a soundness one — so it does not move the class. It had
    /// no arm at all and was reported unverifiable by the wildcard.
    #[test]
    fn conv3d_matches_its_lower_rank_siblings() {
        assert_eq!(soundness_class(&L::Conv3d), SoundnessClass::Exact);
        assert_eq!(soundness_class(&L::Conv2d), SoundnessClass::Exact);
        assert_eq!(soundness_class(&L::Conv1d), SoundnessClass::Exact);
        assert!(is_verifiable(&L::Conv3d));

        let r = report();
        assert!(r.exact.contains(&"Conv3d".to_string()));
        assert!(!r.unsupported.contains(&"Conv3d".to_string()));
    }

    /// Sweep every variant `all_variants()` knows about and assert none is
    /// misclassified by accident — the "exhaustive in spirit" check.
    ///
    /// Two properties, both cheap and both real:
    ///
    /// * `soundness_class` is total over the list (trivially true, but it also
    ///   exercises every arm, so a `todo!`/panic in one would surface here);
    /// * `is_verifiable` agrees with `soundness_class` for every variant, and
    ///   the ONLY non-verifiable ones are the two intended ops — so no variant
    ///   can quietly become unverifiable without a test naming it.
    #[test]
    fn every_variant_is_classified_and_only_the_intended_ops_are_unverifiable() {
        for variant in all_variants() {
            let class = soundness_class(&variant);
            assert_eq!(
                is_verifiable(&variant),
                class != SoundnessClass::Unsupported,
                "{variant}: is_verifiable disagrees with soundness_class"
            );
            if class == SoundnessClass::Unsupported {
                let name = variant.to_string();
                assert!(
                    name == "NonZero" || name == "Unknown",
                    "{name} classified Unsupported without a documented reason — \
                     classify it in soundness_class, or extend the pin in \
                     unsupported_bucket_is_exactly_the_two_intended_ops"
                );
            }
        }
    }

    #[test]
    fn representative_variants_classify_as_expected() {
        // The exhaustive per-variant match guarantees totality at compile time;
        // these spot-checks pin the intended classes.
        assert_eq!(soundness_class(&L::Linear), SoundnessClass::Exact);
        assert_eq!(soundness_class(&L::Reshape), SoundnessClass::Exact);
        // EpsilonExpand's IBP is the exact interval semantics of the layer.
        assert_eq!(soundness_class(&L::EpsilonExpand), SoundnessClass::Exact);
        assert_eq!(soundness_class(&L::ReLU), SoundnessClass::Sound);
        assert_eq!(soundness_class(&L::Sigmoid), SoundnessClass::Sound);
        assert_eq!(
            soundness_class(&L::LayerNorm),
            SoundnessClass::SoundButLoose
        );
        assert_eq!(soundness_class(&L::Softmax), SoundnessClass::SoundButLoose);
        // Where is the union hull of both branches: sound, deliberately loose.
        assert_eq!(soundness_class(&L::Where), SoundnessClass::SoundButLoose);
        assert_eq!(soundness_class(&L::NonZero), SoundnessClass::Unsupported);

        assert!(is_verifiable(&L::Linear));
        assert!(is_verifiable(&L::Softmax));
    }

    /// Comparison layers are verifiable: their `{0, 1}` step semantics are
    /// exact off the threshold and the `[0, 1]` hull across it. Classifying
    /// them `Unsupported` misreported every mask-using graph (nn's kokoro
    /// voiced/unvoiced masks, for one) as unverifiable.
    #[test]
    fn comparison_layers_are_sound_not_unsupported() {
        for layer in [L::Compare, L::CompareTensor] {
            assert_eq!(soundness_class(&layer), SoundnessClass::Sound);
            assert!(is_verifiable(&layer), "{layer} must be verifiable");
        }
        let r = report();
        assert!(r.sound.contains(&"Compare".to_string()));
        assert!(r.sound.contains(&"CompareTensor".to_string()));
        assert!(!r.unsupported.contains(&"Compare".to_string()));
        assert!(!r.unsupported.contains(&"CompareTensor".to_string()));
    }

    /// `Trunc` and `Erf` were silently falling into the `#[non_exhaustive]`
    /// wildcard and being reported unverifiable despite having sound layer
    /// implementations. Pin their classes so the regression cannot recur
    /// unnoticed.
    #[test]
    fn trunc_and_erf_are_classified_not_wildcarded() {
        // Erf is monotone with the Tanh/Sigmoid convexity split.
        assert_eq!(soundness_class(&L::Erf), SoundnessClass::Sound);
        // Trunc is a piecewise-constant step, like Floor/Ceil/Round.
        assert_eq!(soundness_class(&L::Trunc), SoundnessClass::SoundButLoose);
        assert!(is_verifiable(&L::Erf));
        assert!(is_verifiable(&L::Trunc));

        let r = report();
        assert!(r.sound.contains(&"Erf".to_string()));
        assert!(r.loose.contains(&"Trunc".to_string()));
    }
}
