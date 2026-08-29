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
//!   (`exp`/`log`/`sqrt`/`div`) whose interval relaxations
//!   widen quickly. Still safe to verify with; just weaker. The extreme case
//!   is `Custom`, the trace format's explicit opaque escape hatch: lowered to
//!   the graph builder's `OpaqueSkip` substitution, whose `[-inf, +inf]`
//!   bounds are a *vacuous but genuinely sound* over-approximation of any op
//!   (never an identity) — verification succeeds only if nothing downstream
//!   depends on that value.
//! - [`BridgeSoundness::Unsupported`] — the op's output *shape, routing, or
//!   source-coordinate mapping* depends on tensor **values** (`Scatter`,
//!   `IndexPut`, `MoeGating`, `GridSample` — though the last three rows below
//!   spell out why two of those are context-free residues and one is not),
//!   required semantics are absent from
//!   the *serialized op itself*, or no current lowering preserves them. The
//!   bridge must **refuse** these rather than emit a vacuous/incorrect layer
//!   (sound by construction); they are handled by graph segmentation
//!   upstream, not by direct translation. The two "absent semantics" rows are
//!   worth spelling out, because neither is a claim that the op is
//!   intrinsically unlowerable:
//!
//!   - `ToDtype` — the *source* dtype is not part of the op, so this
//!     context-free taxonomy cannot tell a lowerable cast from a lossy one
//!     and keeps the whole variant `Unsupported`. `translate` is not
//!     context-free: it resolves the producer's recorded dtype
//!     graph-contextually and lowers three families — the exact
//!     `F32 -> F32` identity, the lossless `U8`/`Bool -> F32` embedding, and
//!     `F32 -> I32`/`I64` as the guarded round-toward-zero `Trunc` (the same
//!     layer ny-onnx emits for the identical ONNX `Cast`, carrying a
//!     finite/in-range proof obligation) — refusing every other resolved
//!     combination.
//!   - `Scatter` and `IndexPut` — both *overwrite*, so their duplicate
//!     destinations are last-writer-nondeterministic and a sound lowering must
//!     enclose the union of the candidate writes, which requires knowing which
//!     destinations collide. That is a property of the INDEX OPERAND rather
//!     than of the variant, so this context-free taxonomy has to assume the
//!     runtime case and keeps both variants `Unsupported`. `translate` lowers a
//!     translate-time-constant index to `ScatterND` — whose duplicate min/max
//!     combine IS that union — and refuses only the runtime index. (They differ
//!     only in the index's shape: `Scatter`'s names a destination per element,
//!     `IndexPut`'s is 1-D and names one per slice.) Their additive siblings
//!     `ScatterAdd` / `IndexAdd` are NOT in this bucket: accumulation is
//!     order-independent and has a sound rule for runtime indices too, so they
//!     classify `SoundButLoose`.
//!   - `GridSample` — refused twice over, which is why it is not merely the
//!     "value-dependent coordinates" row it looks like. A runtime grid is the
//!     structural case (the output-to-input map is a runtime value). A
//!     translate-time-CONSTANT grid is refused as well, on arithmetic grounds:
//!     grid sampling is not separable, so a constant resampling matrix must
//!     fold `fl(cy * cx)` into one coefficient where nn's kernel evaluates
//!     `fl(fl(v * cy) * cx)` — up to 2 ULP apart, wider than the tightest sound
//!     interval a `Linear` layer can produce for one product, so nn's own value
//!     can fall outside the emitted bounds. Unlike `Scatter`/`IndexPut`, then,
//!     this row is NOT a context-free residue: no operand context rescues it.
//!     `ResizeBilinear` is lowered precisely because its per-axis
//!     factorization lets the matrix carry nn's coefficients verbatim.
//!
//!   `Topk` used to be in this bucket, for the wire's missing `largest` /
//!   `sorted` flags. It has LEFT: nn-core serializes both now, so the traced
//!   values tensor names exactly one of the four functions and the taxonomy
//!   classifies on the flags — `sorted = true` is exact (monotone order
//!   statistics), `sorted = false` is sound-but-loose (the hull over the k
//!   selected values). `translate` still emits only `largest && sorted`, the
//!   one combination `TopkLayer` models, but the other three are now
//!   MISSING-LAYER refusals rather than unreadable-wire refusals — hence
//!   degradable.
//!
//!   `Powf` appears in this bucket only as a fail-closed RESIDUE: every
//!   finite exponent within the f64->f32 narrowing cap classifies `Sound`
//!   above, and only non-finite / beyond-cap exponents fall through.
//!
//! [`coverage`] reports the four buckets over a canonical const list of every
//! variant's name, and the `tests` module asserts that list partitions all 123
//! variants — pinning the catalogue against silent drift from the enum.
//!
//! ## Refusal classes
//!
//! The [`Unsupported`](BridgeSoundness::Unsupported) / everything-else split is
//! also the line a PERMISSIVE translation must respect, so it is exposed
//! directly as [`RefusalClass`] via [`refusal_class`]: an `Unsupported` op's
//! refusal is a [`RefusalClass::Soundness`] refusal that stays hard in every
//! mode, while a refusal of any other op is [`RefusalClass::Unported`] and may
//! be degraded to the vacuous `OpaqueSkip` carrier when the caller opts in.
//! See [`crate::translate::UnportedOpPolicy`].

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
        // Roll: chained per-(shift, dim) `2 x Slice(tail, head) + Concat`
        // rotations, byte-matching nn's own narrow/narrow/cat evaluation
        // (including the `((s % n) + n) % n` normalization and the
        // size-0 / zero-shift skips). Pure reindexing => IBP is exact.
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
        | TraceOp::ReflectionPad1d { .. }
        | TraceOp::ReflectionPad2d { .. }
        | TraceOp::ConstantPadNd { .. }
        | TraceOp::Arange { .. }
        | TraceOp::Dropout
        // RepeatInterleave: pure selection driven by TRANSLATE-TIME-CONSTANT
        // counts (the wire records them as input 1, a ConstantWeight whose
        // payload the translator reads). All-ones lowers to an `Add + 0`
        // identity, a uniform count to Reshape+Tile+Reshape, and the general
        // case to Slice-per-run + Concat — every one a pure reindexing, so
        // IBP is exact. Like `Gather`, the CONDITIONAL refusals stay inside
        // `translate` (absent or non-constant counts => data-dependent output
        // shape; fan-in beyond the decomposition cap), which is exactly the
        // narrower-than-taxonomy contract the module docs describe.
        | TraceOp::RepeatInterleave { .. }
        // Sort: lowered to the sorted VALUES tensor (nn traces only the
        // values on both its CPU and GPU paths). Order statistics are
        // monotone per coordinate, so the interval image is exact and — the
        // load-bearing part — TIE-AGNOSTIC: which of two equal elements the
        // unspecified `sort_unstable` order picks cannot change the value
        // sequence. (Contrast `ArgSort`, whose INDICES are tie-sensitive.)
        | TraceOp::Sort { .. }
        // Topk with `sorted = true`: the p-th order statistic is monotone in
        // every input, so bounding it by the p-th order statistic of each
        // bound lane is EXACT — whichever end of the lane `largest` names.
        // `largest = true` is what `translate` emits today (`TopkLayer` models
        // that one); `largest = false` is exact by the mirror-image argument
        // and refused only for the missing k-smallest layer, which is why it
        // classifies here rather than as `Unsupported` — the trace is
        // unambiguous, so a permissive caller may degrade it. `sorted = false`
        // falls to `SoundButLoose` below.
        | TraceOp::Topk { sorted: true, .. }
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
        // Conv3d: an affine map like its lower-rank siblings, and no longer a
        // classification that outruns the translator — `ny_core::LayerType`
        // has a `Conv3d` variant, ny-propagate has `Conv3dLayer`, ny-build has
        // the convert arm, and `translate` emits the spec. IBP-only: the
        // layer's CROWN backward fails closed, so a Conv3d node degrades the
        // network to IBP at that node (an ENCLOSURE question, not a
        // classification one — the affine map itself is reproduced exactly).
        | TraceOp::Conv3d { .. }
        | TraceOp::ConvTranspose1d { .. }
        | TraceOp::ConvTranspose2d { .. }
        | TraceOp::QLinear { .. }
        // Cumsum: native `CumsumLayer` (#3919) — monotone => exact IBP with
        // outward rounding, O(T) CROWN backward, no size cap.
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
        | TraceOp::Clamp { .. }
        | TraceOp::MaxPool1d { .. }
        | TraceOp::MaxPool2d { .. }
        | TraceOp::AdaptiveMaxPool2d { .. } => Sound,

        // Powf: exact special cases (identity/sqrt/sqr/recip) plus the
        // general `PowConstantLayer` lowering — sound (fail-closed on invalid
        // domains) for every finite exponent within the f64->f32 narrowing
        // cap shared with `translate` ([`crate::translate::MAX_POW_EXPONENT_ABS`]).
        TraceOp::Powf { exponent }
            if exponent.is_finite()
                && exponent.abs() <= crate::translate::MAX_POW_EXPONENT_ABS =>
        {
            Sound
        }

        // Comparisons: lowered to Compare/CompareTensor layers whose
        // {0,1}-interval IBP is exact off the threshold and the [0,1] hull
        // across it — sound, and as tight as intervals allow (IBP-only; a
        // step has no useful CROWN linear relaxation).
        TraceOp::Compare { .. } | TraceOp::CompareTensor { .. } => Sound,

        // Index-producing selection: `Argmax` / `Argmin` / `ArgSort` lower to
        // the native index layers, whose IBP is EXACT wherever the input
        // intervals are strictly separated (the winner — or the whole
        // permutation — is then determined for every point in the box) and
        // falls back to the sound `[0, n-1]` index hull wherever they
        // overlap. Not `Exact`, precisely because of that overlap fallback.
        //
        // The index values themselves are modeled as exact f32 integers,
        // widened OUTWARD by two endpoints above 2²⁴ where consecutive
        // integers stop being representable (ny-propagate
        // `reduction/selection.rs::index_interval`) — never snapped to a
        // wrong integer.
        TraceOp::Argmax { .. } | TraceOp::Argmin { .. } | TraceOp::ArgSort { .. } => Sound,

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
        // Sdpa / SdpaCausal: emitted as ONE native `SelfAttention` layer
        // (not a MatMul + Softmax decomposition) — the softmax interval
        // relaxation is what keeps them loose, decomposition or not.
        | TraceOp::Sdpa { .. }
        | TraceOp::SdpaCausal { .. }
        | TraceOp::RotaryEmbedding { .. }
        // MultiHeadAttention: the attention MATH has the same sound-but-loose
        // relaxation as Sdpa — and `translate` now actually emits it. The
        // WIRE-FORMAT gap that used to force an unconditional refusal here
        // (no projection weights on the wire) is closed: nn-core serializes
        // `scale` and all four projections, so the composite lowers to those
        // four Linears around the same native fused layer `Sdpa` emits.
        // Masked, RoPE-fused and grouped-query nodes are still refused for
        // MISSING LAYERS rather than missing wire fields, which is why the
        // variant classifies here and the refusals stay degradable.
        // (Consumers triage against `translate`, not this taxonomy, for what
        // actually lowers today — see the module docs.)
        | TraceOp::MultiHeadAttention { .. }
        | TraceOp::Lstm { .. }
        | TraceOp::AvgPool1d { .. }
        | TraceOp::AvgPool2d { .. }
        | TraceOp::AdaptiveAvgPool1d { .. }
        | TraceOp::AdaptiveAvgPool2d { .. }
        | TraceOp::KokoroFused(_)
        // Fract: Sub(x, Floor(x)) — NN's floor convention (range [0, 1)),
        // NOT the trunc identity, which differs by 1 for every negative
        // non-integer and produced a non-enclosing bound. Exact within one
        // unit period, loose across period boundaries (the Sub decorrelates
        // x from floor(x)).
        | TraceOp::Fract
        // Embedding: index-agnostic vocabulary-hull surrogate (midpoint
        // Linear + per-dim EpsilonExpand) — a correct enclosure of every
        // row, deliberately loose (independent of the actual index).
        | TraceOp::Embedding { .. }
        // WhereCond: a 3-input conditional select lowered to the native
        // `Where` layer, whose IBP is the UNION HULL of the two branches
        // wherever the condition interval straddles (and the selected branch
        // exactly wherever the condition's sign is decidable). A correct
        // over-approximation, deliberately loose across the straddle, and
        // IBP-only — a select has no useful CROWN linear relaxation.
        | TraceOp::WhereCond
        // Upsample2d: `Nearest` is a pure reindex (Reshape + Tile + Reshape);
        // `Bilinear` is now lowered too, to the SAME separated constant-Linear
        // resampler `ResizeBilinear` uses, because `align_corners` rides the
        // wire and pins which of nn's two coordinate maps ran. The variant sits
        // in this bucket for the bilinear passes' cross-axis decorrelation (see
        // ResizeBilinear below) and for `Bicubic`, which has no nn kernel to
        // replicate and so is refused — degradably, since nothing can record it.
        | TraceOp::Upsample2d { .. }
        // ResizeBilinear: a CONSTANT row-stochastic linear map. `target_h` /
        // `target_w` ride the wire and nn's `resize_bilinear` has exactly one
        // coordinate convention (half-pixel + clamp; no align_corners knob
        // exists on that path), so the source cells and interpolation weights
        // are fully determined at translate time. Lowered to a separated pair
        // of constant `Linear` resampling matrices, which IBP propagates
        // exactly per pass; "loose" only because chaining the two passes
        // decorrelates the shared source cells.
        | TraceOp::ResizeBilinear { .. }
        // ScatterAdd / IndexAdd: additive indexed accumulation, lowered to
        // ny-propagate's `ScatterAddLayer` / `IndexAddLayer`. Unlike the
        // overwriting `Scatter`, these are sound for EVERY index — constant or
        // runtime — which is why they classify here rather than as a
        // context-free residue:
        //
        //   - Duplicate destinations are not ambiguous. `+=` over the reals is
        //     associative and commutative, so a cell named by several index
        //     entries receives the SUM of the contributions under every visit
        //     order: one value, no union needed.
        //   - A constant index gives the EXACT interval image (accumulate the
        //     lower bounds into the lower output, the upper into the upper).
        //   - A runtime index never has to be read: the layers' dynamic rule
        //     widens `data` by the total possible additive contribution of
        //     `src` (the sum of its negative lower bounds below, of its
        //     positive upper bounds above). Every index assignment routes each
        //     src element to exactly one cell, so every realizable output lies
        //     in that hull.
        //
        // "Loose" names that dynamic case — and the constant case's own
        // decorrelation when `data` and `src` share upstream variables. NN's
        // `Add(x, 0)` identity lowering, which the old refusal cited, was
        // never sound and is not what the bridge emits.
        | TraceOp::ScatterAdd { .. }
        | TraceOp::IndexAdd { .. }
        // Topk with `sorted = false`: the k selected values in an UNSPECIFIED
        // order, so output position j may hold any of them and a sound bound
        // must be the hull over all k — correct, but strictly looser than the
        // per-position order statistic the sorted form gets. No layer emits
        // that hull today, so `translate` refuses it; the refusal is
        // degradable, because the wire says unambiguously which function ran.
        | TraceOp::Topk { sorted: false, .. } => SoundButLoose,

        // `Custom` — the explicit opaque escape hatch — is the maximally
        // loose case: lowered to the graph builder's OpaqueSkip substitution,
        // a vacuous-but-sound `[-inf, +inf]` over-approximation of any op
        // (INC-FINAL reconciliation; see translate/mod.rs Custom arm and
        // ny-propagate layers/misc/skip_merge.rs for the verified rule).
        TraceOp::Custom { .. } => SoundButLoose,

        // -- Unsupported: missing or data-dependent semantics -----------------
        // Refuse any op whose required routing/parameters are value-dependent
        // or absent from the serialized trace, and any op without a semantics-
        // preserving lowering.

        // Value-dependent write routing / gating: the destination indices (or
        // expert assignment) are runtime values, so no static layer encloses
        // the result and nn's identity passthrough is not a sound
        // over-approximation.
        //
        // `Scatter` and `IndexPut` are CONTEXT-FREE RESIDUES of that rule,
        // exactly like `ToDtype` below — not a claim that either op is
        // unlowerable. Both OVERWRITE, so their duplicate destinations are
        // last-writer-nondeterministic (nn's CPU `scatter_loop` walks src
        // row-major while `scatter_gpu` races parallel writes; `index_put` is
        // documented last-write-wins, and nn-import emits it for
        // `aten.index_put` with `accumulate = false`, which PyTorch documents
        // as nondeterministic on duplicates). A sound lowering must therefore
        // enclose the UNION of the candidate writes, which in turn requires
        // knowing which destinations collide. Whether that is knowable is a
        // property of the INDEX OPERAND, not of the variant: `translate` is not
        // context-free and lowers a translate-time-constant index of EITHER op
        // to `ScatterND` (whose duplicate min/max combine IS that union),
        // refusing only the runtime index this taxonomy has to assume.
        //
        // NOT here any more: `ScatterAdd` / `IndexAdd`, whose accumulation is
        // order-independent AND has a sound dynamic-index rule — see the
        // SoundButLoose arm.
        | TraceOp::Scatter { .. }
        | TraceOp::IndexPut { .. }
        | TraceOp::MoeGating { .. }
        // ToDtype: context-free refusal — the op records only its TARGET
        // dtype, so this classifier cannot distinguish the identity and
        // lossless-embedding casts `translate` lowers (`F32 -> F32`,
        // `U8`/`Bool -> F32`) from the lossy ones it refuses. See the module
        // docs; `translate` resolves the source dtype from the producer node.
        | TraceOp::ToDtype { .. }
        // Powf: fail-closed RESIDUE only. Every finite exponent within
        // `MAX_POW_EXPONENT_ABS` classified `Sound` in the guarded arm above;
        // this arm catches NaN / +-inf / beyond-cap exponents, where the
        // f64 -> f32 narrowing could flip the integrality or parity that
        // `PowConstantLayer`'s sign-aware branches key on.
        | TraceOp::Powf { .. }
        // GridSample: unlike `ResizeBilinear` (now lowered — see the
        // SoundButLoose arm), its source coordinates come from the `grid`
        // TENSOR OPERAND, so the output-to-input map is generally a function of
        // a runtime value and no static layer encloses it. Its serialized
        // conventions (`padding_mode`, `align_corners`) are complete, so the
        // refusal is never about conventions.
        //
        // And unlike `Scatter`/`IndexPut` above, this is NOT a context-free
        // residue: a translate-time-CONSTANT grid does not rescue it either.
        // Grid sampling is not separable (`ix` and `iy` vary independently per
        // output cell), so a constant resampling matrix has to carry the
        // product `fl(cy * cx)` as one coefficient while nn's kernel evaluates
        // `fl(fl(v * cy) * cx)` — up to 2 ULP apart, where the tightest sound
        // interval a Linear layer can give for a single product spans 1 ULP
        // either side. nn's own value can then fall OUTSIDE the emitted bounds:
        // a silent non-enclosure, not a looseness. `translate` refuses both
        // cases with distinct messages; the witness is pinned in
        // `ops_pad_resample::tests::grid_sample_single_matrix_lowering_would_not_enclose_nn`.
        | TraceOp::GridSample { .. }
        // SegmentBoundary is a pipeline marker rather than a computation
        // (upstream segmentation consumes it).
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

/// Why [`crate::translate`] refused a node — and, decisively, whether a
/// PERMISSIVE translation is allowed to degrade it to the vacuous
/// `OpaqueSkip` carrier instead of failing the whole graph.
///
/// The two classes look identical at the `NyError::UnsupportedOp` boundary,
/// which is exactly why they are separated here rather than by inspecting
/// error text. See [`crate::translate::UnportedOpPolicy`] for the policy that
/// consumes this.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RefusalClass {
    /// A sound lowering EXISTS in principle; the bridge simply has not ported
    /// this op (or this configuration of it) yet.
    ///
    /// Degradable: substituting `[-inf, +inf]` for the node loses all
    /// precision there but is a genuine over-approximation of whatever the op
    /// computes, so a permissive caller may opt into it.
    Unported,
    /// The op is refused for SOUNDNESS: its output shape, write routing, or
    /// source-coordinate mapping is a function of tensor VALUES; or the
    /// serialized op is semantically ambiguous; or its endpoint's rounding
    /// behaviour is unmodeled (a lossy `ToDtype`, a beyond-cap `Powf`).
    ///
    /// `Topk` used to be the headline example of the "ambiguous wire" case and
    /// is deliberately no longer here: nn serializes `largest`/`sorted`, so its
    /// remaining refusals are missing-layer refusals and DO degrade.
    ///
    /// NOT degradable, in either mode. `OpaqueSkip` would be sound for the
    /// node's own value, but these refusals report that the TRACE cannot be
    /// read unambiguously in the first place — the ambiguity is not confined
    /// to one node's bounds, and hiding it behind a warning is exactly the
    /// silent-degradation failure the bridge is built to prevent.
    Soundness,
}

impl RefusalClass {
    /// Whether a permissive translation may substitute the vacuous
    /// `OpaqueSkip` carrier for a node refused with this class.
    #[must_use]
    pub const fn is_degradable(self) -> bool {
        matches!(self, Self::Unported)
    }

    /// Short, stable lowercase label (matches the serde representation).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Unported => "unported",
            Self::Soundness => "soundness",
        }
    }
}

/// Classify a refusal of `op` into the two classes a permissive translation
/// must tell apart.
///
/// The soundness taxonomy already draws exactly this line:
/// [`BridgeSoundness::Unsupported`] is precisely the bucket of ops that must
/// ALWAYS be refused, so it maps to [`RefusalClass::Soundness`] and every other
/// bucket — ops for which a sound lowering exists but may not be implemented —
/// maps to [`RefusalClass::Unported`]. Deriving the class from the taxonomy
/// rather than restating it means a newly classified `TraceOp` variant (which
/// the exhaustive match in [`trace_op_soundness`] forces someone to classify)
/// automatically lands on the right side of the permissive boundary.
///
/// Note the context-free residues named in the module docs land on the SAFE
/// side by construction: `ToDtype`, `Scatter` and a beyond-cap `Powf` classify
/// `Unsupported`, so a refusal of one of them stays hard even in permissive
/// mode — which is the intended behaviour for all three, since each refusal is
/// about semantics the bridge cannot read rather than a missing port.
#[must_use]
pub fn refusal_class(op: &TraceOp) -> RefusalClass {
    if trace_op_soundness(op).is_translatable() {
        RefusalClass::Unported
    } else {
        RefusalClass::Soundness
    }
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

/// A representative fused `MultiHeadAttention` node: maskless, RoPE-free,
/// non-GQA self-attention — the shape `translate` lowers.
///
/// Shared by the catalogue and the refusal-class tests so the twelve-field
/// payload is spelled out once.
fn mha_witness() -> TraceOp {
    let w = || crate::schema::WeightPayload::f32(vec![0.0], vec![1]);
    TraceOp::MultiHeadAttention {
        num_heads: 2,
        num_kv_heads: 2,
        head_dim: 2,
        scale: 0.5,
        q_weight: w(),
        q_bias: None,
        k_weight: w(),
        k_bias: None,
        v_weight: w(),
        v_bias: None,
        out_weight: w(),
        out_bias: None,
        rope: None,
        cross_attention: false,
        has_mask: false,
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
        ("MultiHeadAttention", mha_witness()),
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
                align_corners: false,
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
        (
            "Topk",
            TraceOp::Topk {
                k: 1,
                dim: 0,
                largest: true,
                sorted: true,
            },
        ),
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
    use crate::schema::{DType, WeightPayload};
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

    /// The permissive boundary is derived from the taxonomy, not restated
    /// alongside it: for EVERY variant, `Unsupported` <=> a non-degradable
    /// `Soundness` refusal. A newly added variant is forced through the
    /// exhaustive match in `trace_op_soundness`, so it lands on the right side
    /// of this line automatically.
    #[test]
    fn refusal_class_partitions_the_catalogue_by_translatability() {
        for (name, op) in canonical_trace_ops() {
            let class = refusal_class(&op);
            assert_eq!(
                class == RefusalClass::Soundness,
                trace_op_soundness(&op) == BridgeSoundness::Unsupported,
                "refusal class disagrees with the soundness bucket for {name}"
            );
            assert_eq!(
                class.is_degradable(),
                is_translatable(&op),
                "only an op with a sound lowering in principle may be degraded ({name})"
            );
        }
    }

    /// The ops the permissive translate mode must NEVER degrade, named
    /// explicitly so a reclassification cannot quietly open the gate:
    /// data-dependent routing/coordinates, unmodeled cast and exponent
    /// endpoints, and the pipeline marker.
    ///
    /// `Topk` used to head this list for its ambiguous wire semantics and is
    /// deliberately gone — the flags are serialized now, so its remaining
    /// refusals are missing-layer refusals and DO degrade
    /// (`topk_is_classified_by_the_flags_now_that_the_wire_carries_them`).
    #[test]
    fn soundness_refusals_are_never_degradable() {
        for op in [
            TraceOp::Scatter { dim: 0 },
            TraceOp::IndexPut { dim: 0 },
            TraceOp::MoeGating {
                num_experts: 2,
                top_k: 1,
            },
            TraceOp::GridSample {
                padding_mode: crate::schema::GridSamplePaddingMode::Zeros,
                align_corners: false,
            },
            TraceOp::ToDtype {
                target_dtype: DType::F16,
            },
            TraceOp::Powf {
                exponent: f64::INFINITY,
            },
            TraceOp::SegmentBoundary {
                reason: "boundary".to_string(),
                input_bounds: None,
            },
        ] {
            assert_eq!(
                refusal_class(&op),
                RefusalClass::Soundness,
                "{op:?} must stay a hard refusal in every translate mode"
            );
            assert!(!refusal_class(&op).is_degradable(), "{op:?}");
        }
    }

    /// A representative unported-but-lowerable op is degradable — the case the
    /// permissive mode exists for. `Custom` is degradable too, which is merely
    /// consistent: it already lowers to the same carrier unconditionally.
    #[test]
    fn unported_but_lowerable_ops_are_degradable() {
        for op in [
            TraceOp::Conv3d {
                weight: WeightPayload::f32(vec![0.0], vec![1]),
                bias: None,
                padding: [0, 0, 0],
                stride: [1, 1, 1],
                dilation: [1, 1, 1],
                groups: 1,
            },
            mha_witness(),
            TraceOp::Custom {
                name: "opaque".to_string(),
            },
        ] {
            assert_eq!(refusal_class(&op), RefusalClass::Unported, "{op:?}");
            assert!(refusal_class(&op).is_degradable(), "{op:?}");
        }
    }

    #[test]
    fn refusal_class_labels_round_trip_json() {
        for class in [RefusalClass::Unported, RefusalClass::Soundness] {
            let json = serde_json::to_string(&class).expect("serialize");
            assert_eq!(json, format!("\"{}\"", class.as_str()));
            let back: RefusalClass = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(back, class);
        }
    }

    #[test]
    fn data_dependent_ops_are_unsupported() {
        // The roadmap-pinned data-dependent / semantics-absent set MUST be
        // Unsupported so the bridge refuses them (sound by construction).
        //
        // NOT in this list any more, deliberately: `Argmax` / `Argmin` /
        // `ArgSort` / `Sort` (statically shaped selection with sound index /
        // exact value semantics), `WhereCond` (union-hull select),
        // `RepeatInterleave` (translate-time-constant counts), and
        // `ResizeBilinear` (constant resampling matrices — see
        // `resize_bilinear_is_a_constant_map_grid_sample_is_not`). `Topk` has
        // ALSO left: nn serializes `largest`/`sorted`, so the wire names one
        // function and the variant classifies on the flags — see
        // `topk_is_classified_by_the_flags_now_that_the_wire_carries_them`.
        //
        // `ScatterAdd` / `IndexAdd` also left: their accumulation is
        // order-independent and their dynamic-index rule never reads the index
        // at all, so they are sound for EVERY index. `Scatter` and `IndexPut`
        // stay, but as context-free residues — see
        // `scatter_overwrite_refusal_is_context_free_its_additive_siblings_are_not`.
        // `GridSample` stays and is NOT a residue: no operand context rescues
        // it — see `grid_sample_is_refused_even_when_its_grid_is_constant`.
        let must_be_unsupported = [
            TraceOp::Scatter { dim: 0 },
            TraceOp::IndexPut { dim: 0 },
            TraceOp::MoeGating {
                num_experts: 1,
                top_k: 1,
            },
            TraceOp::GridSample {
                padding_mode: crate::schema::GridSamplePaddingMode::Zeros,
                align_corners: false,
            },
            TraceOp::SegmentBoundary {
                reason: "x".into(),
                input_bounds: None,
            },
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

    /// The scatter family splits on WHETHER DUPLICATE DESTINATIONS ARE
    /// AMBIGUOUS, not on the word "scatter".
    ///
    /// `Scatter` and `IndexPut` overwrite, so a cell named twice keeps
    /// whichever write landed last — an order the wire does not fix (nn's CPU
    /// `scatter_loop` walks src row-major while `scatter_gpu` races parallel
    /// writes; `index_put` is documented last-write-wins, and nn-import emits
    /// it for `aten.index_put` with `accumulate = false`, which PyTorch
    /// documents as nondeterministic on duplicates). Enclosing that needs the
    /// collision pattern, which is a property of the index operand, so this
    /// context-free taxonomy must assume the runtime case (the `ToDtype`
    /// precedent: `translate` is not context-free and does lower the
    /// constant-index instance of BOTH).
    ///
    /// `ScatterAdd` / `IndexAdd` accumulate, and `+=` is associative and
    /// commutative — a cell named twice receives the SUM under every order.
    /// Their dynamic-index rule never reads the index at all, so no operand
    /// context is needed for them to be sound.
    #[test]
    fn scatter_overwrite_refusal_is_context_free_its_additive_siblings_are_not() {
        for overwrite in [TraceOp::Scatter { dim: 0 }, TraceOp::IndexPut { dim: 0 }] {
            assert_eq!(
                trace_op_soundness(&overwrite),
                BridgeSoundness::Unsupported,
                "{overwrite:?} overwrites and its duplicate-write order is unfixed, so the \
                 context-free taxonomy must assume a runtime index"
            );
        }
        for additive in [TraceOp::ScatterAdd { dim: 0 }, TraceOp::IndexAdd { dim: 0 }] {
            assert_eq!(
                trace_op_soundness(&additive),
                BridgeSoundness::SoundButLoose,
                "{additive:?} accumulates, so duplicate destinations sum deterministically and \
                 even a runtime index has a sound (loose) rule"
            );
            assert!(is_translatable(&additive));
        }
    }

    /// The resampling pair splits on WHERE the source coordinates come from,
    /// not on how hard interpolation is: `ResizeBilinear`'s come from
    /// serialized target extents (a constant map, lowerable), `GridSample`'s
    /// come from a tensor operand that is generally a runtime value
    /// (value-dependent, refused). See
    /// `grid_sample_is_refused_even_when_its_grid_is_constant` for the second,
    /// independent reason that keeps the row `Unsupported` when it is not.
    #[test]
    fn resize_bilinear_is_a_constant_map_grid_sample_is_not() {
        let resize = TraceOp::ResizeBilinear {
            target_h: 7,
            target_w: 3,
        };
        assert_eq!(trace_op_soundness(&resize), BridgeSoundness::SoundButLoose);
        assert!(is_translatable(&resize));

        for align_corners in [false, true] {
            for padding_mode in [
                crate::schema::GridSamplePaddingMode::Zeros,
                crate::schema::GridSamplePaddingMode::Border,
            ] {
                // Both convention flags ARE serialized — the refusal is not
                // about conventions, so no flag combination rescues it.
                let grid = TraceOp::GridSample {
                    padding_mode,
                    align_corners,
                };
                assert_eq!(trace_op_soundness(&grid), BridgeSoundness::Unsupported);
                assert!(!is_translatable(&grid));
            }
        }
    }

    /// `GridSample` keeps its `Unsupported` row for a reason NO operand
    /// context can lift, which is what separates it from the `Scatter` /
    /// `IndexPut` / `ToDtype` residues one arm above.
    ///
    /// Those three are refused here only because this classifier cannot see
    /// the operand; `translate` can, and lowers their favourable cases.
    /// `GridSample` has a favourable-looking case too — a translate-time
    /// constant grid — and `translate` refuses it anyway: grid sampling is not
    /// separable, so a constant resampling matrix must fold `fl(cy * cx)` into
    /// one coefficient where nn's kernel evaluates `fl(fl(v * cy) * cx)`, up to
    /// 2 ULP away from what any sound `Linear` layer would bound. The witness
    /// lives with the lowering it disqualifies
    /// (`translate::ops_pad_resample::tests::grid_sample_single_matrix_lowering_would_not_enclose_nn`);
    /// this test pins the CLASSIFICATION consequence, so nobody reclassifies
    /// the row by reasoning "the grid could be constant".
    #[test]
    fn grid_sample_is_refused_even_when_its_grid_is_constant() {
        let grid = TraceOp::GridSample {
            padding_mode: crate::schema::GridSamplePaddingMode::Zeros,
            align_corners: false,
        };
        assert_eq!(trace_op_soundness(&grid), BridgeSoundness::Unsupported);
        assert_eq!(refusal_class(&grid), RefusalClass::Soundness);

        // The contrast, asserted rather than described: its residue neighbours
        // sit in the same bucket for a reason `translate` DOES lift, and their
        // constant-index lowerings are exercised in
        // `translate::ops_misc::tests`.
        for residue in [TraceOp::Scatter { dim: 0 }, TraceOp::IndexPut { dim: 0 }] {
            assert_eq!(
                trace_op_soundness(&residue),
                BridgeSoundness::Unsupported,
                "{residue:?} shares the bucket — but not the reason"
            );
        }
    }

    /// The selection family is no longer a blanket refusal: the index
    /// producers are `Sound` (exact under interval separation, `[0, n-1]`
    /// hull otherwise) and `Sort`'s VALUES tensor is `Exact` (monotone,
    /// tie-agnostic order statistics).
    #[test]
    fn selection_family_is_classified_by_what_it_emits() {
        for op in [
            TraceOp::Argmax { dim: 0 },
            TraceOp::Argmin { dim: 0 },
            TraceOp::ArgSort {
                dim: 0,
                descending: true,
            },
        ] {
            assert_eq!(
                trace_op_soundness(&op),
                BridgeSoundness::Sound,
                "{op:?} emits indices: exact when separated, [0, n-1] hull otherwise"
            );
            assert!(is_translatable(&op));
        }

        let sort = TraceOp::Sort {
            dim: 0,
            descending: false,
        };
        assert_eq!(
            trace_op_soundness(&sort),
            BridgeSoundness::Exact,
            "Sort emits the values tensor: order statistics are monotone and tie-agnostic"
        );
        assert!(is_translatable(&sort));
    }

    /// `Topk` is no longer the one refused selection op, and the taxonomy now
    /// reads the FLAGS rather than shrugging at the variant.
    ///
    /// The predecessor of this test asserted `Topk` was `Unsupported` for every
    /// payload, because the wire withheld `largest`/`sorted` and no payload
    /// could rescue it. nn serializes both now, so:
    ///
    /// * `sorted = true` is `Exact` — the p-th order statistic is monotone in
    ///   every input, so bounding it by the p-th order statistic of each bound
    ///   lane is attained at the box corners, whichever end `largest` names.
    /// * `sorted = false` is `SoundButLoose` — an unspecified output order
    ///   needs the hull over all k selected values.
    ///
    /// Neither is `Unsupported`, so BOTH are degradable: `translate` still
    /// emits only `largest && sorted` (the one function `TopkLayer` models),
    /// but the other three are missing-LAYER refusals over an unambiguous
    /// wire, which a permissive caller may skip.
    #[test]
    fn topk_is_classified_by_the_flags_now_that_the_wire_carries_them() {
        for (k, dim) in [(1usize, 0usize), (3, 1), (8, 2)] {
            for largest in [true, false] {
                let sorted_op = TraceOp::Topk {
                    k,
                    dim,
                    largest,
                    sorted: true,
                };
                assert_eq!(
                    trace_op_soundness(&sorted_op),
                    BridgeSoundness::Exact,
                    "monotone order statistics: {sorted_op:?}"
                );

                let unsorted_op = TraceOp::Topk {
                    k,
                    dim,
                    largest,
                    sorted: false,
                };
                assert_eq!(
                    trace_op_soundness(&unsorted_op),
                    BridgeSoundness::SoundButLoose,
                    "unspecified order needs the hull over k: {unsorted_op:?}"
                );

                // The decisive consequence: neither is a soundness refusal any
                // more, so permissive mode may degrade a Topk it cannot lower.
                for op in [sorted_op, unsorted_op] {
                    assert!(is_translatable(&op), "{op:?}");
                    assert_eq!(refusal_class(&op), RefusalClass::Unported, "{op:?}");
                    assert!(refusal_class(&op).is_degradable(), "{op:?}");
                }
            }
        }

        // Its statically-shaped selection siblings stay translatable.
        assert!(is_translatable(&TraceOp::Argmax { dim: 0 }));
        assert!(is_translatable(&TraceOp::Sort {
            dim: 0,
            descending: false
        }));
    }

    /// `WhereCond` lowers to the native `Where` layer's union-hull IBP — a
    /// correct but knowingly loose over-approximation, not a refusal.
    #[test]
    fn where_cond_is_sound_but_loose() {
        let op = TraceOp::WhereCond;
        assert_eq!(trace_op_soundness(&op), BridgeSoundness::SoundButLoose);
        assert!(is_translatable(&op));
    }

    /// `RepeatInterleave` is pure selection from translate-time-constant
    /// counts, so it classifies `Exact` — its remaining refusals are
    /// conditional and live in `translate`, exactly like `Gather`'s.
    #[test]
    fn repeat_interleave_is_exact_like_gather() {
        assert_eq!(
            trace_op_soundness(&TraceOp::RepeatInterleave { dim: 0 }),
            BridgeSoundness::Exact
        );
        assert_eq!(
            trace_op_soundness(&TraceOp::Gather { dim: 0 }),
            BridgeSoundness::Exact,
            "the precedent this classification follows"
        );
        assert!(is_translatable(&TraceOp::RepeatInterleave { dim: 0 }));
    }

    #[test]
    fn dtype_cast_soundness_is_fail_closed() {
        for target_dtype in [
            DType::F32,
            DType::F64,
            DType::F16,
            DType::Bf16,
            DType::I32,
            DType::I64,
            DType::U32,
            DType::U8,
            DType::Bool,
        ] {
            let op = TraceOp::ToDtype { target_dtype };
            assert_eq!(
                trace_op_soundness(&op),
                BridgeSoundness::Unsupported,
                "{target_dtype:?} cannot be classified without the source dtype"
            );
            assert!(!is_translatable(&op));
        }
    }

    /// The `Powf` Unsupported arm is a fail-closed RESIDUE over exponents the
    /// f64 -> f32 narrowing could misrepresent — not a claim that `Powf` is
    /// unlowerable. Finite in-cap exponents classify `Sound` below.
    #[test]
    fn semantics_losing_misc_lowerings_are_unsupported() {
        for op in [
            TraceOp::Powf { exponent: f64::NAN },
            TraceOp::Powf {
                exponent: f64::INFINITY,
            },
            TraceOp::Powf {
                exponent: f64::NEG_INFINITY,
            },
            TraceOp::Powf {
                exponent: 16_777_218.0,
            },
        ] {
            assert_eq!(
                trace_op_soundness(&op),
                BridgeSoundness::Unsupported,
                "{op:?} must fail closed"
            );
            assert!(!is_translatable(&op));
        }

        for exponent in [0.0, 0.5, 1.0, 2.0, -1.0, -0.5, 3.0, -2.0, 1.0 / 3.0] {
            let op = TraceOp::Powf { exponent };
            assert_eq!(trace_op_soundness(&op), BridgeSoundness::Sound);
            assert!(is_translatable(&op));
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
                weight: WeightPayload::f32(vec![0.0], vec![1]),
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
                weight: WeightPayload::f32(vec![0.0], vec![1]),
                bias: WeightPayload::f32(vec![0.0], vec![1]),
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
