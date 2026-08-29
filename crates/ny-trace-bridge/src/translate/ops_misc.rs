// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Misc elementwise / indexing / masking family: `SwiGlu`, `Powf`, `Fract`,
//! `Atan2`, `Cumsum`, `Flip`, `Roll`, `RepeatInterleave`, `Arange`, `Triu`,
//! `Tril`, `SliceSet`, `Unfold`, `IndexSelect`, `Gather`, `Compare`,
//! `CompareTensor`, `WhereCond`, `Scatter`, `ScatterAdd`, `IndexAdd`,
//! `IndexPut`, `MoeGating`.
//!
//! Ported from NN's `trace_to_graph_layerspec_dispatch_extended.rs` /
//! `trace_to_graph_layerspec_decompose{,_scan}.rs`, retaining compatible
//! emissions where they preserve semantics and failing closed where the legacy
//! lowering or serialized data is insufficient.
//!
//! ## Implemented and audited behavior
//!
//! - `SwiGlu` → `SiLU(gate) * up` (SiLU + Mul).
//! - `Powf` → exact primitives for exponents 0 / 0.5 / 1 / 2 / −1 / −0.5;
//!   every other finite exponent with |p| ≤ 2²⁴ lowers to the x² layer shape
//!   (`LayerType::Pow` with a scalar constant exponent input → ny-propagate's
//!   parity-aware `PowConstantLayer`; see [`translate_powf`]). The former
//!   `Exp(n · Log(Abs(x)))` lowering lost the sign of negative bases and
//!   stays deleted; the native layer refuses the undefined domains
//!   (non-integer p on x < 0, negative p across zero) fail-closed at
//!   propagation time instead.
//! - `Fract` → `Sub(x, Trunc(x))`, the exact truncation-toward-zero identity
//!   for every sign (the former `x − floor(x)` lowering was wrong for
//!   negative inputs; a refusal replaced it until `TruncLayer` landed).
//!   Exact whenever the interval stays within one unit period, loose across
//!   period boundaries (the Sub decorrelates `x` from `trunc(x)`).
//! - `Atan2` → `LayerType::Atan2` binary spec.
//! - `Cumsum` → a single native `LayerType::CumSum` spec (ny-propagate's
//!   `CumsumLayer`, #3919: exact IBP via monotonicity, O(T) CROWN backward) —
//!   no size cap; the former N × Slice + (N−1) × Add + Concat decomposition
//!   was capped at `MAX_DECOMPOSE_DIM` and refused Kokoro's T=24000 harmonic
//!   source.
//! - `Flip` → N × Slice (reversed) + Concat.
//! - `RepeatInterleave` → exact expansion from the captured counts. The counts
//!   ARE on the wire: nn's `repeat_interleave` records `[x, repeats]` as the
//!   node inputs and `trace_input_ids` materializes an untraced repeats tensor
//!   as a `ConstantWeight` node with its actual data (the former "counts are
//!   not serialized" refusal was factually false). The lowering reads the
//!   counts from the captured weight payload at translate time: all-ones →
//!   identity, uniform counts → the Upsample-style Reshape + Tile + Reshape
//!   expansion (uncapped), general per-element counts → Slice-per-run +
//!   Concat along the axis. Refused only when input 1 is absent or not a
//!   translate-time constant (data-dependent counts imply a data-dependent
//!   output shape the fixed-shape graph cannot represent).
//! - `Roll` → per-(shift, dim) rotation: 2 × Slice (tail, head) + Concat,
//!   chained across the dims in trace order — exactly nn's own
//!   `narrow`/`narrow`/`cat` evaluation of `roll`, so IBP is exact (pure
//!   reindexing). All-zero effective shifts collapse to the `Add + 0`
//!   identity. (The former refusal predates this lowering; nn's opaque
//!   fallback was never mirrored — see the module-docs contrast with
//!   `Custom`.)
//! - `Arange` → constant weight tensor (data-independent, fixed at trace time).
//! - `Triu` / `Tril` → `LayerType::Triu` / `Tril` with `diagonal` attribute.
//! - `SliceSet` → up to 3 × Slice + Concat (before / src / after).
//! - `Unfold` → per-window Slice (+ Transpose) + Reshape, Concat, final Reshape.
//! - `IndexSelect` / `Gather` → `LayerType::Gather` with `axis` attribute.
//!   Unlike NN (which emits unconditionally and lets the graph build fail on a
//!   missing index tensor), a < 2-input node is refused here fail-closed.
//! - `WhereCond` → a single 3-input `LayerType::Where` spec in nn's traced
//!   input order (condition, on_true, on_false — verified against nn-core
//!   `selection/where_cond.rs`). ny-propagate's `WhereLayer` union-IBP is
//!   fully wired (`propagate_ibp_ternary`; the graph runtime routes 3-input
//!   Where nodes through it): the per-element `[min(x,y), max(x,y)]` hull is
//!   sound for ANY realizable condition, so the old refusal rationale — which
//!   attacked the `m*x + (1-m)*y` mask-arithmetic decompose ny does not use —
//!   no longer applies. IBP-only (a data-dependent select has no useful CROWN
//!   linear relaxation; `WhereLayer::propagate_linear` fails closed).
//!   Both-branch-constant nodes use the builder's embedded-constant path;
//!   any other UNIFORM constant input (nn's `masked_fill` fill) gets an
//!   exact `Clip[v, v]` activation carrier; a non-uniform single constant is
//!   refused (see [`translate_where_cond`]).
//! - `Argmax` / `Argmin` → `LayerType::Argmax` / `ArgMin` (ny-propagate
//!   `selection.rs`): exact index when one lane element provably dominates
//!   over the whole input box, sound `[0, n-1]` index hull otherwise, with
//!   outward f32 enclosure for indices above 2²⁴. `keepdims = 0` always — nn
//!   records the trace op inside `argreduce_impl` BEFORE the keepdim variants'
//!   `unsqueeze` (which traces as its own node). `select_last_index = 0`
//!   matches nn's strict-compare first-index tie policy on every wired
//!   backend (CPU `v > best`, Metal `val > best_val`; CUDA has no argreduce
//!   override and falls back to CPU).
//! - `ArgSort` → `LayerType::ArgSort`: exact permutation only when every
//!   adjacent pair in the resolved order is strictly interval-separated (then
//!   the permutation is unique under ANY tie policy — nn's `sort_unstable_by`
//!   tie order is unspecified, which is exactly why ties must fall back),
//!   sound `[0, n-1]` hull otherwise.
//! - `Sort` → sorted VALUES via `LayerType::Topk` with `k = n` and
//!   `output = "values"` (nn traces only the values tensor — both the CPU and
//!   GPU paths of `DynTensor::sort` record the op on `values`). The p-th
//!   order statistic is monotone in every input, so Topk's per-lane
//!   descending sort of each bound is EXACT for `descending = true`;
//!   `descending = false` appends the Flip-style N × Slice + Concat reversal
//!   (capped at [`MAX_DECOMPOSE_DIM`]). Values are tie-agnostic (equal values
//!   sort equal), so nn's unstable tie order is irrelevant here.
//!
//! - `ScatterAdd` / `IndexAdd` → `LayerType::ScatterAdd` / `IndexAdd`
//!   (ny-propagate `layers/transform/accumulate.rs`), a native 3-input
//!   `(data, index, src)` layer with a trailing-relative `axis`. NN lowered
//!   both as identity passthroughs (`Add(x, 0)`), which was never sound — an
//!   accumulation moves values outside the first operand's bounds — so the
//!   arms emit the real layer instead. Duplicate destinations need no UNION
//!   here, unlike the overwriting `Scatter` below: over the REALS `+` is
//!   associative and commutative, so a cell hit several times has one
//!   well-defined value rather than a set of order-dependent candidates. That
//!   argument holds in exact arithmetic only — f32 `+` is round-to-nearest and
//!   its result does depend on visit order — so the layer compensates by
//!   rounding the accumulation OUTWARD (`add_down_f32` into the lower output,
//!   `add_up_f32` into the upper), which encloses the exact real sum and every
//!   f32 visit order alike. A non-constant index is left DYNAMIC rather than
//!   refused; the layer then widens `data` by the total possible additive
//!   contribution of `src` (folded in f64 with directed rounding), which
//!   encloses every index assignment.
//! - `Scatter` (OVERWRITE) → `LayerType::ScatterND` with a synthesized
//!   ND-coordinate index constant, for a TRANSLATE-TIME-CONSTANT index only.
//!   The overwrite's duplicate destinations are last-writer-nondeterministic
//!   (nn's CPU `scatter_loop` walks src row-major; `scatter_gpu` races
//!   parallel writes), so the destination's value is a SET; `ScatterNdLayer`
//!   combines duplicate writes with min on the lower bounds and max on the
//!   upper, i.e. the union of the candidates, which encloses every order. See
//!   [`translate_scatter_overwrite`] for the rewrite and the refusals.
//! - `IndexPut` (OVERWRITE) → the SAME `LayerType::ScatterND` rewrite, for a
//!   TRANSLATE-TIME-CONSTANT index only. The op differs from `Scatter` in one
//!   place: its index is 1-D and names a destination per SLICE along `dim`
//!   rather than per element, so the synthesized coordinate for a src element
//!   reads `index[coord[dim]]`. Duplicates get the same UNION, and for a
//!   stronger reason than `Scatter`'s — nn's own loop is deterministic, but
//!   nn-import emits this variant for `aten.index_put`/`index_put_` with
//!   `accumulate = false`, which PyTorch documents as nondeterministic on
//!   duplicate indices. Negative indices are refused rather than wrapped
//!   (nn-core's `index_put` rejects them; there is no Python convention to
//!   mirror), and nn-core's shape preconditions are re-checked so a
//!   multi-axis advanced index whose extra index tensors nn-import dropped
//!   fails closed instead of being read as a single-axis write. See
//!   [`translate_index_put`]. This replaces a blanket refusal whose stated
//!   premise — that the destinations are runtime values — was only ever true
//!   of the runtime-index case.
//! - `Compare` / `CompareTensor`: lowered to ny-propagate's Compare /
//!   CompareTensor layers ({0,1}-interval IBP, sound; no CROWN linear
//!   relaxation exists for a step, so these are IBP-only).
//!
//! The `Argmax`/`Argmin`/`ArgSort` index outputs are U32 in real traces, as is
//! the scatter family's index operand; the graph-wide F32 dtype gate in
//! [`crate::translate`] must carve them out (the selection layers model indices
//! as exact f32 integer intervals; the scatter index is resolved at
//! translate/build time and never becomes a runtime edge) before end-to-end
//! translation reaches these arms — the arms themselves are wired and tested
//! directly.
//!
//! - `Topk` → sorted-descending largest-k VALUES via `LayerType::Topk`, and
//!   ONLY that. `largest` and `sorted` are now serialized (nn-core
//!   `TraceOp::Topk { k, dim, largest, sorted }`), so the wire says which of
//!   the four functions ran and the arm can decide instead of refusing
//!   wholesale. `TopkLayer` sorts each bound lane descending and keeps the
//!   first k, which models `largest && sorted` exactly — the j-th order
//!   statistic is monotone in every input, so the layer's endpoints are the
//!   attained min/max over the box. The other three combinations are refused
//!   BY NAME: `!largest` because the k smallest are not enclosed by bounds on
//!   the k largest, `!sorted` because an unspecified output order needs the
//!   hull over all k selected values and the layer's per-position interval is
//!   strictly narrower. Both would be false proofs, not loose ones. See
//!   [`translate_topk`] for the full argument and the worked counterexamples.
//!
//! ## Refused (fail-closed, deliberate)
//!
//! - `Topk` with `largest = false` or `sorted = false` — see above; the
//!   refusal is now about the FLAG COMBINATION having no layer to land on,
//!   not about the wire withholding the flags.
//! - `MoeGating`: NN lowers it as an identity passthrough (`Add(x, 0)`), which
//!   is **not** a sound over-approximation, and its expert routing is
//!   data-dependent top-k. [`crate::coverage`] classifies it `Unsupported`
//!   (must always be refused).
//! - `Scatter` / `IndexPut` with a RUNTIME index: which destinations collide is
//!   then itself a runtime value, so no static coordinate list encloses the
//!   result.
//!
//! (`ToDtype` is NOT in scope: `ops_core` fails closed because the trace omits
//! its source dtype. The `Topk`/`Argmax`/`Argmin`/`ArgSort`/`Sort` arms below
//! are live once [`crate::translate`]'s dispatch routes those variants here
//! instead of its explicit refusal arm.)

use std::collections::HashMap;

use ndarray::{ArrayD, IxDyn};
use ny_build::{AttributeValue, LayerSpec};
use ny_core::{checked_shape_product, LayerType, NyError, Result};

use crate::schema::{CompareOp, TraceNode, TraceOp};

use super::{
    checked_f64_to_f32, dim_as_i64, first_input, insert_scalar_constant, op_name, ops_core,
    shape_to_i64, simple_spec, Ctx, NodeOutput,
};

// O(N) decompositions (Flip, Unfold) emit N nodes per dimension-size
// element. Capped to avoid exploding graph sizes. (Historically mirrored the
// legacy nn translator's `MAX_DECOMPOSE_DIM`; that translator is deleted and
// this constant is now the single source of truth — nn's
// `HarmonicSourceBounds::MAX_DECOMPOSE_DIM` matches it. Cumsum left the
// capped set when it gained the native `LayerType::CumSum` lowering.)
const MAX_DECOMPOSE_DIM: usize = 2048;

/// General `Powf` exponents beyond this magnitude are refused: 2²⁴ is the
/// largest range in which every integer is exactly f32-representable, so the
/// f64 → f32 exponent narrowing in [`translate_powf`] cannot flip an
/// exponent's integrality or parity — the properties `PowConstantLayer`'s
/// sign-aware IBP branches key on. (The layer's own parity-check saturation
/// guard, #2911, sits far above at i32::MAX.)
pub(crate) const MAX_POW_EXPONENT_ABS: f64 = 16_777_216.0; // 2^24

/// Match the loader's constant-folding resource limit for materialized ranges.
const MAX_ARANGE_ELEMENTS: usize = 10_000_000;

/// Translate a misc-family op (`SwiGlu`, `Powf`, `Fract`, `Atan2`, `Cumsum`, `Flip`, `Roll`, `RepeatInterleave`, `Arange`, `Triu`, `Tril`, `SliceSet`, `Unfold`, `IndexSelect`, `Gather`, `Compare`, `CompareTensor`, `WhereCond`, `Argmax`, `Argmin`, `ArgSort`, `Sort`, `Topk`, `Scatter`, `ScatterAdd`, `IndexAdd`, `IndexPut`, `MoeGating`) node.
///
/// See the module docs for the implemented / refused split. The unsound-to-
/// lower ops (`Topk`, and the `MoeGating` catch-all) return
/// [`NyError::UnsupportedOp`]; Topk's message states the true reason
/// (unserialized largest/sorted flags), the catch-all keeps the pre-split
/// message shape. `Scatter` and `IndexPut` refuse only the runtime-index case,
/// from inside [`translate_scatter_overwrite`] / [`translate_index_put`], where
/// the reason can be stated precisely.
pub(super) fn translate_misc(
    node: &TraceNode,
    name: &str,
    input_tensors: &[String],
    output_tensor: &str,
    _node_names: &HashMap<u64, String>,
    ctx: &mut Ctx,
) -> Result<NodeOutput> {
    let output_shape = &node.output_shape;
    match &node.op {
        TraceOp::SwiGlu => translate_swiglu(name, input_tensors, output_tensor),
        TraceOp::Powf { exponent } => {
            translate_powf(name, *exponent, input_tensors, output_tensor, ctx)
        }
        TraceOp::Fract => translate_fract(name, input_tensors, output_tensor),
        TraceOp::Atan2 => translate_atan2(name, input_tensors, output_tensor),
        // Cumsum preserves rank: trailing-relative axis vs the output rank.
        // Mirrors NN's post-rework dispatch (nn d7144ea7).
        TraceOp::Cumsum { dim } => translate_cumsum(
            name,
            *dim,
            super::trailing_axis(*dim, output_shape.len(), "Cumsum axis")?,
            input_tensors,
            output_tensor,
            output_shape,
            ctx,
        ),
        // Flip preserves rank: trailing-relative axis vs the output rank.
        TraceOp::Flip { dim } => translate_flip(
            name,
            *dim,
            super::trailing_axis(*dim, output_shape.len(), "Flip axis")?,
            input_tensors,
            output_tensor,
            output_shape,
        ),
        // RepeatInterleave preserves rank: trailing-relative axis vs the
        // output rank. The counts arrive as input 1 (a ConstantWeight node)
        // and are resolved from the captured weight payload at translate time.
        TraceOp::RepeatInterleave { dim } => {
            translate_repeat_interleave(name, *dim, input_tensors, output_tensor, output_shape, ctx)
        }
        // Roll preserves rank and shape: rotations decompose per (shift, dim)
        // pair against the output shape.
        TraceOp::Roll { shifts, dims } => translate_roll(
            name,
            shifts,
            dims,
            input_tensors,
            output_tensor,
            output_shape,
            ctx,
        ),
        TraceOp::Arange { start, end, step } => {
            translate_arange(name, *start, *end, *step, output_shape, output_tensor, ctx)
        }
        TraceOp::Triu { diagonal } => {
            translate_triangular(name, *diagonal, false, input_tensors, output_tensor)
        }
        TraceOp::Tril { diagonal } => {
            translate_triangular(name, *diagonal, true, input_tensors, output_tensor)
        }
        TraceOp::SliceSet { dim, start } => {
            // Decompose: Slice(before) + src + Slice(after) → Concat.
            // Requires both self and src input shapes from context.
            if input_tensors.len() < 2 {
                return Err(NyError::InternalError(
                    "SliceSet requires 2 inputs (self, src)".to_string(),
                ));
            }
            let self_shape = lookup_tensor_shape(ctx, &input_tensors[0], "SliceSet: self")?;
            let src_shape = lookup_tensor_shape(ctx, &input_tensors[1], "SliceSet: src")?;
            // Trailing-relative axis vs the self tensor's rank (nn d7144ea7).
            translate_slice_set(
                name,
                *dim,
                super::trailing_axis(*dim, self_shape.len(), "SliceSet axis")?,
                *start,
                input_tensors,
                output_tensor,
                output_shape,
                &self_shape,
                &src_shape,
            )
        }
        TraceOp::Unfold { dim, size, step } => {
            // Decompose: N x Slice + Transpose + Reshape → Concat.
            // Requires input shape from context.
            let data_input = first_input(input_tensors, "Unfold")?;
            let input_shape = lookup_tensor_shape(ctx, &data_input, "Unfold")?;
            // Trailing-relative axis vs the input tensor's rank (nn d7144ea7).
            translate_unfold(
                name,
                *dim,
                super::trailing_axis(*dim, input_shape.len(), "Unfold axis")?,
                *size,
                *step,
                input_tensors,
                output_tensor,
                output_shape,
                &input_shape,
            )
        }
        // Both map to LayerType::Gather with a trailing-relative (negative)
        // axis attribute vs the data tensor's rank. The graph-build backend's
        // convert_gather handles constant vs dynamic indices.
        TraceOp::IndexSelect { dim } => {
            translate_gather("IndexSelect", name, *dim, input_tensors, output_tensor, ctx)
        }
        TraceOp::Gather { dim } => {
            translate_gather("Gather", name, *dim, input_tensors, output_tensor, ctx)
        }

        // Both lower to real ny-propagate comparison layers with sound
        // {0,1}-interval IBP semantics (exact when the operands' intervals
        // do not straddle, [0,1] hull when they do); IBP-only by design.
        TraceOp::Compare { op, value } => {
            translate_compare_scalar(name, *op, *value, input_tensors, output_tensor, ctx)
        }
        TraceOp::CompareTensor { op } => {
            translate_compare_tensor(name, *op, input_tensors, output_tensor)
        }

        // 3-input union-IBP conditional select (condition, on_true, on_false).
        TraceOp::WhereCond => translate_where_cond(name, input_tensors, output_tensor, ctx),

        // -- Selection family (live once the dispatch routes these here; see
        //    module docs for the per-op soundness story and the U32 dtype-gate
        //    carve-out still owned by the dispatch). All axes trailing-relative.
        TraceOp::Argmax { dim } => translate_arg_extremum(
            "Argmax",
            LayerType::Argmax,
            name,
            *dim,
            input_tensors,
            output_tensor,
            ctx,
        ),
        TraceOp::Argmin { dim } => translate_arg_extremum(
            "Argmin",
            LayerType::ArgMin,
            name,
            *dim,
            input_tensors,
            output_tensor,
            ctx,
        ),
        TraceOp::ArgSort { dim, descending } => translate_argsort(
            name,
            *dim,
            *descending,
            input_tensors,
            output_tensor,
            output_shape,
        ),
        TraceOp::Sort { dim, descending } => translate_sort(
            name,
            *dim,
            *descending,
            input_tensors,
            output_tensor,
            output_shape,
        ),
        // Topk: the flags are ON THE WIRE now, so the arm DECIDES instead of
        // refusing wholesale. `TopkLayer` models exactly one of the four
        // functions — sorted-descending largest-k — so exactly that
        // combination lowers and the other three are refused BY NAME with the
        // reason each one actually fails for (see [`translate_topk`]).
        TraceOp::Topk {
            k,
            dim,
            largest,
            sorted,
        } => translate_topk(
            name,
            *k,
            *dim,
            *largest,
            *sorted,
            input_tensors,
            output_tensor,
            output_shape,
        ),

        // -- Additive indexed accumulation. Both lower to a native 3-input
        //    (data, index, src) layer with a TRAILING-RELATIVE axis. Over the
        //    REALS `+` is associative and commutative, so duplicate
        //    destinations sum to one well-defined value and need no union
        //    treatment; in f32 the sum is order-dependent, which the layer
        //    absorbs by rounding each endpoint's accumulation OUTWARD. A
        //    non-constant index stays dynamic, where the layer's conservative
        //    additive hull is sound for every index assignment (see module
        //    docs). --
        TraceOp::ScatterAdd { dim } => translate_indexed_accumulate(
            "ScatterAdd",
            LayerType::ScatterAdd,
            name,
            *dim,
            input_tensors,
            output_tensor,
            ctx,
        ),
        TraceOp::IndexAdd { dim } => translate_indexed_accumulate(
            "IndexAdd",
            LayerType::IndexAdd,
            name,
            *dim,
            input_tensors,
            output_tensor,
            ctx,
        ),

        // -- Overwriting scatter. Lowered ONLY for a translate-time-constant
        //    index, by rewriting the axis-relative destinations into the
        //    absolute ND coordinates `ScatterNdLayer` consumes; that layer
        //    combines duplicate destinations by min/max hull, which is the
        //    union every write order lands in. A runtime index is refused. --
        TraceOp::Scatter { dim } => {
            translate_scatter_overwrite(name, *dim, input_tensors, output_tensor, ctx)
        }

        // -- Overwriting indexed write. The SAME rewrite as `Scatter` above,
        //    with a per-SLICE 1-D index instead of a per-ELEMENT one: the
        //    destinations become absolute ND coordinates, `ScatterNdLayer`
        //    unions duplicates by min/max, and a runtime index is refused with
        //    the precise reason. NN's `Add(x, 0)` identity passthrough is not
        //    an option — an indexed overwrite moves values outside the first
        //    operand's bounds. --
        TraceOp::IndexPut { dim } => {
            translate_index_put(name, *dim, input_tensors, output_tensor, ctx)
        }

        // -- Refused fail-closed (see module docs for the per-op rationale):
        //    MoeGating (NN's identity passthrough is not a sound
        //    over-approximation). --
        op => Err(NyError::UnsupportedOp(format!(
            "{} not supported in NY trace translation",
            op_name(op)
        ))),
    }
}

// ---------------------------------------------------------------------------
// Elementwise decompositions
// ---------------------------------------------------------------------------

/// Translate `TraceOp::SwiGlu` by decomposing into `SiLU(gate) * up`.
///
/// SwiGlu requires exactly 2 inputs: gate (`input[0]`) and up (`input[1]`).
/// Mirrors NN's `translate_swiglu` (#3557).
fn translate_swiglu(
    name: &str,
    input_tensors: &[String],
    output_tensor: &str,
) -> Result<NodeOutput> {
    if input_tensors.len() < 2 {
        return Err(NyError::UnsupportedOp(format!(
            "SwiGlu requires 2 inputs (gate, up), got {}",
            input_tensors.len()
        )));
    }
    let gate = input_tensors[0].clone();
    let up = input_tensors[1].clone();

    // Step 1: SiLU(gate).
    let silu_name = format!("{name}_silu");
    let silu_out = format!("{silu_name}_out");
    let silu_spec = simple_spec(
        &silu_name,
        LayerType::SiLU,
        vec![gate],
        &silu_out,
        HashMap::new(),
    );

    // Step 2: silu_out * up.
    let mul_spec = simple_spec(
        name,
        LayerType::Mul,
        vec![silu_out, up],
        output_tensor,
        HashMap::new(),
    );

    Ok(NodeOutput {
        specs: vec![silu_spec, mul_spec],
    })
}

/// Translate `TraceOp::Powf` to a sound power lowering.
///
/// Special cases map to exact primitives (mirrors NN's `translate_powf`,
/// #3557):
///   - 0 → Clip\[1,1\] (constant 1.0)
///   - 0.5 → Sqrt
///   - 1.0 → identity (`Add + 0`)
///   - 2.0 → Sqr (Pow(2))
///   - −1.0 → Reciprocal
///   - −0.5 → Sqrt + Reciprocal
///
/// Every other finite exponent with |p| ≤ [`MAX_POW_EXPONENT_ABS`] lowers to
/// the same layer shape the x² case uses — a `LayerType::Pow` spec with the
/// exponent as a scalar constant input — which ny-build converts to
/// ny-propagate's `PowConstantLayer`:
///   - integer exponents (positive and negative): parity-aware IBP (even →
///     U-shaped sign-aware, odd → monotone, both with outward directed
///     rounding); negative exponents refuse zero-straddling intervals at
///     propagation time;
///   - non-integer exponents: monotone IBP on x ≥ 0; the layer REFUSES a
///     negative input domain fail-closed at propagation time (real x^p is
///     undefined there — `powf(negative, fractional)` is NaN), so emitting
///     the spec can never yield silently-wrong bounds.
///
/// The f64 → f32 exponent narrowing reconstructs the traced op exactly: the
/// producing tracer's forward narrows the same way before computing (nn's
/// `DynTensor::powf` runs `checked_f64_to_f32(exponent)` and then f32
/// `x.powf(e)`; its GPU decomposition is f32 throughout), so
/// `PowConstantLayer` — which stores the f32 exponent and encloses the real
/// x^p by computing in f64 and rounding outward to f32 — bounds the function
/// the model actually ran. The [`MAX_POW_EXPONENT_ABS`] cap keeps the
/// narrowing from flipping integrality or parity (see the constant's docs);
/// exponents beyond it, and non-finite exponents, stay refused fail-closed.
fn translate_powf(
    name: &str,
    exponent: f64,
    input_tensors: &[String],
    output_tensor: &str,
    ctx: &mut Ctx,
) -> Result<NodeOutput> {
    let data_input = first_input(input_tensors, "Powf")?;

    // Special cases for exact primitives.
    if exponent == 0.0 {
        // x^0 = 1.0 — emit Clip[1,1] to produce constant 1.0.
        let mut attrs = HashMap::new();
        attrs.insert("min".to_string(), AttributeValue::Float(1.0));
        attrs.insert("max".to_string(), AttributeValue::Float(1.0));
        return Ok(NodeOutput::one(simple_spec(
            name,
            LayerType::Clip,
            vec![data_input],
            output_tensor,
            attrs,
        )));
    }
    if exponent == 1.0 {
        return ops_core::translate_identity_add_zero(
            name,
            "Powf exponent 1",
            input_tensors,
            output_tensor,
            ctx,
        );
    }
    if exponent == 2.0 {
        return ops_core::translate_sqr(name, input_tensors, output_tensor, ctx);
    }
    if exponent == 0.5 {
        return ops_core::translate_unary_activation(
            &TraceOp::Sqrt,
            name,
            input_tensors,
            output_tensor,
        );
    }
    if exponent == -1.0 {
        // x^(-1) = 1/x → Reciprocal.
        return Ok(NodeOutput::one(simple_spec(
            name,
            LayerType::Reciprocal,
            vec![data_input],
            output_tensor,
            HashMap::new(),
        )));
    }
    if exponent == -0.5 {
        // x^(-0.5) = 1/sqrt(x) → Sqrt + Reciprocal.
        let sqrt_name = format!("{name}_sqrt");
        let sqrt_out = format!("{sqrt_name}_out");
        let sqrt_spec = simple_spec(
            &sqrt_name,
            LayerType::Sqrt,
            vec![data_input],
            &sqrt_out,
            HashMap::new(),
        );
        let recip_spec = simple_spec(
            name,
            LayerType::Reciprocal,
            vec![sqrt_out],
            output_tensor,
            HashMap::new(),
        );
        return Ok(NodeOutput {
            specs: vec![sqrt_spec, recip_spec],
        });
    }

    // General finite exponents: the x² layer shape (`LayerType::Pow` with the
    // exponent as a scalar constant input) → ny-build's `convert_pow` →
    // ny-propagate's parity-aware `PowConstantLayer` (see doc comment for the
    // narrowing and domain-guard rationale).
    if exponent.is_finite() && exponent.abs() <= MAX_POW_EXPONENT_ABS {
        let exponent_f32 = checked_f64_to_f32(exponent, "Powf exponent")?;
        // `PowConstantLayer` classifies integrality with a 1e-6 TOLERANCE
        // ((p - p.round()).abs() < 1e-6): an exponent like 3.0000004 would be
        // treated as the exact odd cube — sign-preserving, negative domain
        // admitted — while the traced `powf` computed NaN there. Refuse the
        // ambiguous band (near-integral but not exactly integral) so the
        // layer's classifier can never diverge from the traced semantics;
        // exactly-integral and clearly-fractional exponents pass through.
        let nearest = exponent_f32.round();
        if exponent_f32 != nearest && (exponent_f32 - nearest).abs() < 1e-6 {
            return Err(NyError::UnsupportedOp(format!(
                "Powf exponent {exponent} lies inside PowConstantLayer's integer-classification \
                 tolerance without being exactly integral; refusing the ambiguous lowering"
            )));
        }
        let const_name = format!("{name}_pow_exp");
        insert_scalar_constant(ctx, &const_name, exponent_f32)?;
        let mut attrs = HashMap::new();
        attrs.insert("power".to_string(), AttributeValue::Float(exponent_f32));
        return Ok(NodeOutput::one(simple_spec(
            name,
            LayerType::Pow,
            vec![data_input, const_name],
            output_tensor,
            attrs,
        )));
    }

    Err(NyError::UnsupportedOp(format!(
        "Powf exponent {exponent} has no sound lowering outside the supported special cases \
         (finite exponents with |p| <= 2^24)"
    )))
}

/// Translate `TraceOp::Atan2` — binary `LayerType::Atan2` spec.
///
/// Mirrors NN's `translate_binary` arm `TraceOp::Atan2 => LayerType::Atan2`
/// (Kokoro STFT polar-to-rect, NN #2218 F9). Kept local because the bridge's
/// `ops_core::translate_binary` covers only the core Add/Mul/Div/Max/Min set.
/// Fract → `Sub(x, Trunc(x))` — the exact truncation-toward-zero identity.
///
/// `fract(x) = x - trunc(x)` for every sign (`floor` differs on negatives:
/// `floor(-1.5) = -2` but `trunc(-1.5) = -1`). `TruncLayer`'s IBP is exact
/// (monotone step function), so the composition is a sound enclosure — EXACT
/// whenever `trunc(lower) == trunc(upper)` (the interval stays within one
/// unit period; the kokoro sinegen frame-rate phases live there) and loose
/// across period boundaries, where the Sub decorrelates `x` from `trunc(x)`.
fn translate_fract(
    name: &str,
    input_tensors: &[String],
    output_tensor: &str,
) -> Result<NodeOutput> {
    let data_input = first_input(input_tensors, "Fract")?;
    // FLOOR, not trunc. `TraceOp::Fract` is NN's op and NN defines it
    // `x - floor(x)` with range `[0, 1)` (`nn-core/src/dyn_tensor/ops/math.rs`:
    // `UnaryOp::Fract => f32_arr.mapv(|x| x - x.floor())`, and the same
    // convention in NN's Metal kernel and its IR docs, which call out the
    // difference from Rust's trunc-based `f32::fract()` explicitly).
    //
    // This arm used to emit `Sub(x, Trunc(x))`. The two agree for x >= 0 and
    // differ by exactly 1 for every negative non-integer: at x = -1.7 NN
    // computes 0.3 while the trunc identity gives -0.7. The emitted interval
    // therefore EXCLUDED the traced value on any input box reaching below
    // zero — a false proof, not a loose bound.
    let floor_name = format!("{name}_floor");
    let floor_out = format!("{floor_name}_out");
    let floor_spec = simple_spec(
        &floor_name,
        LayerType::Floor,
        vec![data_input.clone()],
        &floor_out,
        HashMap::new(),
    );
    let sub_spec = simple_spec(
        name,
        LayerType::Sub,
        vec![data_input, floor_out],
        output_tensor,
        HashMap::new(),
    );
    Ok(NodeOutput {
        specs: vec![floor_spec, sub_spec],
    })
}

fn translate_atan2(
    name: &str,
    input_tensors: &[String],
    output_tensor: &str,
) -> Result<NodeOutput> {
    if input_tensors.len() < 2 {
        return Err(NyError::UnsupportedOp(format!(
            "Atan2 requires two inputs, got {}",
            input_tensors.len()
        )));
    }
    Ok(NodeOutput::one(simple_spec(
        name,
        LayerType::Atan2,
        input_tensors.to_vec(),
        output_tensor,
        HashMap::new(),
    )))
}

// ---------------------------------------------------------------------------
// O(N) scan decompositions (mirror NN's decompose_scan)
// ---------------------------------------------------------------------------

/// Flip → N × Slice (reverse order) + Concat.
///
/// `dim` is the raw trace dimension (for `output_shape[dim]` indexing);
/// `axis` is the trailing-relative negative encoding of `dim` (pre-encoded by
/// the caller via `super::trailing_axis`): correct in every ny-build
/// conversion regime. Mirrors NN's post-rework `translate_flip` (nn d7144ea7).
fn translate_flip(
    name: &str,
    dim: usize,
    axis: i64,
    input_tensors: &[String],
    output_tensor: &str,
    output_shape: &[usize],
) -> Result<NodeOutput> {
    let data_input = first_input(input_tensors, "Flip")?;

    if dim >= output_shape.len() {
        return Err(NyError::UnsupportedOp(format!(
            "Flip: dim {dim} >= rank {}",
            output_shape.len()
        )));
    }
    let n = output_shape[dim];
    if n == 0 {
        return Err(NyError::UnsupportedOp("Flip: dim size is 0".to_string()));
    }
    if n > MAX_DECOMPOSE_DIM {
        return Err(NyError::UnsupportedOp(format!(
            "Flip: dim size {n} exceeds decomposition limit {MAX_DECOMPOSE_DIM}"
        )));
    }

    let axis_i64 = axis;
    let mut specs = Vec::new();
    let mut slice_outs = Vec::new();

    for i in (0..n).rev() {
        let sn = format!("{name}_s{i}");
        let so = format!("{sn}_out");
        specs.push(slice_spec(
            &sn,
            vec![data_input.clone()],
            &so,
            axis_i64,
            i,
            i + 1,
        )?);
        slice_outs.push(so);
    }

    if n == 1 {
        specs.push(reshape_to_output(
            name,
            slice_outs.remove(0),
            output_tensor,
            output_shape,
        )?);
    } else {
        specs.push(concat_spec(name, slice_outs, output_tensor, axis_i64));
    }

    Ok(NodeOutput { specs })
}

/// Roll → per-(shift, dim) rotation: 2 × Slice (tail, head) + Concat, chained.
///
/// Mirrors nn's `DynTensor::roll` (nn-core `shape/device_and_repeat.rs`)
/// EXACTLY: pairs are applied sequentially in trace order, each shift is
/// normalized to `[0, n)` via `((shift % n) + n) % n`, size-0 dims and
/// zero-normalized shifts are skipped, and each effective rotation evaluates
/// `cat([narrow(n - s, s), narrow(0, n - s)])` — here `Slice` tail + `Slice`
/// head + `Concat`. Pure reindexing, so IBP is exact per element. When no
/// pair rotates (all shifts ≡ 0 mod n, or empty lists), the op is the
/// `Add + 0` identity.
///
/// Roll preserves shape, so every axis is trailing-relative vs the output
/// rank and dim sizes come from `output_shape`. A `shifts`/`dims` length
/// mismatch is refused fail-closed (nn errors before recording, so such a
/// node cannot come from a real trace).
fn translate_roll(
    name: &str,
    shifts: &[i64],
    dims: &[usize],
    input_tensors: &[String],
    output_tensor: &str,
    output_shape: &[usize],
    ctx: &mut Ctx,
) -> Result<NodeOutput> {
    let data_input = first_input(input_tensors, "Roll")?;

    if shifts.len() != dims.len() {
        return Err(NyError::UnsupportedOp(format!(
            "Roll: shifts length {} != dims length {}",
            shifts.len(),
            dims.len()
        )));
    }
    let rank = output_shape.len();
    for &dim in dims {
        if dim >= rank {
            return Err(NyError::UnsupportedOp(format!(
                "Roll: dim {dim} >= rank {rank}"
            )));
        }
    }

    // Normalize each pair; keep only effective rotations, in trace order.
    let mut effective: Vec<(i64, usize, usize)> = Vec::new(); // (axis, split, n)
    for (&shift, &dim) in shifts.iter().zip(dims.iter()) {
        let n = output_shape[dim];
        if n == 0 {
            // nn skips size-0 dims (`if dim_size == 0 { continue; }`).
            continue;
        }
        let n_i64 = dim_as_i64(n, "Roll dim size")?;
        let normalized = ((shift % n_i64) + n_i64) % n_i64;
        if normalized == 0 {
            continue;
        }
        // `normalized` is in (0, n), so `n - normalized` is in (0, n): both
        // slices are non-empty and the usize casts cannot fail.
        let split = n - normalized as usize;
        effective.push((super::trailing_axis(dim, rank, "Roll axis")?, split, n));
    }

    if effective.is_empty() {
        return ops_core::translate_identity_add_zero(
            name,
            "Roll (all shifts vanish mod dim size)",
            input_tensors,
            output_tensor,
            ctx,
        );
    }

    let mut specs = Vec::new();
    let mut current = data_input;
    let last = effective.len() - 1;
    for (index, (axis, split, n)) in effective.into_iter().enumerate() {
        let stage = format!("{name}_r{index}");
        let tail_name = format!("{stage}_tail");
        let tail_out = format!("{tail_name}_out");
        specs.push(slice_spec(
            &tail_name,
            vec![current.clone()],
            &tail_out,
            axis,
            split,
            n,
        )?);
        let head_name = format!("{stage}_head");
        let head_out = format!("{head_name}_out");
        specs.push(slice_spec(
            &head_name,
            vec![current],
            &head_out,
            axis,
            0,
            split,
        )?);
        let (cat_name, cat_out) = if index == last {
            (name.to_string(), output_tensor.to_string())
        } else {
            (stage.clone(), format!("{stage}_out"))
        };
        specs.push(concat_spec(
            &cat_name,
            vec![tail_out, head_out],
            &cat_out,
            axis,
        ));
        current = cat_out;
    }

    Ok(NodeOutput { specs })
}

/// Cumsum → a single native `LayerType::CumSum` spec.
///
/// ny-propagate's `CumsumLayer` (#3919) was built for exactly this op: IBP is
/// exact because cumsum is monotone in every input (partial sums are
/// directed-rounded outward), and CROWN backward is an O(T) suffix sum. No
/// size cap is needed — the former N × Slice + (N−1) × Add + Concat
/// decomposition emitted O(N) nodes, so it was capped at `MAX_DECOMPOSE_DIM`
/// and refused Kokoro's T=24000 harmonic source.
///
/// ny-build's `convert_cumsum` reads the axis from a scalar constant tensor
/// at input\[1\] (the ONNX CumSum convention — NOT an `axis` attribute) and
/// `exclusive` / `reverse` from attributes. `TraceOp::Cumsum` is the
/// inclusive forward scan, so both stay 0. The trailing-relative (negative)
/// `axis` passes through ny-build's `remap_axis_trailing` verbatim in every
/// conversion regime (see [`super::trailing_axis`]).
fn translate_cumsum(
    name: &str,
    dim: usize,
    axis: i64,
    input_tensors: &[String],
    output_tensor: &str,
    output_shape: &[usize],
    ctx: &mut Ctx,
) -> Result<NodeOutput> {
    let data_input = first_input(input_tensors, "Cumsum")?;

    if dim >= output_shape.len() {
        return Err(NyError::UnsupportedOp(format!(
            "Cumsum: dim {dim} >= rank {}",
            output_shape.len()
        )));
    }
    if output_shape[dim] == 0 {
        return Err(NyError::UnsupportedOp("Cumsum: dim size is 0".to_string()));
    }

    // The bridge's weight store is f32; the trailing-relative axis is a small
    // negative integer (|axis| <= rank), but verify the round-trip instead of
    // assuming a rank stays inside f32's exact-integer range.
    let axis_f32 = axis as f32;
    if axis_f32 as i64 != axis {
        return Err(NyError::InternalError(format!(
            "Cumsum: axis {axis} is not exactly representable as f32"
        )));
    }
    let axis_const = format!("{name}_axis");
    insert_scalar_constant(ctx, &axis_const, axis_f32)?;

    let mut attrs = HashMap::new();
    attrs.insert("exclusive".to_string(), AttributeValue::Int(0));
    attrs.insert("reverse".to_string(), AttributeValue::Int(0));
    Ok(NodeOutput::one(simple_spec(
        name,
        LayerType::CumSum,
        vec![data_input, axis_const],
        output_tensor,
        attrs,
    )))
}

/// RepeatInterleave → exact expansion from the captured constant counts.
///
/// nn's `repeat_interleave` records `[x, repeats]` as the node inputs
/// (`record_repeat_interleave_trace`, nn-core `ops_ext/mod.rs`), and an
/// untraced repeats tensor is materialized as a `ConstantWeight` node with
/// its actual data (`trace_input_ids`, nn-core `trace.rs`). This lowering
/// reads those counts from the captured weight payload at translate time and
/// mirrors nn's own validation (`repeat_interleave_validate_counts`):
/// non-negative integers, one per source element along `dim`.
///
/// Three exact emissions, all pure per-element selection (IBP-exact):
/// - all counts 1 → `Add + 0` identity;
/// - uniform counts `c > 1` → Reshape (insert a size-1 axis after `dim`) +
///   Tile (reps = c) + Reshape — the Upsample1d/2d nearest-neighbor
///   machinery, which IS repeat-interleave with uniform counts; uncapped;
/// - general counts → Slice-per-run + Concat along the axis: maximal runs of
///   count-1 elements share one pass-through Slice, each count-`c` element
///   gets one Slice referenced `c` times in the Concat input list. Capped at
///   [`MAX_DECOMPOSE_DIM`] concat inputs (the concat arity equals the output
///   dim size in the worst case).
///
/// Refused fail-closed when input 1 is absent or is not a translate-time
/// constant: data-dependent counts imply a data-dependent output shape (nn
/// marks a `SegmentBoundary` right after Kokoro's `length_regulate` for
/// exactly that reason), which the fixed-shape graph cannot represent.
/// Invalid serialized counts (non-integral, negative, non-finite, wrong
/// length, or a sum that contradicts the declared output shape) are
/// `ModelLoad` errors — a real nn trace cannot contain them.
fn translate_repeat_interleave(
    name: &str,
    dim: usize,
    input_tensors: &[String],
    output_tensor: &str,
    output_shape: &[usize],
    ctx: &mut Ctx,
) -> Result<NodeOutput> {
    if input_tensors.len() < 2 {
        return Err(NyError::UnsupportedOp(format!(
            "RepeatInterleave requires 2 inputs (data, counts), got {} — the per-element \
             counts tensor is missing",
            input_tensors.len()
        )));
    }
    let data_input = input_tensors[0].clone();
    let counts_name = &input_tensors[1];
    if !ctx.constant_tensors.contains(counts_name.as_str()) {
        return Err(NyError::UnsupportedOp(format!(
            "RepeatInterleave counts input {counts_name} is not a translate-time constant; \
             data-dependent counts imply a data-dependent output shape and cannot be lowered"
        )));
    }
    let counts_arr = ctx.weights.get(counts_name).ok_or_else(|| {
        NyError::InternalError(format!(
            "RepeatInterleave counts tensor {counts_name} marked constant but absent from the \
             weight store"
        ))
    })?;

    let rank = output_shape.len();
    if dim >= rank {
        return Err(NyError::UnsupportedOp(format!(
            "RepeatInterleave: dim {dim} >= rank {rank}"
        )));
    }
    let input_shape_i64 = lookup_tensor_shape(ctx, &data_input, "RepeatInterleave: data")?;
    let input_shape: Vec<usize> = input_shape_i64
        .iter()
        .enumerate()
        .map(|(i, &v)| checked_i64_to_usize(v, &format!("RepeatInterleave input dim {i}")))
        .collect::<Result<_>>()?;
    if input_shape.len() != rank {
        return Err(NyError::ModelLoad(format!(
            "RepeatInterleave: input rank {} != declared output rank {rank} (rank-preserving op)",
            input_shape.len()
        )));
    }
    for (i, (&in_dim, &out_dim)) in input_shape.iter().zip(output_shape.iter()).enumerate() {
        if i != dim && in_dim != out_dim {
            return Err(NyError::ModelLoad(format!(
                "RepeatInterleave: non-repeat dim {i} changes size {in_dim} -> {out_dim}"
            )));
        }
    }
    let n = input_shape[dim];

    // Counts: mirror nn's validation — flatten, length must equal the source
    // dim size, every element a non-negative integer.
    let counts_f32: Vec<f32> = counts_arr.iter().copied().collect();
    if counts_f32.len() != n {
        return Err(NyError::ModelLoad(format!(
            "RepeatInterleave: counts length {} != dim {dim} size {n}",
            counts_f32.len()
        )));
    }
    let mut counts = Vec::with_capacity(n);
    for (i, &v) in counts_f32.iter().enumerate() {
        if !v.is_finite() || v < 0.0 || v != v.trunc() {
            return Err(NyError::ModelLoad(format!(
                "RepeatInterleave: count[{i}] = {v} must be a non-negative integer"
            )));
        }
        counts.push(v as usize);
    }
    let total: usize = counts.iter().try_fold(0usize, |acc, &c| {
        acc.checked_add(c).ok_or_else(|| {
            NyError::ModelLoad("RepeatInterleave: counts sum overflows usize".to_string())
        })
    })?;
    if total != output_shape[dim] {
        return Err(NyError::ModelLoad(format!(
            "RepeatInterleave: declared output dim {dim} size {} but counts sum to {total}",
            output_shape[dim]
        )));
    }
    if total == 0 {
        return Err(NyError::UnsupportedOp(
            "RepeatInterleave: all counts are 0 (zero-size output)".to_string(),
        ));
    }

    // All-ones: identity (only the data input participates).
    if counts.iter().all(|&c| c == 1) {
        return ops_core::translate_identity_add_zero(
            name,
            "RepeatInterleave (all counts 1)",
            &input_tensors[..1],
            output_tensor,
            ctx,
        );
    }

    // Uniform counts c > 1: Reshape + Tile + Reshape (nearest-upsample
    // machinery generalized to an interior dim). Exact — the tiled axis has
    // size 1, so repetition is pure broadcast of each element's interval.
    let uniform = counts[0];
    if uniform > 1 && counts.iter().all(|&c| c == uniform) {
        let mut unsq_shape = input_shape;
        unsq_shape.insert(dim + 1, 1);
        let unsq_name = format!("{name}_unsq");
        let unsq_out = format!("{unsq_name}_out");
        let unsq = reshape_to_output(&unsq_name, data_input, &unsq_out, &unsq_shape)?;

        // Positive tile axis, verbatim under the unbatched convention —
        // mirrors `translate_upsample2d`'s interior-axis tiles.
        let tile_name = format!("{name}_tile");
        let tile_out = format!("{tile_name}_out");
        let tile = tile_spec(
            &tile_name,
            vec![unsq_out],
            &tile_out,
            dim_as_i64(dim + 1, "RepeatInterleave tile axis")?,
            uniform,
        )?;

        let merge = reshape_to_output(name, tile_out, output_tensor, output_shape)?;
        return Ok(NodeOutput {
            specs: vec![unsq, tile, merge],
        });
    }

    // General counts: Slice-per-run + Concat. Segments are (start, end, reps):
    // maximal count-1 runs collapse into one slice with reps = 1; a
    // count-c element (c > 1) is its own slice with reps = c; count-0
    // elements emit nothing.
    let mut segments: Vec<(usize, usize, usize)> = Vec::new();
    let mut run_start: Option<usize> = None;
    for (i, &c) in counts.iter().enumerate() {
        match c {
            1 => {
                if run_start.is_none() {
                    run_start = Some(i);
                }
            }
            _ => {
                if let Some(start) = run_start.take() {
                    segments.push((start, i, 1));
                }
                if c > 1 {
                    segments.push((i, i + 1, c));
                }
            }
        }
    }
    if let Some(start) = run_start.take() {
        segments.push((start, n, 1));
    }

    // Cap the expansion on the CONCAT FAN-IN (= emitted slice references),
    // not the raw output dim size: long count-1 runs collapse into single
    // slices, so the fan-in — which bounds both the emitted node count and
    // the concat arity — is the actual graph-size cost.
    let fan_in: usize = segments.iter().map(|&(_, _, reps)| reps).sum();
    if fan_in > MAX_DECOMPOSE_DIM {
        return Err(NyError::UnsupportedOp(format!(
            "RepeatInterleave: expansion needs {fan_in} concat inputs, exceeding decomposition \
             limit {MAX_DECOMPOSE_DIM}"
        )));
    }

    let axis = super::trailing_axis(dim, rank, "RepeatInterleave axis")?;
    let mut specs = Vec::new();
    let mut concat_inputs = Vec::new();
    for (index, &(start, end, reps)) in segments.iter().enumerate() {
        let sn = format!("{name}_s{index}");
        let so = format!("{sn}_out");
        specs.push(slice_spec(
            &sn,
            vec![data_input.clone()],
            &so,
            axis,
            start,
            end,
        )?);
        for _ in 0..reps {
            concat_inputs.push(so.clone());
        }
    }

    if concat_inputs.len() == 1 {
        specs.push(reshape_to_output(
            name,
            concat_inputs.remove(0),
            output_tensor,
            output_shape,
        )?);
    } else {
        specs.push(concat_spec(name, concat_inputs, output_tensor, axis));
    }
    Ok(NodeOutput { specs })
}

/// Unfold → N × Slice (+ Transpose) + Reshape, Concat, final Reshape.
///
/// For input shape `[d0, ..., d_dim, ..., dN]` with `unfold(dim, size, step)`,
/// produces `[d0, ..., n_windows, ..., dN, size]` where
/// `n_windows = (d_dim - size) / step + 1`. Mirrors NN's `translate_unfold`
/// (#3094).
#[allow(clippy::too_many_arguments)]
fn translate_unfold(
    name: &str,
    dim: usize,
    axis: i64,
    size: usize,
    step: usize,
    input_tensors: &[String],
    output_tensor: &str,
    output_shape: &[usize],
    input_shape: &[i64],
) -> Result<NodeOutput> {
    let data_input = first_input(input_tensors, "Unfold")?;

    if step == 0 {
        return Err(NyError::UnsupportedOp(
            "Unfold: step must be > 0".to_string(),
        ));
    }
    if size == 0 {
        return Err(NyError::UnsupportedOp(
            "Unfold: size must be > 0".to_string(),
        ));
    }
    let rank = input_shape.len();
    if dim >= rank {
        return Err(NyError::UnsupportedOp(format!(
            "Unfold: dim {dim} >= input rank {rank}"
        )));
    }
    let dim_size = checked_i64_to_usize(input_shape[dim], &format!("Unfold input dim {dim}"))?;
    if size > dim_size {
        return Err(NyError::UnsupportedOp(format!(
            "Unfold: size ({size}) > dim size ({dim_size})"
        )));
    }
    let n_windows = (dim_size - size) / step + 1;
    if n_windows == 0 {
        return Err(NyError::UnsupportedOp("Unfold: no windows fit".to_string()));
    }
    if n_windows > MAX_DECOMPOSE_DIM {
        return Err(NyError::UnsupportedOp(format!(
            "Unfold: n_windows ({n_windows}) exceeds decomposition limit {MAX_DECOMPOSE_DIM}"
        )));
    }

    let axis_i64 = axis;
    let need_permute = dim < rank - 1;

    // After slicing along dim: same shape but dim axis has `size` elements.
    let narrow_shape: Vec<usize> = input_shape
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            if i == dim {
                Ok(size)
            } else {
                checked_i64_to_usize(v, &format!("Unfold dim {i}"))
            }
        })
        .collect::<Result<_>>()?;

    // Permute axes: move dim to end [0, ..., dim-1, dim+1, ..., rank-1, dim].
    let perm_axes: Vec<usize> = if need_permute {
        let mut a: Vec<usize> = (0..rank).filter(|&x| x != dim).collect();
        a.push(dim);
        a
    } else {
        (0..rank).collect()
    };
    let perm_shape: Vec<usize> = perm_axes.iter().map(|&a| narrow_shape[a]).collect();

    // Window shape: insert 1 at position `dim` (window count axis).
    let mut window_shape = perm_shape;
    window_shape.insert(dim, 1);

    // Transpose perm VERBATIM in trace-rank convention: ny-build's
    // convert_transpose applies perms as-is and the runtime TransposeLayer
    // validates them against the actual rank, so the historic `+1`
    // pretend-batched perm (values 1..=rank on a rank-`rank` tensor) was
    // never a valid permutation. Mirrors NN's post-rework `translate_unfold`
    // (nn d7144ea7).
    let perm_i64: Vec<i64> = perm_axes
        .iter()
        .map(|&a| dim_as_i64(a, "Unfold perm"))
        .collect::<Result<_>>()?;

    // Concat axis trailing-relative w.r.t. the window tensors' rank
    // (`rank + 1`: the per-window Reshape inserts a size-1 window-count axis
    // at position `dim`).
    let cat_onnx_axis = dim as i64 - (rank as i64 + 1);

    let mut specs = Vec::new();
    let mut window_outs = Vec::new();

    for w in 0..n_windows {
        let start = w * step;
        let end = start + size;

        // Slice: extract [start, end) along dim.
        let sn = format!("{name}_w{w}_sl");
        let so = format!("{sn}_out");
        specs.push(slice_spec(
            &sn,
            vec![data_input.clone()],
            &so,
            axis_i64,
            start,
            end,
        )?);

        // Transpose (if needed): move dim to last position.
        let prev_out = if need_permute {
            let tn = format!("{name}_w{w}_tr");
            let to = format!("{tn}_out");
            let mut tr_attrs = HashMap::new();
            tr_attrs.insert("perm".to_string(), AttributeValue::Ints(perm_i64.clone()));
            specs.push(simple_spec(
                &tn,
                LayerType::Transpose,
                vec![so],
                &to,
                tr_attrs,
            ));
            to
        } else {
            so
        };

        // Reshape: insert a size-1 window dim at position `dim`.
        let rn = format!("{name}_w{w}_rs");
        let ro = format!("{rn}_out");
        specs.push(reshape_to_output(&rn, prev_out, &ro, &window_shape)?);
        window_outs.push(ro);
    }

    // Concat all windows, then reshape to final output.
    if n_windows == 1 {
        specs.push(reshape_to_output(
            name,
            window_outs.remove(0),
            output_tensor,
            output_shape,
        )?);
    } else {
        // Concat produces [..., n_windows, ..., size] but may not exactly
        // match output_shape if the permute reordered axes. Final reshape
        // ensures correct shape.
        let cn = format!("{name}_cat");
        let co = format!("{cn}_out");
        specs.push(concat_spec(&cn, window_outs, &co, cat_onnx_axis));
        specs.push(reshape_to_output(name, co, output_tensor, output_shape)?);
    }

    Ok(NodeOutput { specs })
}

/// SliceSet → up to 3 × Slice + Concat.
///
/// `SliceSet { dim, start }` writes `src` (`input[1]`) into `self`
/// (`input[0]`) at `[start..start+src_len]` along `dim`; the output has the
/// shape of `self`. Decomposes into before-slice / src / after-slice
/// concatenated along `dim`. Mirrors NN's `translate_slice_set` (#3094).
#[allow(clippy::too_many_arguments)]
fn translate_slice_set(
    name: &str,
    dim: usize,
    axis: i64,
    start: usize,
    input_tensors: &[String],
    output_tensor: &str,
    output_shape: &[usize],
    input_shape: &[i64],
    src_shape: &[i64],
) -> Result<NodeOutput> {
    if input_tensors.len() < 2 {
        return Err(NyError::InternalError(
            "SliceSet requires 2 inputs (self, src)".to_string(),
        ));
    }
    let self_input = input_tensors[0].clone();
    let src_input = input_tensors[1].clone();

    let rank = input_shape.len();
    if dim >= rank {
        return Err(NyError::UnsupportedOp(format!(
            "SliceSet: dim {dim} >= rank {rank}"
        )));
    }
    let dim_size = checked_i64_to_usize(input_shape[dim], &format!("SliceSet self dim {dim}"))?;

    // Determine src length along dim from src_shape.
    if dim >= src_shape.len() {
        return Err(NyError::UnsupportedOp(format!(
            "SliceSet: dim {dim} >= src rank {}",
            src_shape.len()
        )));
    }
    let src_len = checked_i64_to_usize(src_shape[dim], &format!("SliceSet src dim {dim}"))?;
    let end = start.checked_add(src_len).ok_or_else(|| {
        NyError::InternalError(format!(
            "SliceSet: start ({start}) + src_len ({src_len}) overflows"
        ))
    })?;
    if end > dim_size {
        return Err(NyError::UnsupportedOp(format!(
            "SliceSet: start ({start}) + src_len ({src_len}) = {end} > dim_size ({dim_size})"
        )));
    }

    let axis_i64 = axis;
    let mut specs = Vec::new();
    let mut concat_inputs = Vec::new();

    // Before part: self[0..start] along dim.
    if start > 0 {
        let bn = format!("{name}_before");
        let bo = format!("{bn}_out");
        specs.push(slice_spec(
            &bn,
            vec![self_input.clone()],
            &bo,
            axis_i64,
            0,
            start,
        )?);
        concat_inputs.push(bo);
    }

    // Middle part: src replaces self[start..end].
    concat_inputs.push(src_input);

    // After part: self[end..dim_size] along dim.
    if end < dim_size {
        let an = format!("{name}_after");
        let ao = format!("{an}_out");
        specs.push(slice_spec(
            &an,
            vec![self_input],
            &ao,
            axis_i64,
            end,
            dim_size,
        )?);
        concat_inputs.push(ao);
    }

    // If only one piece (full replacement), just reshape to output.
    if concat_inputs.len() == 1 {
        specs.push(reshape_to_output(
            name,
            concat_inputs.remove(0),
            output_tensor,
            output_shape,
        )?);
    } else {
        specs.push(concat_spec(name, concat_inputs, output_tensor, axis_i64));
    }

    Ok(NodeOutput { specs })
}

// ---------------------------------------------------------------------------
// Constant / mask / gather emissions (mirror NN's dispatch_extended)
// ---------------------------------------------------------------------------

/// Translate `TraceOp::Arange` to a constant weight tensor.
///
/// Arange is data-independent: `[start, start+step, start+2*step, ...]` is
/// fully determined at trace time. Emit as a constant weight so downstream
/// ops treat it as a fixed tensor, not a variable input. Mirrors NN's
/// `translate_arange` (#2271).
fn translate_arange(
    name: &str,
    start: f64,
    end: f64,
    step: f64,
    output_shape: &[usize],
    output_tensor: &str,
    ctx: &mut Ctx,
) -> Result<NodeOutput> {
    if !start.is_finite() || !end.is_finite() || !step.is_finite() {
        return Err(NyError::ModelLoad(format!(
            "Arange start/end/step must be finite, got start={start}, end={end}, step={step}"
        )));
    }
    if step == 0.0 {
        return Err(NyError::UnsupportedOp(
            "Arange with step=0 is undefined".to_string(),
        ));
    }
    if (step > 0.0 && start > end) || (step < 0.0 && start < end) {
        return Err(NyError::ModelLoad(format!(
            "Arange step {step} points away from end {end} starting at {start}"
        )));
    }

    let ratio = (end - start) / step;
    let count_f64 = ratio.ceil().max(0.0);
    if !count_f64.is_finite() {
        return Err(NyError::ModelLoad(format!(
            "Arange element count is non-finite for start={start}, end={end}, step={step}"
        )));
    }
    if count_f64 > MAX_ARANGE_ELEMENTS as f64 {
        return Err(NyError::ModelLoad(format!(
            "Arange requires {count_f64} elements, exceeding the {MAX_ARANGE_ELEMENTS}-element limit"
        )));
    }
    let count = count_f64 as usize;

    if output_shape.len() != 1 {
        return Err(NyError::ModelLoad(format!(
            "Arange must declare a rank-1 output shape, got {output_shape:?}"
        )));
    }
    let declared_elements = checked_shape_product(output_shape).ok_or_else(|| {
        NyError::ModelLoad(format!(
            "Arange output shape {output_shape:?} has a dimension product that overflows usize"
        ))
    })?;
    if declared_elements != count {
        return Err(NyError::ModelLoad(format!(
            "Arange declared output shape {output_shape:?} has {declared_elements} elements but parameters produce {count}"
        )));
    }

    let mut data = Vec::new();
    data.try_reserve_exact(count).map_err(|error| {
        NyError::ModelLoad(format!(
            "Arange allocation failed for {count} elements: {error}"
        ))
    })?;
    for index in 0..count {
        let val = start + (index as f64) * step;
        let val_f32 = checked_f64_to_f32(val, "Arange element")?;
        data.push(val_f32);
    }

    let arr = ArrayD::from_shape_vec(IxDyn(output_shape), data)
        .map_err(|e| NyError::InternalError(format!("{name} Arange shape mismatch: {e}")))?;
    ctx.insert_weight(output_tensor, arr)?;
    Ok(NodeOutput::none())
}

/// Translate `TraceOp::Triu` / `TraceOp::Tril` to NY triangular mask layers.
///
/// The graph-build backend's `convert_triangular` generates a binary mask from
/// the input shape and diagonal offset, producing a `MulConstantLayer`. The
/// LayerSpec just needs the `diagonal` attribute. Mirrors NN's
/// `translate_triangular` (#2271).
fn translate_triangular(
    name: &str,
    diagonal: i64,
    lower: bool,
    input_tensors: &[String],
    output_tensor: &str,
) -> Result<NodeOutput> {
    let layer_type = if lower {
        LayerType::Tril
    } else {
        LayerType::Triu
    };
    let mut attrs = HashMap::new();
    attrs.insert("diagonal".to_string(), AttributeValue::Int(diagonal));
    Ok(NodeOutput::one(simple_spec(
        name,
        layer_type,
        input_tensors.to_vec(),
        output_tensor,
        attrs,
    )))
}

/// Translate `TraceOp::IndexSelect` / `TraceOp::Gather` to `LayerType::Gather`
/// with a TRAILING-RELATIVE (negative) `axis` attribute w.r.t. the DATA
/// tensor's rank (ONNX Gather axes index the data input).
///
/// Mirrors NN's `IndexSelect | Gather` arm. NN emits unconditionally and lets
/// the graph build reject a missing index tensor; the bridge refuses < 2-input
/// nodes here (fail-closed — a 1-input gather has no index tensor and cannot
/// be lowered, so translation must not succeed).
///
/// Axis-audit note (consolidation pass): the historic pretend-batched `+1`
/// encoding was WRONG under ny-build's recorded-rank convention for every
/// exercised configuration — the data input is always a recorded tensor, so
/// `dim + 1` either fails the range check (`dim == rank - 1`) or silently
/// selects dimension `dim + 1` (interior dims; e.g. an embedding-table
/// lookup `dim=0` on a rank-2 table would gather along the embedding axis).
/// It was latent only because no suite drives a 2-input Gather to
/// propagation. Trailing-relative negative resolves to `dim` in every
/// ny-build conversion regime; the `axis == 0 && constant-data` embedding
/// special case in convert_gather is unaffected (it triggers only on
/// ONNX-front-end positive-0 emissions).
fn translate_gather(
    op_desc: &str,
    name: &str,
    dim: usize,
    input_tensors: &[String],
    output_tensor: &str,
    ctx: &Ctx,
) -> Result<NodeOutput> {
    if input_tensors.len() < 2 {
        return Err(NyError::UnsupportedOp(format!(
            "{op_desc} requires 2 inputs (data, indices), got {} — index tensor missing",
            input_tensors.len()
        )));
    }
    let data_rank = lookup_tensor_shape(ctx, &input_tensors[0], op_desc)?.len();
    let axis = super::trailing_axis(dim, data_rank, &format!("{op_desc} axis"))?;
    let mut attrs = HashMap::new();
    attrs.insert("axis".to_string(), AttributeValue::Int(axis));
    Ok(NodeOutput::one(simple_spec(
        name,
        LayerType::Gather,
        input_tensors.to_vec(),
        output_tensor,
        attrs,
    )))
}

// ---------------------------------------------------------------------------
// Indexed accumulation / scatter
// ---------------------------------------------------------------------------

/// Largest total element count for a synthesized ScatterND index constant.
/// A `Scatter` over `N` src elements of rank `R` materializes `N * R` i64
/// coordinates; cap it the way the other materializing lowerings are capped.
const MAX_SCATTER_INDEX_ELEMENTS: usize = 10_000_000;

/// Translate `TraceOp::ScatterAdd` / `TraceOp::IndexAdd` to their native
/// `LayerType` with a TRAILING-RELATIVE (negative) `axis` attribute w.r.t. the
/// DATA tensor's rank.
///
/// nn records both as `(self, index, src)` (`nn-core`
/// `dyn_tensor/selection/accumulate.rs`), which is exactly the operand order
/// `ScatterAddLayer` / `IndexAddLayer` resolve, so the three inputs pass
/// through unchanged and the graph-build backend decides per operand whether
/// to embed a constant or keep an activation edge.
///
/// Axis convention matches [`translate_gather`]: `dim` is already resolved
/// against the tensor's own rank by nn's `Dim::to_index`, and the negative
/// re-encoding names the same dimension whether or not the runtime tensor kept
/// a leading batch axis. The layers resolve it against the actual runtime rank.
///
/// SOUNDNESS. Unlike the overwriting `Scatter` below, this family needs no
/// duplicate-index UNION: `+` over the REALS is associative and commutative, so
/// a destination named by several index entries receives the SUM of those
/// contributions — one value, not a set of order-dependent candidates. That is
/// what makes a single accumulating layer the right target instead of a hull
/// over write orders.
///
/// The exactness is where the argument stops. In f32 `+` is round-to-NEAREST
/// and therefore neither associative nor order-independent: `1e8 + 1.0` rounds
/// straight back to `1e8`, so a long accumulation can silently drop every small
/// term, and it does so IDENTICALLY on both endpoints — which translates the
/// certified interval off the true value rather than widening it around the
/// order-dependence. `ScatterAddLayer` / `IndexAddLayer` close that by rounding
/// each endpoint's accumulation OUTWARD (`add_down_f32` builds the lower
/// output, `add_up_f32` the upper; the dynamic-index additive hull folds in f64
/// with directed adds and one directed cast). The resulting interval encloses
/// the exact real sum, and hence every f32 visit order of it, so this lowering
/// still needs no order analysis — but that is a property of the layer's
/// rounding discipline, not of f32 addition.
///
/// The refusal these arms replace claimed NN's `Add(x, 0)` identity lowering
/// was the only option; it is not, and the identity was never sound (an
/// accumulation moves values outside the first operand's bounds).
fn translate_indexed_accumulate(
    op_desc: &str,
    layer_type: LayerType,
    name: &str,
    dim: usize,
    input_tensors: &[String],
    output_tensor: &str,
    ctx: &Ctx,
) -> Result<NodeOutput> {
    if input_tensors.len() != 3 {
        return Err(NyError::UnsupportedOp(format!(
            "{op_desc} requires 3 inputs (data, index, src), got {}",
            input_tensors.len()
        )));
    }
    let data_rank = lookup_tensor_shape(ctx, &input_tensors[0], op_desc)?.len();
    let axis = super::trailing_axis(dim, data_rank, &format!("{op_desc} axis"))?;
    let mut attrs = HashMap::new();
    attrs.insert("axis".to_string(), AttributeValue::Int(axis));
    Ok(NodeOutput::one(simple_spec(
        name,
        layer_type,
        input_tensors.to_vec(),
        output_tensor,
        attrs,
    )))
}

/// Translate `TraceOp::Scatter { dim }` — the OVERWRITING scatter — by
/// rewriting its axis-relative destinations into absolute ND coordinates and
/// emitting a `LayerType::ScatterND`.
///
/// ## Why the overwrite needs a union and the additive family does not
///
/// `scatter` writes `out[i][index[i][j][k]][k] = src[i][j][k]`. When two index
/// entries name the same destination, the surviving value is whichever write
/// happened LAST — and that order is not part of the op's semantics. nn's own
/// backends disagree by construction: the CPU path walks `src` in row-major
/// order (`scatter_loop`), while `scatter_gpu` dispatches a parallel kernel
/// whose duplicate writes race. PyTorch documents the same nondeterminism.
///
/// So a duplicate destination's value is a SET of candidates, and the only
/// sound enclosure is their union. Picking either order would be a false
/// proof for the other.
///
/// ## Why `ScatterNdLayer` is the right target
///
/// `ScatterNdLayer::scatter_tensor` (ny-propagate
/// `layers/transform/scatter_nd/mod.rs`) applies the FIRST write to a cell
/// directly and folds every subsequent write into it with a combiner —
/// `nan_propagating_min` when propagating the lower bounds and
/// `nan_propagating_max` for the upper. A cell hit by candidates `u1, u2`
/// therefore ends up bounded by `[min(lo1, lo2), max(up1, up2)]`, which
/// contains both `u1` and `u2` — the union, independent of order. Its CROWN
/// backward refuses duplicate targets outright (`build_write_map` returns
/// `UnsupportedOp`), falling back to that sound IBP rule rather than
/// linearizing a non-linear choice. Unwritten cells keep `data` exactly.
///
/// ## The rewrite
///
/// `ScatterND` addresses destinations by full coordinates: `indices` of shape
/// `prefix ++ [index_depth]` with `updates` of shape
/// `prefix ++ data_shape[index_depth..]`. Choosing `index_depth = data_rank`
/// makes the remainder empty, so setting `prefix = src.shape()` requires
/// `updates.shape() == src.shape()` — i.e. `src` feeds the layer UNCHANGED,
/// with no reshape and no flattening-order hazard. Each src coordinate
/// contributes one row: itself, with the scatter axis replaced by that
/// element's (normalized) index.
///
/// ## Refusals
///
/// A non-constant index is refused: the destinations would be runtime values,
/// and no static coordinate list encloses them. Out-of-range and non-integral
/// index values are refused rather than clamped or rounded onto a real cell.
fn translate_scatter_overwrite(
    name: &str,
    dim: usize,
    input_tensors: &[String],
    output_tensor: &str,
    ctx: &mut Ctx,
) -> Result<NodeOutput> {
    if input_tensors.len() != 3 {
        return Err(NyError::UnsupportedOp(format!(
            "Scatter requires 3 inputs (data, index, src), got {}",
            input_tensors.len()
        )));
    }
    // Resolve the index FIRST, before any shape lookup: a runtime index is the
    // refusal whose reason has to be stated precisely, and a missing recorded
    // shape must not pre-empt it with a vaguer message.
    let index_name = &input_tensors[1];
    let Some((index_shape, index_values)) = constant_index_values(ctx, index_name, "Scatter")?
    else {
        return Err(NyError::UnsupportedOp(format!(
            "Scatter index input '{index_name}' is not a translate-time constant. The scatter is \
             an OVERWRITE, so a duplicate destination keeps whichever write landed last — an \
             order nn does not fix (its CPU `scatter_loop` walks src in row-major order while \
             `scatter_gpu` races parallel writes), which makes the destination's value a set of \
             candidates rather than one value. A sound lowering must enclose that whole set, and \
             that requires knowing which destinations collide; with runtime indices the collision \
             pattern is itself a runtime value, so no static layer encloses the result"
        )));
    };

    let data_shape = shape_as_usize(ctx, &input_tensors[0], "Scatter: data")?;
    let src_shape = shape_as_usize(ctx, &input_tensors[2], "Scatter: src")?;
    let rank = data_shape.len();
    if dim >= rank {
        return Err(NyError::UnsupportedOp(format!(
            "Scatter: dim {dim} out of range for data rank {rank}"
        )));
    }

    if index_shape != src_shape {
        return Err(NyError::UnsupportedOp(format!(
            "Scatter: index shape {index_shape:?} must equal src shape {src_shape:?}"
        )));
    }
    if index_shape.len() != rank {
        return Err(NyError::UnsupportedOp(format!(
            "Scatter: index rank {} must equal data rank {rank}",
            index_shape.len()
        )));
    }
    for (axis, (&src_dim, &data_dim)) in src_shape.iter().zip(data_shape.iter()).enumerate() {
        if axis != dim && src_dim > data_dim {
            return Err(NyError::UnsupportedOp(format!(
                "Scatter: src dim {axis} size {src_dim} exceeds data dim size {data_dim}"
            )));
        }
    }

    let src_elems = checked_shape_product(&src_shape).ok_or_else(|| {
        NyError::InternalError(format!("Scatter src shape {src_shape:?} overflows usize"))
    })?;
    let coordinate_elems = src_elems.checked_mul(rank).ok_or_else(|| {
        NyError::InternalError("Scatter ND index element count overflows usize".to_string())
    })?;
    if coordinate_elems > MAX_SCATTER_INDEX_ELEMENTS {
        return Err(NyError::UnsupportedOp(format!(
            "Scatter would materialize a {coordinate_elems}-element ND index constant, exceeding \
             the {MAX_SCATTER_INDEX_ELEMENTS}-element limit"
        )));
    }

    // One ND row per src element, in src's row-major order: the element's own
    // coordinate with the scatter axis replaced by its destination index.
    let axis_len = data_shape[dim];
    let mut coordinates: Vec<i64> = Vec::new();
    coordinates
        .try_reserve_exact(coordinate_elems)
        .map_err(|e| {
            NyError::ModelLoad(format!(
                "Scatter ND index allocation failed for {coordinate_elems} elements: {e}"
            ))
        })?;
    let mut coord = vec![0usize; rank];
    for raw in index_values.iter().copied() {
        let destination = normalize_scatter_index(raw, axis_len)?;
        for (axis, &position) in coord.iter().enumerate() {
            let value = if axis == dim { destination } else { position };
            coordinates.push(dim_as_i64(value, "Scatter ND coordinate")?);
        }
        advance_row_major(&mut coord, &src_shape);
    }

    // `prefix ++ [rank]`, so ScatterND's expected updates shape is exactly
    // `src_shape` and `src` needs no reshape.
    let mut index_tensor_shape = src_shape;
    index_tensor_shape.push(rank);
    let nd_index_tensor = format!("{name}_nd_indices");
    insert_index_constant(ctx, &nd_index_tensor, &index_tensor_shape, coordinates)?;

    Ok(NodeOutput::one(simple_spec(
        name,
        LayerType::ScatterND,
        vec![
            input_tensors[0].clone(),
            nd_index_tensor,
            input_tensors[2].clone(),
        ],
        output_tensor,
        HashMap::new(),
    )))
}

/// Translate `TraceOp::IndexPut { dim }` — the OVERWRITING indexed write —
/// through the same constant-index rewrite `Scatter` receives.
///
/// ## What the op is
///
/// nn-core `dyn_tensor/selection/index_put.rs`: `index_put(dim, index, src)`
/// writes `out[.., index[j], ..] = src[.., j, ..]` for every `j`, leaving every
/// other cell of `self` untouched, and records `(self, index, src)` as the
/// traced inputs. Its preconditions are narrow and are RE-CHECKED here rather
/// than assumed (see "Why the shape checks are load-bearing"): `index` is 1-D,
/// `src` has `self`'s rank and `self`'s extents on every axis but `dim`, and
/// `src.dims()[dim] == index.len()`.
///
/// So it is `Scatter` with one degree of freedom removed: a scatter's index is
/// shaped like `src` and names a destination per ELEMENT, while an index_put's
/// index is 1-D and names a destination per SLICE along `dim`. Everything
/// downstream of that difference — the ND-coordinate rewrite, the layer, the
/// duplicate rule, the refusals — is `Scatter`'s, and deliberately so.
///
/// ## Duplicate destinations: union, never pick-one
///
/// nn documents "duplicate indices: last write wins", and its own CPU loop is
/// deterministic (the GPU path round-trips through it). That determinism is NOT
/// enough to justify modelling one order, for two reasons, and the second is
/// decisive:
///
/// 1. "last" is an artifact of the flat `src` walk, not of the op's contract;
///    PyTorch documents the duplicate case of both `index_put_(accumulate=False)`
///    and `index_copy_` — the op nn's own doc comment cites — as
///    nondeterministic.
/// 2. `TraceOp::IndexPut` is not written only by nn-core. `nn-import`'s
///    `map_index_put` / `map_index_put_hacked_twin` emit this same variant for
///    `aten.index_put.default` / `.hacked_twin` with `accumulate = false`,
///    which is exactly PyTorch's nondeterministic-on-duplicates operator.
///
/// So a contested cell's value is a SET of candidates, and the emitted bounds
/// must contain all of them. `ScatterNdLayer` combines duplicate writes with
/// min on the lower bounds and max on the upper — the union, independent of
/// order — which is the same property that makes it the right target for
/// `Scatter` (see [`translate_scatter_overwrite`] for the layer argument in
/// full). Enclosing the union is looser than nn's deterministic CPU result on a
/// duplicated destination and exactly equal to it everywhere else; being loose
/// there is the price of not assuming an order the wire does not fix.
///
/// ## Why the shape checks are load-bearing
///
/// nn-import's `map_index_put` handles PyTorch's ADVANCED indexing by taking
/// `index_names[0]` and discarding any further index tensors, so an imported
/// `x[i, j] = v` arrives here as `IndexPut { dim: 0 }` carrying only `i`. That
/// trace does not describe the model, and no lowering can rescue it — but it
/// also does not satisfy nn-core's preconditions (a two-axis advanced write
/// gives `src` a lower rank than `self`, or an extent that does not match), so
/// re-checking them turns the mis-encoded trace into a REFUSAL instead of a
/// confident bound on the wrong function. The checks are cheap; the failure
/// mode they close is silent.
///
/// ## Refusals
///
/// A non-constant index is refused: which destinations collide is then itself a
/// runtime value. A NEGATIVE index is refused rather than wrapped — nn-core
/// rejects negative `I64` indices outright and its `U32` path cannot express
/// one, so the Python convention `normalize_scatter_index` applies for
/// `Scatter` has no counterpart on this op, and applying it anyway would bound
/// a computation nn cannot perform. Out-of-range and non-integral values are
/// refused for the same fail-closed reason they are under `Scatter`.
fn translate_index_put(
    name: &str,
    dim: usize,
    input_tensors: &[String],
    output_tensor: &str,
    ctx: &mut Ctx,
) -> Result<NodeOutput> {
    if input_tensors.len() != 3 {
        return Err(NyError::UnsupportedOp(format!(
            "IndexPut requires 3 inputs (data, index, src), got {}",
            input_tensors.len()
        )));
    }
    // Resolve the index FIRST, exactly as `Scatter` does: the runtime-index
    // refusal is the one whose reason has to be stated precisely, and a missing
    // recorded shape must not pre-empt it with a vaguer message.
    let index_name = &input_tensors[1];
    let Some((index_shape, index_values)) = constant_index_values(ctx, index_name, "IndexPut")?
    else {
        return Err(NyError::UnsupportedOp(format!(
            "IndexPut index input '{index_name}' is not a translate-time constant. The write is \
             an OVERWRITE, so a destination named twice keeps whichever write landed last — an \
             order the wire does not fix (nn documents last-write-wins, and nn-import emits this \
             same op for aten.index_put with accumulate=false, which PyTorch documents as \
             nondeterministic on duplicate indices), which makes the destination's value a set \
             of candidates rather than one value. A sound lowering must enclose that whole set, \
             and that requires knowing which destinations collide; with runtime indices the \
             collision pattern is itself a runtime value, so no static layer encloses the result"
        )));
    };

    let data_shape = shape_as_usize(ctx, &input_tensors[0], "IndexPut: data")?;
    let src_shape = shape_as_usize(ctx, &input_tensors[2], "IndexPut: src")?;
    let rank = data_shape.len();
    if dim >= rank {
        return Err(NyError::UnsupportedOp(format!(
            "IndexPut: dim {dim} out of range for data rank {rank}"
        )));
    }

    // nn-core's own preconditions, re-checked (see the doc comment): a 1-D
    // index, and a src that matches data everywhere but `dim`, where it matches
    // the index length.
    if index_shape.len() != 1 {
        return Err(NyError::UnsupportedOp(format!(
            "IndexPut: index '{index_name}' has shape {index_shape:?}, but nn-core's index_put \
             requires a 1-D index (one destination per slice along dim {dim}); a differently \
             shaped index means this node does not describe the op the bridge would lower"
        )));
    }
    if src_shape.len() != rank {
        return Err(NyError::UnsupportedOp(format!(
            "IndexPut: src rank {} must equal data rank {rank} (src shape {src_shape:?}, data \
             shape {data_shape:?}); a rank mismatch is how a multi-axis advanced index reaches \
             the wire with its extra index tensors dropped, so it is refused rather than read \
             as a single-axis write",
            src_shape.len()
        )));
    }
    for (axis, (&src_dim, &data_dim)) in src_shape.iter().zip(data_shape.iter()).enumerate() {
        let expected = if axis == dim {
            index_values.len()
        } else {
            data_dim
        };
        if src_dim != expected {
            return Err(NyError::UnsupportedOp(format!(
                "IndexPut: src dim {axis} size {src_dim} must be {expected} (data shape \
                 {data_shape:?}, index length {}, write axis {dim})",
                index_values.len()
            )));
        }
    }

    let src_elems = checked_shape_product(&src_shape).ok_or_else(|| {
        NyError::InternalError(format!("IndexPut src shape {src_shape:?} overflows usize"))
    })?;
    let coordinate_elems = src_elems.checked_mul(rank).ok_or_else(|| {
        NyError::InternalError("IndexPut ND index element count overflows usize".to_string())
    })?;
    if coordinate_elems > MAX_SCATTER_INDEX_ELEMENTS {
        return Err(NyError::UnsupportedOp(format!(
            "IndexPut would materialize a {coordinate_elems}-element ND index constant, \
             exceeding the {MAX_SCATTER_INDEX_ELEMENTS}-element limit"
        )));
    }

    // Resolve every destination up front so an out-of-range index fails closed
    // even when `src` is empty and the coordinate walk below never reads it.
    let axis_len = data_shape[dim];
    let destinations = index_values
        .iter()
        .map(|&raw| index_put_destination(raw, axis_len))
        .collect::<Result<Vec<_>>>()?;

    // One ND row per src element, in src's row-major order: the element's own
    // coordinate with the write axis replaced by its slice's destination. The
    // ONLY difference from `Scatter` is the lookup — `destinations[coord[dim]]`
    // (per slice) instead of the per-element index value.
    let mut coordinates: Vec<i64> = Vec::new();
    coordinates
        .try_reserve_exact(coordinate_elems)
        .map_err(|e| {
            NyError::ModelLoad(format!(
                "IndexPut ND index allocation failed for {coordinate_elems} elements: {e}"
            ))
        })?;
    let mut coord = vec![0usize; rank];
    for _ in 0..src_elems {
        let destination = destinations[coord[dim]];
        for (axis, &position) in coord.iter().enumerate() {
            let value = if axis == dim { destination } else { position };
            coordinates.push(dim_as_i64(value, "IndexPut ND coordinate")?);
        }
        advance_row_major(&mut coord, &src_shape);
    }

    // `prefix ++ [rank]`, so ScatterND's expected updates shape is exactly
    // `src_shape` and `src` needs no reshape.
    let mut index_tensor_shape = src_shape;
    index_tensor_shape.push(rank);
    let nd_index_tensor = format!("{name}_nd_indices");
    insert_index_constant(ctx, &nd_index_tensor, &index_tensor_shape, coordinates)?;

    Ok(NodeOutput::one(simple_spec(
        name,
        LayerType::ScatterND,
        vec![
            input_tensors[0].clone(),
            nd_index_tensor,
            input_tensors[2].clone(),
        ],
        output_tensor,
        HashMap::new(),
    )))
}

/// Normalize one `IndexPut` destination against the write axis length.
///
/// Unlike [`normalize_scatter_index`] there is NO negative-index convention:
/// nn-core's `index_put` rejects a negative `I64` index with
/// `"index_put: negative index {v}"` and its `U32` path cannot represent one,
/// so a negative value on this wire is not a Python-style index — it is a
/// tensor nn would have refused to execute. Wrapping it would produce bounds on
/// a computation the traced program cannot perform.
fn index_put_destination(raw: i64, axis_len: usize) -> Result<usize> {
    let axis_len_i64 = i64::try_from(axis_len).map_err(|_| {
        NyError::InternalError(format!("IndexPut axis length {axis_len} exceeds i64"))
    })?;
    if raw < 0 {
        return Err(NyError::UnsupportedOp(format!(
            "IndexPut: index {raw} is negative, which nn-core's index_put rejects outright \
             (it has no negative-index convention, unlike Scatter); refusing rather than \
             wrapping it onto a cell nn would never have written"
        )));
    }
    if raw >= axis_len_i64 {
        return Err(NyError::UnsupportedOp(format!(
            "IndexPut: index {raw} out of bounds for axis length {axis_len}"
        )));
    }
    Ok(raw as usize)
}

/// Look up a tensor's recorded shape as `usize` dims.
///
/// Bridge shapes come from `TraceNode::output_shape` (already `usize`), so a
/// NEGATIVE extent can only be a dynamic-dimension marker that survived some
/// other path — refused, since a scatter destination cannot be laid out without
/// real extents. A ZERO extent is a genuine empty dimension and is kept: it
/// makes the scatter a no-op that passes `data` through, which is correct.
fn shape_as_usize(ctx: &Ctx, tensor: &str, context: &str) -> Result<Vec<usize>> {
    lookup_tensor_shape(ctx, tensor, context)?
        .iter()
        .enumerate()
        .map(|(axis, &dim)| {
            if dim < 0 {
                return Err(NyError::UnsupportedOp(format!(
                    "{context}: dimension {axis} of '{tensor}' is dynamic ({dim})"
                )));
            }
            checked_i64_to_usize(dim, &format!("{context} dim {axis}"))
        })
        .collect()
}

/// Read a translate-time-constant index tensor as exact integers.
///
/// Returns `Ok(None)` when the tensor is not a translate-time constant (the
/// caller states the op-specific refusal). Non-integral or out-of-f32-range
/// values are an ERROR, never a rounded index: a write destination that has to
/// be guessed is exactly the case that must fail closed.
fn constant_index_values(
    ctx: &Ctx,
    tensor: &str,
    op_desc: &str,
) -> Result<Option<(Vec<usize>, Vec<i64>)>> {
    if !ctx.constant_tensors.contains(tensor) {
        return Ok(None);
    }
    let Some(values) = ctx.weights.get(tensor) else {
        return Ok(None);
    };
    let shape = values.shape().to_vec();
    let mut integers = Vec::with_capacity(values.len());
    for (position, &value) in values.iter().enumerate() {
        if !value.is_finite() || value.fract() != 0.0 {
            return Err(NyError::UnsupportedOp(format!(
                "{op_desc}: index tensor '{tensor}' element {position} is {value}, which is not an \
                 exact integer; refusing rather than rounding a write destination"
            )));
        }
        // AT 2^24 consecutive integers already stop being f32-representable, so
        // the captured value may be a neighbour of the authored index. The
        // comparison is `>=`, not `>`: 2^24 itself is the FIRST magnitude that
        // is ambiguous, because 2^24 + 1 is not representable and rounds down
        // onto exactly 2^24. Accepting `|value| == 2^24` therefore admits an
        // authored index of +-(2^24 + 1) disguised as +-2^24 — and on the
        // NEGATIVE branch `normalize_scatter_index` adds `axis_len`, so for an
        // axis longer than 2^24 the disguised value normalizes to a real but
        // WRONG destination cell instead of failing the range check. This
        // matches the sibling guard in `insert_index_constant` and the doc
        // phrase "at or beyond" that both share.
        if value.abs() >= F32_EXACT_INTEGER_LIMIT {
            return Err(NyError::UnsupportedOp(format!(
                "{op_desc}: index tensor '{tensor}' element {position} is {value}, at or beyond \
                 the f32 exact-integer limit +/-{F32_EXACT_INTEGER_LIMIT}; the captured value may \
                 not be the authored index"
            )));
        }
        integers.push(value as i64);
    }
    Ok(Some((shape, integers)))
}

/// Largest magnitude at which every integer is exactly f32-representable.
const F32_EXACT_INTEGER_LIMIT: f32 = 16_777_216.0; // 2^24

/// Normalize one scatter destination index against an axis length, applying
/// the Python negative-index convention and refusing anything out of range.
fn normalize_scatter_index(raw: i64, axis_len: usize) -> Result<usize> {
    let axis_len_i64 = i64::try_from(axis_len).map_err(|_| {
        NyError::InternalError(format!("Scatter axis length {axis_len} exceeds i64"))
    })?;
    let normalized = if raw < 0 { axis_len_i64 + raw } else { raw };
    if normalized < 0 || normalized >= axis_len_i64 {
        return Err(NyError::UnsupportedOp(format!(
            "Scatter: index {raw} out of bounds for axis length {axis_len}"
        )));
    }
    Ok(normalized as usize)
}

/// Step a row-major odometer over `shape`. Wraps to all-zero after the last
/// coordinate, which the callers never observe (they stop after `N` steps).
fn advance_row_major(coord: &mut [usize], shape: &[usize]) {
    for axis in (0..coord.len()).rev() {
        coord[axis] += 1;
        if coord[axis] < shape[axis] {
            return;
        }
        coord[axis] = 0;
    }
}

/// Register a synthesized ND-coordinate constant as an f32 weight.
///
/// The bridge's weight surface is f32 throughout — `GraphModelBuilder` carries
/// only that map — so the coordinates travel as floats and ny-build's
/// `discrete_constant_i64` reads them back through its float fallback, which
/// accepts a value only when it is finite, exactly integral, and inside the
/// non-saturating cast range.
///
/// That round trip is EXACT here, not merely usually-exact: every coordinate is
/// a destination index bounded by the data tensor's own extents, and the guard
/// below refuses anything at or beyond 2²⁴ — the magnitude where consecutive
/// integers stop being f32-representable. Below it `i64 -> f32 -> i64` is the
/// identity, so no coordinate can drift onto a neighbouring cell.
///
/// The insert also marks the tensor constant, which is what keeps
/// `find_activation_inputs` from turning it into a graph input.
fn insert_index_constant(
    ctx: &mut Ctx,
    tensor_name: &str,
    shape: &[usize],
    values: Vec<i64>,
) -> Result<()> {
    let mirror: Vec<f32> = values
        .iter()
        .map(|&v| {
            if v.unsigned_abs() >= F32_EXACT_INTEGER_LIMIT as u64 {
                return Err(NyError::UnsupportedOp(format!(
                    "{tensor_name}: destination coordinate {v} is at or beyond the f32 \
                     exact-integer limit {F32_EXACT_INTEGER_LIMIT}, which the f32 weight surface \
                     cannot carry without risking a different cell"
                )));
            }
            Ok(v as f32)
        })
        .collect::<Result<_>>()?;
    let floats = ArrayD::from_shape_vec(IxDyn(shape), mirror)
        .map_err(|e| NyError::InternalError(format!("{tensor_name} index shape mismatch: {e}")))?;
    ctx.insert_weight(tensor_name, floats)
}

// ---------------------------------------------------------------------------
// Local spec helpers (mirror NN's decompose_scan/decompose module-privates;
// not in the bridge's shared mod.rs — dedupe later if another family needs
// them)
// ---------------------------------------------------------------------------

/// Convert `i64` to `usize`, rejecting negatives/overflow. NN keeps this in
/// `graph_tensor`; the bridge has no shared equivalent yet.
fn checked_i64_to_usize(val: i64, context: &str) -> Result<usize> {
    usize::try_from(val)
        .map_err(|_| NyError::InternalError(format!("{context}: dimension {val} is negative")))
}

fn slice_spec(
    name: &str,
    inputs: Vec<String>,
    output: &str,
    axis: i64,
    start: usize,
    end: usize,
) -> Result<LayerSpec> {
    let mut attrs = HashMap::new();
    attrs.insert("axis".to_string(), AttributeValue::Int(axis));
    attrs.insert(
        "start".to_string(),
        AttributeValue::Int(dim_as_i64(start, "slice start")?),
    );
    attrs.insert(
        "end".to_string(),
        AttributeValue::Int(dim_as_i64(end, "slice end")?),
    );
    Ok(simple_spec(name, LayerType::Slice, inputs, output, attrs))
}

fn concat_spec(name: &str, inputs: Vec<String>, output: &str, axis: i64) -> LayerSpec {
    let mut attrs = HashMap::new();
    attrs.insert("axis".to_string(), AttributeValue::Int(axis));
    simple_spec(name, LayerType::Concat, inputs, output, attrs)
}

/// Build a Tile LayerSpec with `axis` + `reps` attributes (duplicate of
/// `ops_pad_resample`'s module-private helper; dedupe into mod.rs when a
/// third family needs it).
fn tile_spec(
    name: &str,
    inputs: Vec<String>,
    output: &str,
    axis: i64,
    reps: usize,
) -> Result<LayerSpec> {
    let mut attrs = HashMap::new();
    attrs.insert("axis".to_string(), AttributeValue::Int(axis));
    attrs.insert(
        "reps".to_string(),
        AttributeValue::Int(dim_as_i64(reps, "tile reps")?),
    );
    Ok(simple_spec(name, LayerType::Tile, inputs, output, attrs))
}

fn reshape_to_output(
    name: &str,
    input: String,
    output: &str,
    shape: &[usize],
) -> Result<LayerSpec> {
    let mut attrs = HashMap::new();
    attrs.insert(
        "shape".to_string(),
        AttributeValue::Ints(shape_to_i64(shape, name)?),
    );
    Ok(simple_spec(
        name,
        LayerType::Reshape,
        vec![input],
        output,
        attrs,
    ))
}

/// Look up a tensor's recorded shape, mirroring NN's
/// `ctx.tensor_shapes.get(...)` lookups with the same error contexts.
fn lookup_tensor_shape(ctx: &Ctx, tensor: &str, context: &str) -> Result<Vec<i64>> {
    ctx.tensor_shapes.get(tensor).cloned().ok_or_else(|| {
        NyError::InternalError(format!("{context} input shape not found for {tensor}"))
    })
}

// ---------------------------------------------------------------------------
// Comparisons
// ---------------------------------------------------------------------------

/// `CompareOp` -> the `compare_op` attribute string ny-build's converter
/// expects. Exhaustive: the schema enum is NY-owned, so a new variant is a
/// compile error here — never a silently substituted operator.
fn compare_op_attr(op: CompareOp) -> &'static str {
    match op {
        CompareOp::Gt => "Gt",
        CompareOp::Ge => "Ge",
        CompareOp::Lt => "Lt",
        CompareOp::Le => "Le",
        CompareOp::Eq => "Eq",
        CompareOp::Ne => "Ne",
    }
}

/// Translate `TraceOp::Compare { op, value }`: a 2-input Compare LayerSpec
/// with the scalar threshold as a constant weight and the `compare_op`
/// attribute (mirrors NN's `translate_compare_scalar`).
fn translate_compare_scalar(
    name: &str,
    op: CompareOp,
    value: f64,
    input_tensors: &[String],
    output_tensor: &str,
    ctx: &mut Ctx,
) -> Result<NodeOutput> {
    let data_input = first_input(input_tensors, "Compare")?;
    let threshold = checked_f64_to_f32(value, "Compare threshold")?;
    let const_name = format!("{name}_threshold");
    insert_scalar_constant(ctx, &const_name, threshold)?;
    let mut attrs = HashMap::new();
    attrs.insert(
        "compare_op".to_string(),
        AttributeValue::String(compare_op_attr(op).to_string()),
    );
    Ok(NodeOutput::one(simple_spec(
        name,
        LayerType::Compare,
        vec![data_input, const_name],
        output_tensor,
        attrs,
    )))
}

/// Translate `TraceOp::CompareTensor { op }`: element-wise comparison of two
/// activation inputs (mirrors NN's `translate_compare_tensor`).
fn translate_compare_tensor(
    name: &str,
    op: CompareOp,
    input_tensors: &[String],
    output_tensor: &str,
) -> Result<NodeOutput> {
    if input_tensors.len() < 2 {
        return Err(NyError::UnsupportedOp(
            "CompareTensor requires two inputs".to_string(),
        ));
    }
    let mut attrs = HashMap::new();
    attrs.insert(
        "compare_op".to_string(),
        AttributeValue::String(compare_op_attr(op).to_string()),
    );
    Ok(NodeOutput::one(simple_spec(
        name,
        LayerType::CompareTensor,
        input_tensors.to_vec(),
        output_tensor,
        attrs,
    )))
}

/// Translate `TraceOp::WhereCond` — a 3-input `LayerType::Where` spec, with
/// exact `Clip[v, v]` activation carriers for uniform constant inputs.
///
/// nn records the inputs as `[condition, on_true, on_false]` (nn-core
/// `selection/where_cond.rs`: `trace_input_ids(&[self, on_true, on_false])`
/// where `self` is the mask), which is exactly the order ny-propagate's
/// `WhereLayer::propagate_ibp_ternary(condition, x, y)` consumes; the graph
/// runtime routes 3-input Where nodes through that ternary rule
/// (`classify_node_inputs`). The union IBP — per-element
/// `[min(x_l, y_l), max(x_u, y_u)]`, NaN-propagating — encloses the select
/// for ANY realizable condition value, binary or not, because the layer never
/// evaluates the `m*x + (1-m)*y` mask arithmetic whose non-binary-mask
/// unsoundness motivated the old refusal. IBP-only:
/// `WhereLayer::propagate_linear` fails closed (a data-dependent select has
/// no useful CROWN linear relaxation), matching the Compare/CompareTensor
/// precedent.
///
/// ## Constant inputs
///
/// The graph builder wires a Where node in exactly two shapes: all three
/// inputs as activations, or BOTH branches constant (embedded into the layer
/// via `WhereLayer::with_constants`, condition as the sole activation).
/// Anything else filters the constant out of activation wiring and fails the
/// ternary arity check at build time. The bridge therefore:
/// - emits the spec verbatim when both branches are translate-time constants
///   (the builder's embedded path — also the tighter one: it selects exactly
///   when the condition is determined);
/// - gives any OTHER constant input (a single constant branch — nn's
///   `masked_fill` sugar traces `full_like` fills this way — or a constant
///   condition) an activation CARRIER when its captured tensor is UNIFORM
///   (every element the same finite `v`): `Clip[v, v]` applied to a variable
///   branch, the x^0 lowering's constant-producing shape. Clip's IBP is
///   `[clamp(l), clamp(u)] = [v, v]` regardless of the carrier input's
///   bounds — exact — and the carrier inherits the variable branch's shape,
///   which is the shape `WhereLayer` requires the branches to share (the
///   union IBP never reads the condition's shape or values, so a
///   branch-shaped condition carrier is equally sound);
/// - refuses a NON-uniform single constant fail-closed: it has no exact
///   activation carrier, and emitting the raw spec would only defer the same
///   failure to graph build.
fn translate_where_cond(
    name: &str,
    input_tensors: &[String],
    output_tensor: &str,
    ctx: &Ctx,
) -> Result<NodeOutput> {
    if input_tensors.len() < 3 {
        return Err(NyError::UnsupportedOp(format!(
            "WhereCond requires 3 inputs (condition, on_true, on_false), got {}",
            input_tensors.len()
        )));
    }
    let is_const = |t: &str| ctx.constant_tensors.contains(t);
    let on_true = &input_tensors[1];
    let on_false = &input_tensors[2];

    // Both branches constant: the builder embeds them into the layer.
    if is_const(on_true) && is_const(on_false) {
        return Ok(NodeOutput::one(simple_spec(
            name,
            LayerType::Where,
            input_tensors[..3].to_vec(),
            output_tensor,
            HashMap::new(),
        )));
    }

    // At least one branch is variable here; it anchors every carrier's shape.
    let source = if is_const(on_true) {
        on_false.clone()
    } else {
        on_true.clone()
    };
    let mut specs = Vec::new();
    let mut wired = Vec::with_capacity(3);
    for (role, tensor) in [
        ("condition", &input_tensors[0]),
        ("on_true", on_true),
        ("on_false", on_false),
    ] {
        if !is_const(tensor) {
            wired.push(tensor.clone());
            continue;
        }
        let values = ctx.weights.get(tensor).ok_or_else(|| {
            NyError::InternalError(format!(
                "WhereCond {role} input {tensor} marked constant but absent from the weight store"
            ))
        })?;
        let Some(&first) = values.iter().next() else {
            return Err(NyError::UnsupportedOp(format!(
                "WhereCond: constant {role} input {tensor} is empty"
            )));
        };
        if !first.is_finite() || values.iter().any(|&v| v != first) {
            return Err(NyError::UnsupportedOp(format!(
                "WhereCond: constant {role} input {tensor} is not a uniform finite tensor; the \
                 graph runtime embeds Where constants only when both branches are constant, and \
                 a non-uniform single-side constant has no exact activation carrier"
            )));
        }
        let carrier_name = format!("{name}_{role}_carrier");
        let carrier_out = format!("{carrier_name}_out");
        let mut attrs = HashMap::new();
        attrs.insert("min".to_string(), AttributeValue::Float(first));
        attrs.insert("max".to_string(), AttributeValue::Float(first));
        specs.push(simple_spec(
            &carrier_name,
            LayerType::Clip,
            vec![source.clone()],
            &carrier_out,
            attrs,
        ));
        wired.push(carrier_out);
    }
    specs.push(simple_spec(
        name,
        LayerType::Where,
        wired,
        output_tensor,
        HashMap::new(),
    ));
    Ok(NodeOutput { specs })
}

// ---------------------------------------------------------------------------
// Selection family (Argmax / Argmin / ArgSort / Sort; Topk stays refused)
// ---------------------------------------------------------------------------

/// Translate `TraceOp::Argmax` / `TraceOp::Argmin` to the native index layer.
///
/// One spec, three pinned attributes:
/// - `axis`: trailing-relative vs the DATA tensor's rank (the reduction
///   removes the axis, so the output rank is one lower — the data rank comes
///   from the recorded input shape, mirroring [`translate_gather`]).
/// - `keepdims = 0`: nn records the trace op inside `argreduce_impl` BEFORE
///   the `argmax_keepdim`/`argmin_keepdim` `unsqueeze` (which traces as its
///   own node), so a traced Argmax/Argmin output shape is ALWAYS the reduced
///   shape.
/// - `select_last_index = 0`: nn's tie policy is first-index on every wired
///   backend — CPU `if v > best_val` / `v < best_val` (strict), Metal
///   `val > best_val` / `val < best_val` (strict), and CUDA has no
///   `GpuFullBackend` argreduce override so it falls back to the CPU path.
///   ny-propagate's `exact_argext_index` claims a tie-broken exact index only
///   under this policy; a mismatch would be a false proof on exact ties.
///
/// The layer's IBP emits the exact index when one element provably dominates
/// the lane over the whole input box, and the sound `[0, n-1]` hull
/// otherwise; indices above 2²⁴ get an outward two-endpoint f32 enclosure
/// (`index_interval`). The traced output dtype is U32 — the graph-wide F32
/// gate must carve that out before end-to-end translation reaches this arm.
fn translate_arg_extremum(
    op_desc: &str,
    layer_type: LayerType,
    name: &str,
    dim: usize,
    input_tensors: &[String],
    output_tensor: &str,
    ctx: &Ctx,
) -> Result<NodeOutput> {
    let data_input = first_input(input_tensors, op_desc)?;
    let data_rank = lookup_tensor_shape(ctx, &data_input, op_desc)?.len();
    let axis = super::trailing_axis(dim, data_rank, &format!("{op_desc} axis"))?;
    let mut attrs = HashMap::new();
    attrs.insert("axis".to_string(), AttributeValue::Int(axis));
    attrs.insert("keepdims".to_string(), AttributeValue::Int(0));
    attrs.insert("select_last_index".to_string(), AttributeValue::Int(0));
    Ok(NodeOutput::one(simple_spec(
        name,
        layer_type,
        vec![data_input],
        output_tensor,
        attrs,
    )))
}

/// Translate `TraceOp::ArgSort` to the native `LayerType::ArgSort` layer.
///
/// Rank-preserving, so `axis` is trailing-relative vs the output rank;
/// `descending` maps straight through (nn serializes it —
/// `TraceOp::ArgSort { dim, descending: !ascending }`).
///
/// Soundness: `ArgSortLayer` emits the exact permutation ONLY when every
/// adjacent pair in the resolved order is strictly separated by intervals —
/// in that regime the permutation is unique under ANY tie policy, which
/// matters because nn's `arg_sort` uses `sort_unstable_by` (tie order
/// unspecified). Any ambiguity, tie, or non-finite bound falls back to the
/// sound uniform `[0, n-1]` index hull per lane. The traced output dtype is
/// U32 — same gate carve-out note as [`translate_arg_extremum`].
fn translate_argsort(
    name: &str,
    dim: usize,
    descending: bool,
    input_tensors: &[String],
    output_tensor: &str,
    output_shape: &[usize],
) -> Result<NodeOutput> {
    let data_input = first_input(input_tensors, "ArgSort")?;
    let axis = super::trailing_axis(dim, output_shape.len(), "ArgSort axis")?;
    let mut attrs = HashMap::new();
    attrs.insert("axis".to_string(), AttributeValue::Int(axis));
    attrs.insert(
        "descending".to_string(),
        AttributeValue::Int(i64::from(descending)),
    );
    Ok(NodeOutput::one(simple_spec(
        name,
        LayerType::ArgSort,
        vec![data_input],
        output_tensor,
        attrs,
    )))
}

/// Translate `TraceOp::Sort` (sorted VALUES) via `LayerType::Topk` with
/// `k = n`, `output = "values"`.
///
/// nn traces only the values tensor of `DynTensor::sort` (both the CPU and
/// GPU-bitonic paths record the op on `values`; the indices tensor gets no
/// trace id), and `descending` IS serialized — unlike Topk's largest/sorted
/// flags — so the traced function is unambiguous.
///
/// Soundness/exactness: the p-th order statistic is monotone
/// (non-decreasing) in every input coordinate, so over an interval box its
/// minimum is attained at the lower-bound vector and its maximum at the
/// upper-bound vector. `TopkLayer`'s values IBP sorts each bound lane
/// descending independently — with `k = n` that is exactly
/// `[sorted_desc(lower)[p], sorted_desc(upper)[p]]` per output position:
/// EXACT for `descending = true`. Ascending output is the elementwise
/// reversal of the descending output (values are tie-agnostic: equal values
/// sort equal under any tie policy), lowered as the Flip-style N × Slice +
/// Concat reversal and therefore capped at [`MAX_DECOMPOSE_DIM`]; the
/// single-spec descending form is uncapped. Non-finite input bounds collapse
/// to the sound whole-lane `[-inf, +inf]` fallback inside the layer.
/// IBP-only (`TopkLayer::propagate_linear` fails closed).
fn translate_sort(
    name: &str,
    dim: usize,
    descending: bool,
    input_tensors: &[String],
    output_tensor: &str,
    output_shape: &[usize],
) -> Result<NodeOutput> {
    let data_input = first_input(input_tensors, "Sort")?;
    let rank = output_shape.len();
    if dim >= rank {
        return Err(NyError::UnsupportedOp(format!(
            "Sort: dim {dim} >= rank {rank}"
        )));
    }
    let n = output_shape[dim];
    if n == 0 {
        return Err(NyError::UnsupportedOp("Sort: dim size is 0".to_string()));
    }
    let axis = super::trailing_axis(dim, rank, "Sort axis")?;

    let topk = |spec_name: &str, spec_out: &str| -> Result<LayerSpec> {
        let mut attrs = HashMap::new();
        attrs.insert("axis".to_string(), AttributeValue::Int(axis));
        attrs.insert(
            "k".to_string(),
            AttributeValue::Int(dim_as_i64(n, "Sort k")?),
        );
        attrs.insert(
            "output".to_string(),
            AttributeValue::String("values".to_string()),
        );
        Ok(simple_spec(
            spec_name,
            LayerType::Topk,
            vec![data_input.clone()],
            spec_out,
            attrs,
        ))
    };

    // Descending — and n == 1, where both orders coincide: one Topk spec.
    if descending || n == 1 {
        return Ok(NodeOutput::one(topk(name, output_tensor)?));
    }

    // Ascending: descending Topk + elementwise reversal along the axis.
    if n > MAX_DECOMPOSE_DIM {
        return Err(NyError::UnsupportedOp(format!(
            "Sort (ascending): dim size {n} exceeds decomposition limit {MAX_DECOMPOSE_DIM} \
             (the reversal is N x Slice + Concat)"
        )));
    }
    let desc_name = format!("{name}_desc");
    let desc_out = format!("{desc_name}_out");
    let mut specs = vec![topk(&desc_name, &desc_out)?];
    let mut slice_outs = Vec::new();
    for i in (0..n).rev() {
        let sn = format!("{name}_s{i}");
        let so = format!("{sn}_out");
        specs.push(slice_spec(
            &sn,
            vec![desc_out.clone()],
            &so,
            axis,
            i,
            i + 1,
        )?);
        slice_outs.push(so);
    }
    specs.push(concat_spec(name, slice_outs, output_tensor, axis));
    Ok(NodeOutput { specs })
}

/// Translate `TraceOp::Topk` — but ONLY the one selection semantics
/// `TopkLayer` actually models.
///
/// # Which combination lowers, and why exactly that one
///
/// `TopkLayer::propagate_values_ibp` (ny-propagate
/// `layers/reduction/selection.rs`) sorts the lower-bound lane and the
/// upper-bound lane **descending, independently**, and writes the first `k` of
/// each into output positions `0..k`. Output position `j` is therefore claimed
/// to hold the `(j+1)`-th LARGEST value of the lane. That is the
/// `largest = true, sorted = true` function and no other:
///
/// * **`largest && sorted` — LOWERED, and EXACT.** The `j`-th order statistic
///   is monotone (non-decreasing) in every input coordinate, so over an
///   interval box its minimum is attained at the lower-bound vector and its
///   maximum at the upper-bound vector: `sorted_desc(l)[j] <= sorted_desc(x)[j]
///   <= sorted_desc(u)[j]` for every `x` in the box. Those are precisely the
///   two endpoints the layer emits, so the interval ENCLOSES the traced value
///   at every position and is the tightest interval that does.
/// * **`largest && !sorted` — REFUSED.** The traced tensor holds the same k
///   values in an *unspecified* order, so position `j` may hold ANY of the k
///   largest. Enclosing that needs the hull over all k of them; the layer's
///   per-position `[sorted_desc(l)[j], sorted_desc(u)[j]]` is strictly
///   narrower. Concretely, an exact lane `[3, 1]` with `k = 2` gives position 0
///   the degenerate interval `[3, 3]`, which does not contain the `1` an
///   unsorted kernel is free to place there — a FALSE PROOF, not a loose one.
/// * **`!largest` (either `sorted`) — REFUSED.** The k SMALLEST values. The
///   layer would bound them by the k largest, and the two selections share no
///   value in general (lane `[0, 1, 2, 3]`, `k = 2`: smallest `{0,1}` vs
///   largest `{3,2}`), so the emitted interval can miss the traced value
///   entirely.
///
/// The refusals name the flag combination rather than the op, because the op
/// itself is no longer the problem — the wire now says which function ran, and
/// three of the four have no layer to land on.
///
/// # Lowering
///
/// One `LayerType::Topk` spec with `k`, the trailing-relative `axis`, and
/// `output = "values"` — the same spec shape [`translate_sort`] emits with
/// `k = n`. nn traces only the VALUES tensor (`topk_ext` records the op on
/// `values`; the indices tensor gets no trace id), which is why no `indices`
/// variant is emitted here. IBP-only: `TopkLayer::propagate_linear` fails
/// closed, so a Topk node degrades the graph to IBP at that node.
///
/// Non-finite input bounds collapse to the sound whole-lane `[-inf, +inf]`
/// fallback inside the layer.
#[allow(clippy::too_many_arguments)]
fn translate_topk(
    name: &str,
    k: usize,
    dim: usize,
    largest: bool,
    sorted: bool,
    input_tensors: &[String],
    output_tensor: &str,
    output_shape: &[usize],
) -> Result<NodeOutput> {
    if !largest {
        return Err(NyError::UnsupportedOp(format!(
            "Topk with largest=false (k smallest, sorted={sorted}) cannot be lowered: \
             TopkLayer's values IBP sorts each bound lane DESCENDING and keeps the first k, so it \
             models the k LARGEST values only. The k smallest and the k largest of a lane share \
             no value in general (lane [0,1,2,3], k=2: smallest {{0,1}} vs largest {{3,2}}), so \
             bounding one by the other can exclude the traced value — a false proof rather than a \
             loose one. Only largest=true, sorted=true lowers"
        )));
    }
    if !sorted {
        return Err(NyError::UnsupportedOp(
            "Topk with sorted=false cannot be lowered: the traced tensor holds the k largest \
             values in an UNSPECIFIED order, so output position j may hold any of them, while \
             TopkLayer bounds position j by the j-th largest bound specifically. That per-position \
             interval is strictly narrower than the hull the unsorted semantics needs — an exact \
             lane [3,1] with k=2 gives position 0 the degenerate interval [3,3], which does not \
             contain the 1 an unsorted kernel may place there. Only largest=true, sorted=true \
             lowers"
                .to_string(),
        ));
    }

    let data_input = first_input(input_tensors, "Topk")?;
    let rank = output_shape.len();
    if dim >= rank {
        return Err(NyError::UnsupportedOp(format!(
            "Topk: dim {dim} >= rank {rank}"
        )));
    }
    if k == 0 {
        return Err(NyError::UnsupportedOp("Topk: k is 0".to_string()));
    }
    // The recorded output extent along `dim` IS k; a disagreement means the
    // trace contradicts the op it records, so fail closed rather than trust
    // either number.
    if output_shape[dim] != k {
        return Err(NyError::InternalError(format!(
            "Topk: recorded output extent {} along dim {dim} contradicts the op's k={k}",
            output_shape[dim]
        )));
    }
    let axis = super::trailing_axis(dim, rank, "Topk axis")?;

    let mut attrs = HashMap::new();
    attrs.insert("axis".to_string(), AttributeValue::Int(axis));
    attrs.insert(
        "k".to_string(),
        AttributeValue::Int(dim_as_i64(k, "Topk k")?),
    );
    attrs.insert(
        "output".to_string(),
        AttributeValue::String("values".to_string()),
    );
    Ok(NodeOutput::one(simple_spec(
        name,
        LayerType::Topk,
        vec![data_input],
        output_tensor,
        attrs,
    )))
}

#[cfg(test)]
mod tests {
    use ny_build::{AttributeValue, GraphModel, LayerSpec};
    use ny_core::{LayerType, NyError};

    use crate::schema::{
        CompareOp, ComputationGraph, DType, NodeId, TraceNode, TraceOp, WeightPayload,
    };
    use crate::translate::{translate, translate_multi_input};

    fn node(id: u64, name: &str, op: TraceOp, inputs: &[u64], shape: &[usize]) -> TraceNode {
        TraceNode::new(
            NodeId(id),
            name,
            op,
            inputs.iter().map(|&i| NodeId(i)).collect(),
            shape.to_vec(),
            DType::F32,
        )
    }

    fn count(model: &GraphModel, lt: &LayerType) -> usize {
        model
            .network
            .layers
            .iter()
            .filter(|l| &l.layer_type == lt)
            .count()
    }

    fn layer<'a>(model: &'a GraphModel, name: &str) -> &'a LayerSpec {
        model
            .network
            .layers
            .iter()
            .find(|l| l.name == name)
            .unwrap_or_else(|| panic!("layer {name} not found"))
    }

    fn assert_builds(model: &GraphModel, what: &str) {
        model
            .build_graph_network(ny_build::GraphNetworkOptions::default())
            .unwrap_or_else(|e| panic!("{what}: GraphModel must build a graph network: {e:?}"));
    }

    /// SwiGlu decomposes into SiLU(gate) + Mul(silu_out, up) — NN #3557.
    #[test]
    fn swiglu_decomposes_to_silu_mul() {
        // gate = relu(x), up = x — both derive from the single input.
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[4]),
            node(1, "gate", TraceOp::Relu, &[0], &[4]),
            node(2, "sg", TraceOp::SwiGlu, &[1, 0], &[4]),
        ]);
        let model = translate(&graph).expect("SwiGlu translates");
        assert_eq!(count(&model, &LayerType::SiLU), 1, "one SiLU");
        let silu = layer(&model, "layer0_trace_2_silu");
        assert_eq!(silu.inputs, vec!["layer0_trace_1_out".to_string()]);
        let mul = layer(&model, "layer0_trace_2");
        assert_eq!(mul.layer_type, LayerType::Mul);
        assert_eq!(
            mul.inputs,
            vec![
                "layer0_trace_2_silu_out".to_string(),
                "layer0_trace_0_out".to_string()
            ]
        );
        assert_builds(&model, "SwiGlu");
    }

    /// SwiGlu with a single input is refused (needs gate + up).
    #[test]
    fn swiglu_single_input_refused() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[4]),
            node(1, "sg", TraceOp::SwiGlu, &[0], &[4]),
        ]);
        let err = translate(&graph).expect_err("1-input SwiGlu refused");
        assert!(
            matches!(err, NyError::UnsupportedOp(ref m) if m.contains("SwiGlu requires 2 inputs")),
            "got {err:?}"
        );
    }

    /// Powf special-case exponents map to NN's exact primitives.
    #[test]
    fn powf_special_cases_match_nn() {
        let build = |exponent: f64| {
            let graph = ComputationGraph::from_nodes(vec![
                node(0, "x", TraceOp::Input, &[], &[4]),
                node(1, "p", TraceOp::Powf { exponent }, &[0], &[4]),
            ]);
            translate(&graph).expect("Powf translates")
        };

        // x^0 → Clip[1,1].
        let m = build(0.0);
        let clip = layer(&m, "layer0_trace_1");
        assert_eq!(clip.layer_type, LayerType::Clip);
        assert_eq!(
            clip.attributes.get("min"),
            Some(&AttributeValue::Float(1.0))
        );
        assert_eq!(
            clip.attributes.get("max"),
            Some(&AttributeValue::Float(1.0))
        );

        // x^1 → identity (Add + 0), no dtype-cast Clip.
        let m = build(1.0);
        assert_eq!(layer(&m, "layer0_trace_1").layer_type, LayerType::Add);
        assert_eq!(count(&m, &LayerType::Clip), 0);

        // x^2 → Pow with power=2 attr and the _pow2 constant.
        let m = build(2.0);
        let pow = layer(&m, "layer0_trace_1");
        assert_eq!(pow.layer_type, LayerType::Pow);
        assert_eq!(
            pow.attributes.get("power"),
            Some(&AttributeValue::Float(2.0))
        );
        assert!(m.weights.contains_key("layer0_trace_1_pow2"));

        // x^0.5 → Sqrt.
        let m = build(0.5);
        assert_eq!(layer(&m, "layer0_trace_1").layer_type, LayerType::Sqrt);

        // x^-1 → Reciprocal.
        let m = build(-1.0);
        assert_eq!(
            layer(&m, "layer0_trace_1").layer_type,
            LayerType::Reciprocal
        );

        // x^-0.5 → Sqrt + Reciprocal.
        let m = build(-0.5);
        assert_eq!(layer(&m, "layer0_trace_1_sqrt").layer_type, LayerType::Sqrt);
        assert_eq!(
            layer(&m, "layer0_trace_1").layer_type,
            LayerType::Reciprocal
        );
        assert_builds(&m, "Powf(-0.5)");
    }

    /// General exponents lower to the x² layer shape: `LayerType::Pow` with
    /// the (f32-narrowed) exponent as a scalar constant input, converted by
    /// ny-build to ny-propagate's parity-aware `PowConstantLayer`.
    #[test]
    fn powf_general_exponents_lower_to_pow_constant() {
        // Integer (odd/even, positive/negative) and non-integer exponents.
        // The non-integer ones are sound to emit because `PowConstantLayer`
        // refuses negative input domains fail-closed at propagation time.
        for exponent in [3.0, 4.0, -2.0, -3.0, 1.0 / 3.0, 1.5, -1.5] {
            let graph = ComputationGraph::from_nodes(vec![
                node(0, "x", TraceOp::Input, &[], &[4]),
                node(1, "p", TraceOp::Powf { exponent }, &[0], &[4]),
            ]);
            let m = translate(&graph).expect("general Powf translates");
            let pow = layer(&m, "layer0_trace_1");
            assert_eq!(pow.layer_type, LayerType::Pow, "{exponent}");
            assert_eq!(
                pow.attributes.get("power"),
                Some(&AttributeValue::Float(exponent as f32)),
                "{exponent}"
            );
            assert_eq!(
                pow.inputs,
                vec![
                    "layer0_trace_0_out".to_string(),
                    "layer0_trace_1_pow_exp".to_string()
                ],
                "{exponent}"
            );
            let w = m
                .weights
                .get("layer0_trace_1_pow_exp")
                .expect("exponent stored as scalar constant");
            assert_eq!(w.iter().copied().collect::<Vec<_>>(), vec![exponent as f32]);
            assert_builds(&m, "Powf general");
        }
    }

    /// Non-finite exponents and exponents beyond the f32-exact-integer cap
    /// (|p| > 2^24, where narrowing could flip integrality or parity) still
    /// refuse fail-closed.
    #[test]
    fn powf_general_exponents_fail_closed() {
        for exponent in [
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            16_777_218.0,
            -16_777_218.0,
            1e300,
        ] {
            let graph = ComputationGraph::from_nodes(vec![
                node(0, "x", TraceOp::Input, &[], &[4]),
                node(1, "p", TraceOp::Powf { exponent }, &[0], &[4]),
            ]);
            let err = translate(&graph).expect_err("unlowerable Powf must fail closed");
            assert!(
                matches!(err, NyError::UnsupportedOp(ref message)
                    if message.contains("Powf exponent")
                        && message.contains("no sound lowering")),
                "{exponent:?}: got {err:?}"
            );
        }
    }

    /// `PowConstantLayer` classifies integrality with a 1e-6 tolerance, so a
    /// near-integral-but-not-integral exponent (e.g. 3.0000005) would be
    /// bounded as the exact odd cube — admitting negative domains where the
    /// traced `powf` computed NaN. The bridge must refuse that ambiguous band;
    /// exactly-integral and clearly-fractional exponents keep lowering.
    #[test]
    fn powf_ambiguous_integrality_band_is_refused() {
        for exponent in [
            f64::from(3.000_000_5_f32),
            f64::from(-7.000_000_5_f32),
            f64::from(2.000_000_2_f32),
        ] {
            let narrowed = exponent as f32;
            assert_ne!(narrowed, narrowed.round(), "fixture must not be integral");
            assert!(
                (narrowed - narrowed.round()).abs() < 1e-6,
                "fixture must sit inside the layer's tolerance band"
            );
            let graph = ComputationGraph::from_nodes(vec![
                node(0, "x", TraceOp::Input, &[], &[4]),
                node(1, "p", TraceOp::Powf { exponent }, &[0], &[4]),
            ]);
            let err = translate(&graph).expect_err("ambiguous integrality must fail closed");
            assert!(
                matches!(err, NyError::UnsupportedOp(ref message)
                    if message.contains("integer-classification tolerance")),
                "{exponent:?}: got {err:?}"
            );
        }
    }

    #[test]
    fn fract_lowers_to_floor_sub_and_encloses_nns_value_on_negatives() {
        // fract(x) = x - FLOOR(x), because `TraceOp::Fract` is NN's op and NN
        // defines it floor-based, range [0, 1). NN's IR doc says so in as many
        // words — "differs from Rust `f32::fract()` which uses trunc …
        // `fract(-1.7)` = 0.3 (floor-based), not -0.7" — and its CPU reference
        // (`x - x.floor()`) and Metal kernel (`metal::fract`) agree.
        //
        // This arm previously emitted `Sub(x, Trunc(x))`, and the test that
        // pinned it had the truth backwards: it called the trunc identity
        // "the truth" and the floor result "disjoint from" it. On the box
        // below the two are indeed disjoint — and it is the TRUNC form that
        // misses. NN's reachable set there is [0.5, 0.8]; the trunc lowering
        // claimed [-0.5, -0.2], sharing not one point with it. A false proof,
        // and the maximally bad kind: not a loose bound, a disjoint one.
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[1]),
            node(1, "f", TraceOp::Fract, &[0], &[1]),
        ]);
        let model = translate(&graph).expect("Fract lowers via Sub(x, Floor(x))");
        assert_eq!(count(&model, &LayerType::Floor), 1);
        assert_eq!(
            count(&model, &LayerType::Trunc),
            0,
            "trunc is the wrong convention"
        );
        assert_eq!(count(&model, &LayerType::Sub), 1);
        let network = model
            .build_graph_network(ny_build::GraphNetworkOptions::default())
            .expect("Fract composition builds");

        // Single period, negative domain: floor == -2 on all of [-1.5, -1.2],
        // so the composition is EXACT.
        let bounds = ny_tensor::BoundedTensor::new(
            ndarray::arr1(&[-1.5f32]).into_dyn(),
            ndarray::arr1(&[-1.2f32]).into_dyn(),
        )
        .expect("bounds");
        let out = network.propagate_ibp(&bounds).expect("Fract IBP");
        let lo = out.lower()[[0]];
        let hi = out.upper()[[0]];
        // Compare against NN's own computation at f32 inputs, not decimal
        // literals, with one-ULP outward slack.
        let true_lo = -1.5f32 - (-1.5f32).floor();
        let true_hi = -1.2f32 - (-1.2f32).floor();
        assert!(
            lo <= true_lo && lo >= true_lo - 1e-6,
            "lower {lo} must enclose NN's {true_lo} tightly"
        );
        assert!(
            hi >= true_hi && hi <= true_hi + 1e-6,
            "upper {hi} must enclose NN's {true_hi} tightly"
        );

        // The regression itself: the old trunc lowering's interval must not be
        // what we emit. trunc(-1.5) = -1, so it would have produced [-0.5, -0.2].
        assert!(
            lo > 0.0,
            "a trunc-based lowering yields a negative interval here; got [{lo}, {hi}]"
        );
    }

    /// Atan2 → binary LayerType::Atan2 spec (inputs y, x) — NN binary arm.
    #[test]
    fn atan2_maps_to_atan2_layer() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[4]),
            node(1, "y", TraceOp::Relu, &[0], &[4]),
            node(2, "a", TraceOp::Atan2, &[1, 0], &[4]),
        ]);
        let model = translate(&graph).expect("Atan2 translates");
        let a2 = layer(&model, "layer0_trace_2");
        assert_eq!(a2.layer_type, LayerType::Atan2);
        assert_eq!(
            a2.inputs,
            vec![
                "layer0_trace_1_out".to_string(),
                "layer0_trace_0_out".to_string()
            ]
        );
        // NOT just `assert_builds`: until 2026-08-26 ny-build had no
        // `LayerType::Atan2` arm, so the builder swallowed the resulting
        // UnsupportedOp into a conservative `OpaqueSkip` and this node
        // silently propagated [-inf, +inf]. `assert_builds` passes on that
        // substitution — assert the BUILT layer is really Atan2, and that its
        // bounds are the quadrant-aware enclosure rather than the vacuous one.
        let network = model
            .build_graph_network(ny_build::GraphNetworkOptions::default())
            .expect("Atan2 builds");
        let bounds = ny_tensor::BoundedTensor::new(
            ndarray::arr1(&[1.0_f32, 1.0, 1.0, 1.0]).into_dyn(),
            ndarray::arr1(&[2.0_f32, 2.0, 2.0, 2.0]).into_dyn(),
        )
        .expect("input box");
        let out = network.propagate_ibp(&bounds).expect("Atan2 IBP");
        for (&lo, &hi) in out.lower().iter().zip(out.upper().iter()) {
            assert!(
                lo.is_finite() && hi.is_finite(),
                "Atan2 must produce a real enclosure, got [{lo}, {hi}] \
                 (an infinite box means the build fell back to OpaqueSkip)"
            );
            // y = ReLU(x) and x share the box [1, 2], so the angle is in the
            // first quadrant: atan2 in [atan(1/2), atan(2/1)].
            assert!(
                lo >= 0.0 && hi <= std::f32::consts::FRAC_PI_2,
                "first-quadrant angle expected, got [{lo}, {hi}]"
            );
        }
    }

    /// Cumsum([3], dim 0) → a single native CumSum spec: scalar constant axis
    /// (trailing-relative -1 for dim 0 of rank 1) at input[1] per the ONNX
    /// CumSum convention ny-build's `convert_cumsum` reads, with the
    /// inclusive-forward `exclusive`/`reverse` defaults pinned to 0.
    #[test]
    fn cumsum_lowers_to_native_cumsum_layer() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[3]),
            node(1, "c", TraceOp::Cumsum { dim: 0 }, &[0], &[3]),
        ]);
        let model = translate(&graph).expect("Cumsum translates");
        assert_eq!(count(&model, &LayerType::CumSum), 1);
        // No decomposition artifacts: only the input-identity Add remains.
        assert_eq!(count(&model, &LayerType::Slice), 0);
        assert_eq!(count(&model, &LayerType::Concat), 0);
        assert_eq!(count(&model, &LayerType::Add), 1);
        let cs = layer(&model, "layer0_trace_1");
        assert_eq!(
            cs.inputs,
            vec![
                "layer0_trace_0_out".to_string(),
                "layer0_trace_1_axis".to_string()
            ]
        );
        assert_eq!(
            cs.attributes.get("exclusive"),
            Some(&AttributeValue::Int(0))
        );
        assert_eq!(cs.attributes.get("reverse"), Some(&AttributeValue::Int(0)));
        let axis = model
            .weights
            .get("layer0_trace_1_axis")
            .expect("axis stored as scalar constant");
        assert_eq!(axis.iter().copied().collect::<Vec<_>>(), vec![-1.0]);
        assert_builds(&model, "Cumsum");
    }

    /// Kokoro-scale dims translate, build AND propagate: the native layer has
    /// no size cap, unlike the deleted O(N) decomposition that refused the
    /// T=24000 harmonic source (and anything past `MAX_DECOMPOSE_DIM`).
    #[test]
    fn cumsum_large_dims_translate_without_cap() {
        for n in [3000usize, 24000] {
            let graph = ComputationGraph::from_nodes(vec![
                node(0, "x", TraceOp::Input, &[], &[n]),
                node(1, "c", TraceOp::Cumsum { dim: 0 }, &[0], &[n]),
            ]);
            let model = translate(&graph).expect("large Cumsum translates natively");
            assert_eq!(count(&model, &LayerType::CumSum), 1, "n={n}");
            let network = model
                .build_graph_network(ny_build::GraphNetworkOptions::default())
                .expect("large Cumsum builds");

            // End-to-end IBP at scale: x in [0, 1]^n → the last cumsum upper
            // bound must cover n (exact monotone IBP with outward rounding).
            let bounds = ny_tensor::BoundedTensor::new(
                ndarray::ArrayD::from_elem(ndarray::IxDyn(&[n]), 0.0_f32),
                ndarray::ArrayD::from_elem(ndarray::IxDyn(&[n]), 1.0_f32),
            )
            .expect("valid input bounds");
            let out = network.propagate_ibp(&bounds).expect("large Cumsum IBP");
            // Lower bounds sit at 0 minus the layer's outward directed
            // rounding (each partial sum is next_down'ed), never far below.
            assert!(out.lower().iter().all(|&v| v >= -1e-3), "n={n}");
            let last = out
                .upper()
                .iter()
                .last()
                .copied()
                .expect("nonempty Cumsum output");
            assert!(last >= n as f32, "n={n}: upper end {last} must cover n");
            // Per-step outward rounding drifts ~1 ULP per accumulation
            // (~0.1% at n=24000) — tight, never vacuous.
            assert!(
                last <= n as f32 * 1.01,
                "n={n}: upper end {last} must stay tight, not vacuous"
            );
        }
    }

    /// Flip([3], dim 0) → 3 Slices in reverse order + Concat.
    #[test]
    fn flip_decomposes_to_reversed_slices() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[3]),
            node(1, "f", TraceOp::Flip { dim: 0 }, &[0], &[3]),
        ]);
        let model = translate(&graph).expect("Flip translates");
        assert_eq!(count(&model, &LayerType::Slice), 3);
        let cat = layer(&model, "layer0_trace_1");
        assert_eq!(cat.layer_type, LayerType::Concat);
        // Reverse order: s2, s1, s0.
        assert_eq!(
            cat.inputs,
            vec![
                "layer0_trace_1_s2_out".to_string(),
                "layer0_trace_1_s1_out".to_string(),
                "layer0_trace_1_s0_out".to_string()
            ]
        );
        let s2 = layer(&model, "layer0_trace_1_s2");
        assert_eq!(s2.attributes.get("start"), Some(&AttributeValue::Int(2)));
        assert_eq!(s2.attributes.get("end"), Some(&AttributeValue::Int(3)));
        assert_builds(&model, "Flip");
    }

    /// A 1-input RepeatInterleave has no counts tensor on the wire (nn's
    /// expected arity is 2): refused fail-closed.
    #[test]
    fn repeat_interleave_without_counts_input_refused() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[2]),
            node(1, "r", TraceOp::RepeatInterleave { dim: 0 }, &[0], &[4]),
        ]);
        let err = translate(&graph).expect_err("counts-less RepeatInterleave must fail closed");
        assert!(
            matches!(err, NyError::UnsupportedOp(ref message)
                if message.contains("RepeatInterleave")
                    && message.contains("counts tensor is missing")),
            "got {err:?}"
        );
    }

    /// Data-dependent (non-constant) counts imply a data-dependent output
    /// shape: refused fail-closed with the accurate reason.
    #[test]
    fn repeat_interleave_nonconstant_counts_refused() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[3]),
            node(1, "c", TraceOp::Relu, &[0], &[3]),
            node(2, "r", TraceOp::RepeatInterleave { dim: 0 }, &[0, 1], &[6]),
        ]);
        let err = translate(&graph).expect_err("variable counts must fail closed");
        assert!(
            matches!(err, NyError::UnsupportedOp(ref message)
                if message.contains("RepeatInterleave")
                    && message.contains("not a translate-time constant")),
            "got {err:?}"
        );
    }

    /// General constant counts lower to Slice-per-run + Concat, exactly:
    /// counts [2, 1, 3] on [a, b, c] must produce [a, a, b, c, c, c] with a
    /// slice per run (the count-1 middle element shares no slice with its
    /// repeated neighbors) and repeated concat references for c > 1.
    #[test]
    fn repeat_interleave_general_counts_lower_exactly() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[3]),
            node(
                1,
                "counts",
                TraceOp::ConstantWeight {
                    weight: WeightPayload::f32(vec![2.0, 1.0, 3.0], vec![3]),
                },
                &[],
                &[3],
            ),
            node(2, "r", TraceOp::RepeatInterleave { dim: 0 }, &[0, 1], &[6]),
        ]);
        let model = translate(&graph).expect("constant-counts RepeatInterleave translates");
        assert_eq!(count(&model, &LayerType::Slice), 3, "one slice per run");
        let cat = layer(&model, "layer0_trace_2");
        assert_eq!(cat.layer_type, LayerType::Concat);
        assert_eq!(
            cat.inputs,
            vec![
                "layer0_trace_2_s0_out".to_string(),
                "layer0_trace_2_s0_out".to_string(),
                "layer0_trace_2_s1_out".to_string(),
                "layer0_trace_2_s2_out".to_string(),
                "layer0_trace_2_s2_out".to_string(),
                "layer0_trace_2_s2_out".to_string(),
            ],
            "each source element repeats exactly its count"
        );
        let network = model
            .build_graph_network(ny_build::GraphNetworkOptions::default())
            .expect("RepeatInterleave builds");
        let bounds = ny_tensor::BoundedTensor::new(
            ndarray::arr1(&[1.0f32, 2.0, 3.0]).into_dyn(),
            ndarray::arr1(&[1.0f32, 2.0, 3.0]).into_dyn(),
        )
        .expect("bounds");
        let out = network
            .propagate_ibp(&bounds)
            .expect("RepeatInterleave IBP");
        let expected = [1.0f32, 1.0, 2.0, 3.0, 3.0, 3.0];
        assert_eq!(out.lower().as_slice().unwrap(), &expected, "exact lower");
        assert_eq!(out.upper().as_slice().unwrap(), &expected, "exact upper");
    }

    /// Zero counts drop their source elements: counts [0, 2, 0] keeps only
    /// the middle element, twice (a Concat referencing one slice twice).
    #[test]
    fn repeat_interleave_zero_counts_drop_elements() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[3]),
            node(
                1,
                "counts",
                TraceOp::ConstantWeight {
                    weight: WeightPayload::f32(vec![0.0, 2.0, 0.0], vec![3]),
                },
                &[],
                &[3],
            ),
            node(2, "r", TraceOp::RepeatInterleave { dim: 0 }, &[0, 1], &[2]),
        ]);
        let model = translate(&graph).expect("zero-count RepeatInterleave translates");
        assert_eq!(
            count(&model, &LayerType::Slice),
            1,
            "dropped runs emit nothing"
        );
        let network = model
            .build_graph_network(ny_build::GraphNetworkOptions::default())
            .expect("builds");
        let bounds = ny_tensor::BoundedTensor::new(
            ndarray::arr1(&[1.0f32, 2.0, 3.0]).into_dyn(),
            ndarray::arr1(&[1.0f32, 2.0, 3.0]).into_dyn(),
        )
        .expect("bounds");
        let out = network.propagate_ibp(&bounds).expect("IBP");
        assert_eq!(out.lower().as_slice().unwrap(), &[2.0, 2.0]);
        assert_eq!(out.upper().as_slice().unwrap(), &[2.0, 2.0]);
    }

    /// Uniform counts > 1 take the Reshape + Tile + Reshape expansion (the
    /// nearest-upsample machinery) — no O(N) slice decomposition — and stay
    /// exact, including on an interior (non-trailing) dim.
    #[test]
    fn repeat_interleave_uniform_counts_use_tile_expansion() {
        // Trailing dim of a rank-1 tensor.
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[3]),
            node(
                1,
                "counts",
                TraceOp::ConstantWeight {
                    weight: WeightPayload::f32(vec![2.0, 2.0, 2.0], vec![3]),
                },
                &[],
                &[3],
            ),
            node(2, "r", TraceOp::RepeatInterleave { dim: 0 }, &[0, 1], &[6]),
        ]);
        let model = translate(&graph).expect("uniform RepeatInterleave translates");
        assert_eq!(count(&model, &LayerType::Tile), 1);
        assert_eq!(count(&model, &LayerType::Slice), 0);
        assert_eq!(count(&model, &LayerType::Reshape), 2);
        let network = model
            .build_graph_network(ny_build::GraphNetworkOptions::default())
            .expect("builds");
        let bounds = ny_tensor::BoundedTensor::new(
            ndarray::arr1(&[1.0f32, 2.0, 3.0]).into_dyn(),
            ndarray::arr1(&[1.0f32, 2.0, 3.0]).into_dyn(),
        )
        .expect("bounds");
        let out = network.propagate_ibp(&bounds).expect("IBP");
        let expected = [1.0f32, 1.0, 2.0, 2.0, 3.0, 3.0];
        assert_eq!(out.lower().as_slice().unwrap(), &expected);
        assert_eq!(out.upper().as_slice().unwrap(), &expected);

        // Interior dim 0 of a rank-2 tensor: rows repeat interleaved.
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[2, 2]),
            node(
                1,
                "counts",
                TraceOp::ConstantWeight {
                    weight: WeightPayload::f32(vec![3.0, 3.0], vec![2]),
                },
                &[],
                &[2],
            ),
            node(
                2,
                "r",
                TraceOp::RepeatInterleave { dim: 0 },
                &[0, 1],
                &[6, 2],
            ),
        ]);
        let model = translate(&graph).expect("interior-dim uniform translates");
        assert_eq!(count(&model, &LayerType::Tile), 1);
        let tile = layer(&model, "layer0_trace_2_tile");
        assert_eq!(tile.attributes.get("axis"), Some(&AttributeValue::Int(1)));
        assert_eq!(tile.attributes.get("reps"), Some(&AttributeValue::Int(3)));
        let network = model
            .build_graph_network(ny_build::GraphNetworkOptions::default())
            .expect("builds");
        let bounds = ny_tensor::BoundedTensor::new(
            ndarray::arr2(&[[1.0f32, 2.0], [3.0, 4.0]]).into_dyn(),
            ndarray::arr2(&[[1.0f32, 2.0], [3.0, 4.0]]).into_dyn(),
        )
        .expect("bounds");
        let out = network.propagate_ibp(&bounds).expect("IBP");
        let expected = ndarray::arr2(&[
            [1.0f32, 2.0],
            [1.0, 2.0],
            [1.0, 2.0],
            [3.0, 4.0],
            [3.0, 4.0],
            [3.0, 4.0],
        ])
        .into_dyn();
        assert_eq!(out.lower(), &expected);
        assert_eq!(out.upper(), &expected);
    }

    /// All-ones counts are the identity (Add + 0), like Powf(1).
    #[test]
    fn repeat_interleave_all_ones_is_identity() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[3]),
            node(
                1,
                "counts",
                TraceOp::ConstantWeight {
                    weight: WeightPayload::f32(vec![1.0, 1.0, 1.0], vec![3]),
                },
                &[],
                &[3],
            ),
            node(2, "r", TraceOp::RepeatInterleave { dim: 0 }, &[0, 1], &[3]),
        ]);
        let model = translate(&graph).expect("all-ones RepeatInterleave translates");
        assert_eq!(count(&model, &LayerType::Slice), 0);
        assert_eq!(count(&model, &LayerType::Concat), 0);
        assert_eq!(count(&model, &LayerType::Tile), 0);
        // Input identity Add + the RepeatInterleave identity Add.
        assert_eq!(count(&model, &LayerType::Add), 2);
        assert_builds(&model, "RepeatInterleave identity");
    }

    /// Invalid serialized counts fail closed: non-integral, negative, wrong
    /// length, a sum contradicting the declared output shape, an all-zero
    /// expansion, and a fan-in past the decomposition cap.
    #[test]
    fn repeat_interleave_invalid_counts_fail_closed() {
        let build = |counts: Vec<f32>, counts_shape: Vec<usize>, out_dim: usize| {
            let counts_len = counts_shape.iter().product::<usize>();
            assert_eq!(counts.len(), counts_len, "fixture consistency");
            ComputationGraph::from_nodes(vec![
                node(0, "x", TraceOp::Input, &[], &[3]),
                node(
                    1,
                    "counts",
                    TraceOp::ConstantWeight {
                        weight: WeightPayload::f32(counts, counts_shape),
                    },
                    &[],
                    &[counts_len],
                ),
                node(
                    2,
                    "r",
                    TraceOp::RepeatInterleave { dim: 0 },
                    &[0, 1],
                    &[out_dim],
                ),
            ])
        };

        let err = translate(&build(vec![1.5, 1.0, 1.0], vec![3], 4))
            .expect_err("non-integral count must fail closed");
        assert!(
            matches!(err, NyError::ModelLoad(ref m) if m.contains("non-negative integer")),
            "got {err:?}"
        );

        let err = translate(&build(vec![-1.0, 2.0, 3.0], vec![3], 4))
            .expect_err("negative count must fail closed");
        assert!(
            matches!(err, NyError::ModelLoad(ref m) if m.contains("non-negative integer")),
            "got {err:?}"
        );

        let err = translate(&build(vec![2.0, 2.0], vec![2], 4))
            .expect_err("length mismatch must fail closed");
        assert!(
            matches!(err, NyError::ModelLoad(ref m) if m.contains("counts length")),
            "got {err:?}"
        );

        let err = translate(&build(vec![2.0, 1.0, 3.0], vec![3], 5))
            .expect_err("sum mismatch must fail closed");
        assert!(
            matches!(err, NyError::ModelLoad(ref m) if m.contains("counts sum to 6")),
            "got {err:?}"
        );

        let err = translate(&build(vec![0.0, 0.0, 0.0], vec![3], 0))
            .expect_err("all-zero counts must fail closed");
        assert!(
            matches!(err, NyError::UnsupportedOp(ref m) if m.contains("all counts are 0")),
            "got {err:?}"
        );

        // Fan-in cap: a single element expanded past MAX_DECOMPOSE_DIM concat
        // inputs (non-uniform path forced by the second, zero-count element).
        let oversized = super::MAX_DECOMPOSE_DIM + 1;
        let err = translate(&build(vec![oversized as f32, 0.0, 0.0], vec![3], oversized))
            .expect_err("oversized fan-in must fail closed");
        assert!(
            matches!(err, NyError::UnsupportedOp(ref m) if m.contains("decomposition limit")),
            "got {err:?}"
        );
    }

    /// Arange is emitted as a constant weight (no layer), like NN #2271.
    #[test]
    fn arange_becomes_constant_weight() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[3]),
            node(
                1,
                "ar",
                TraceOp::Arange {
                    start: 0.0,
                    end: 3.0,
                    step: 1.0,
                },
                &[],
                &[3],
            ),
            node(2, "sum", TraceOp::Add, &[0, 1], &[3]),
        ]);
        let model = translate(&graph).expect("Arange translates");
        let w = model
            .weights
            .get("layer0_trace_1_out")
            .expect("arange stored as weight");
        assert_eq!(w.as_slice().unwrap(), &[0.0, 1.0, 2.0]);
        // Only the input-identity Add and the consuming Add — no Arange layer.
        assert_eq!(count(&model, &LayerType::Add), 2);
        assert_builds(&model, "Arange");
    }

    /// Arange with step=0 is refused (undefined range).
    #[test]
    fn arange_zero_step_refused() {
        let graph = ComputationGraph::from_nodes(vec![node(
            0,
            "ar",
            TraceOp::Arange {
                start: 0.0,
                end: 3.0,
                step: 0.0,
            },
            &[],
            &[3],
        )]);
        let err = translate(&graph).expect_err("step=0 refused");
        assert!(
            matches!(err, NyError::UnsupportedOp(ref m) if m.contains("Arange with step=0")),
            "got {err:?}"
        );
    }

    #[test]
    fn arange_rejects_non_finite_or_inconsistent_parameters() {
        for (start, end, step, expected) in [
            (f64::NAN, 3.0, 1.0, "must be finite"),
            (0.0, f64::INFINITY, 1.0, "must be finite"),
            (0.0, 3.0, f64::NEG_INFINITY, "must be finite"),
            (0.0, 3.0, -1.0, "points away"),
            (3.0, 0.0, 1.0, "points away"),
        ] {
            let graph = ComputationGraph::from_nodes(vec![node(
                0,
                "ar",
                TraceOp::Arange { start, end, step },
                &[],
                &[3],
            )]);
            let err = translate(&graph).expect_err("invalid Arange must fail closed");
            assert!(
                matches!(err, NyError::ModelLoad(ref message)
                    if message.contains("Arange") && message.contains(expected)),
                "{start:?}, {end:?}, {step:?}: got {err:?}"
            );
        }
    }

    #[test]
    fn arange_validates_declared_shape_and_resource_cap() {
        let shape_mismatch = ComputationGraph::from_nodes(vec![node(
            0,
            "ar",
            TraceOp::Arange {
                start: 0.0,
                end: 3.0,
                step: 1.0,
            },
            &[],
            &[2],
        )]);
        let err = translate(&shape_mismatch).expect_err("shape mismatch must fail closed");
        assert!(
            matches!(err, NyError::ModelLoad(ref message)
                if message.contains("declared output shape")
                    && message.contains("parameters produce 3")),
            "got {err:?}"
        );

        let oversized = super::MAX_ARANGE_ELEMENTS + 1;
        let over_cap = ComputationGraph::from_nodes(vec![node(
            0,
            "ar",
            TraceOp::Arange {
                start: 0.0,
                end: oversized as f64,
                step: 1.0,
            },
            &[],
            &[oversized],
        )]);
        let err = translate(&over_cap).expect_err("oversized Arange must fail before allocation");
        assert!(
            matches!(err, NyError::ModelLoad(ref message)
                if message.contains("exceeding")
                    && message.contains("element limit")),
            "got {err:?}"
        );
    }

    #[test]
    fn arange_negative_step_translates_exactly() {
        let graph = ComputationGraph::from_nodes(vec![node(
            0,
            "ar",
            TraceOp::Arange {
                start: 3.0,
                end: 0.0,
                step: -1.0,
            },
            &[],
            &[3],
        )]);
        let model = translate(&graph).expect("negative-step Arange translates");
        assert_eq!(
            model
                .weights
                .get("layer0_trace_0_out")
                .expect("range weight")
                .as_slice()
                .expect("contiguous"),
            &[3.0, 2.0, 1.0]
        );
    }

    /// Triu / Tril map to the triangular-mask layers with the diagonal attr.
    #[test]
    fn triu_tril_carry_diagonal() {
        for (op, lt, diag) in [
            (TraceOp::Triu { diagonal: 1 }, LayerType::Triu, 1),
            (TraceOp::Tril { diagonal: -1 }, LayerType::Tril, -1),
        ] {
            let graph = ComputationGraph::from_nodes(vec![
                node(0, "x", TraceOp::Input, &[], &[3, 3]),
                node(1, "t", op, &[0], &[3, 3]),
            ]);
            let model = translate(&graph).expect("triangular translates");
            let tri = layer(&model, "layer0_trace_1");
            assert_eq!(tri.layer_type, lt);
            assert_eq!(
                tri.attributes.get("diagonal"),
                Some(&AttributeValue::Int(diag))
            );
            assert_builds(&model, "Triu/Tril");
        }
    }

    /// SliceSet(self=[4], src=[2], dim 0, start 1) → before/after Slices +
    /// Concat(before, src, after) on ONNX axis 1.
    #[test]
    fn slice_set_decomposes_to_slices_concat() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "self", TraceOp::Input, &[], &[4]),
            node(
                1,
                "src",
                TraceOp::ConstantWeight {
                    weight: WeightPayload::f32(vec![9.0, 9.0], vec![2]),
                },
                &[],
                &[2],
            ),
            node(
                2,
                "ss",
                TraceOp::SliceSet { dim: 0, start: 1 },
                &[0, 1],
                &[4],
            ),
        ]);
        let model = translate(&graph).expect("SliceSet translates");
        let before = layer(&model, "layer0_trace_2_before");
        assert_eq!(
            before.attributes.get("start"),
            Some(&AttributeValue::Int(0))
        );
        assert_eq!(before.attributes.get("end"), Some(&AttributeValue::Int(1)));
        let after = layer(&model, "layer0_trace_2_after");
        assert_eq!(after.attributes.get("start"), Some(&AttributeValue::Int(3)));
        assert_eq!(after.attributes.get("end"), Some(&AttributeValue::Int(4)));
        let cat = layer(&model, "layer0_trace_2");
        assert_eq!(cat.layer_type, LayerType::Concat);
        assert_eq!(
            cat.inputs,
            vec![
                "layer0_trace_2_before_out".to_string(),
                "layer0_trace_1_out".to_string(),
                "layer0_trace_2_after_out".to_string()
            ]
        );
        assert_builds(&model, "SliceSet");
    }

    /// SliceSet with two variable inputs works through the multi-input entry
    /// (mirrors NN's test fixture shape).
    #[test]
    fn slice_set_multi_input_builds() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "self", TraceOp::Input, &[], &[2, 6]),
            node(1, "src", TraceOp::Input, &[], &[2, 2]),
            node(
                2,
                "ss",
                TraceOp::SliceSet { dim: 1, start: 2 },
                &[0, 1],
                &[2, 6],
            ),
        ]);
        let translation = translate_multi_input(&graph).expect("multi-input SliceSet translates");
        assert_builds(&translation.model, "SliceSet multi-input");
    }

    /// Unfold([4], dim 0, size 2, step 1) → 3 windows of Slice + Reshape,
    /// Concat, final Reshape to [3, 2].
    #[test]
    fn unfold_decomposes_to_windows() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[4]),
            node(
                1,
                "u",
                TraceOp::Unfold {
                    dim: 0,
                    size: 2,
                    step: 1,
                },
                &[0],
                &[3, 2],
            ),
        ]);
        let model = translate(&graph).expect("Unfold translates");
        assert_eq!(count(&model, &LayerType::Slice), 3, "one Slice per window");
        // No permute needed for the last axis: 3 window Reshapes + final.
        assert_eq!(count(&model, &LayerType::Transpose), 0);
        assert_eq!(count(&model, &LayerType::Reshape), 4);
        let w1 = layer(&model, "layer0_trace_1_w1_sl");
        assert_eq!(w1.attributes.get("start"), Some(&AttributeValue::Int(1)));
        assert_eq!(w1.attributes.get("end"), Some(&AttributeValue::Int(3)));
        let cat = layer(&model, "layer0_trace_1_cat");
        // Trailing-relative vs the rank-2 window tensors: 0 - (1 + 1) = -2.
        assert_eq!(cat.attributes.get("axis"), Some(&AttributeValue::Int(-2)));
        let final_rs = layer(&model, "layer0_trace_1");
        assert_eq!(
            final_rs.attributes.get("shape"),
            Some(&AttributeValue::Ints(vec![3, 2]))
        );
        assert_builds(&model, "Unfold");
    }

    /// IndexSelect / Gather map to LayerType::Gather with a trailing-relative
    /// axis vs the data tensor's rank.
    #[test]
    fn gather_and_index_select_map_to_gather_layer() {
        for op in [TraceOp::IndexSelect { dim: 0 }, TraceOp::Gather { dim: 0 }] {
            let graph = ComputationGraph::from_nodes(vec![
                node(0, "x", TraceOp::Input, &[], &[4]),
                node(
                    1,
                    "idx",
                    TraceOp::ConstantWeight {
                        weight: WeightPayload::f32(vec![0.0, 2.0], vec![2]),
                    },
                    &[],
                    &[2],
                ),
                node(2, "g", op, &[0, 1], &[2]),
            ]);
            let model = translate(&graph).expect("gather translates");
            let g = layer(&model, "layer0_trace_2");
            assert_eq!(g.layer_type, LayerType::Gather);
            // Trailing-relative: trace dim 0 of the rank-1 data → -1.
            assert_eq!(g.attributes.get("axis"), Some(&AttributeValue::Int(-1)));
            assert_eq!(
                g.inputs,
                vec![
                    "layer0_trace_0_out".to_string(),
                    "layer0_trace_1_out".to_string()
                ]
            );
            assert_builds(&model, "Gather/IndexSelect");
        }
    }

    /// A 1-input Gather/IndexSelect has no index tensor: refused fail-closed
    /// (NN defers to a graph-build failure; translation must not succeed).
    #[test]
    fn gather_without_indices_refused() {
        for (op, opname) in [
            (TraceOp::IndexSelect { dim: 0 }, "IndexSelect"),
            (TraceOp::Gather { dim: 0 }, "Gather"),
        ] {
            let graph = ComputationGraph::from_nodes(vec![
                node(0, "x", TraceOp::Input, &[], &[4]),
                node(1, "g", op, &[0], &[4]),
            ]);
            let err = translate(&graph).expect_err("1-input gather refused");
            assert!(
                matches!(err, NyError::UnsupportedOp(ref m) if m.contains(opname)),
                "error must name {opname}, got {err:?}"
            );
        }
    }

    /// The deliberately-refused misc ops keep the exact fail-closed
    /// UnsupportedOp refusal (coverage-taxonomy Unsupported — NN's identity
    /// passthrough is not a sound over-approximation; see module docs).
    ///
    /// `ScatterAdd` / `IndexAdd` LEFT this list: their accumulation is
    /// order-independent and their dynamic-index rule never reads the index, so
    /// both now lower to native layers (see
    /// `additive_scatter_ops_emit_native_layers_with_trailing_axis`). `Scatter`
    /// was never in it — its refusal is conditional on a runtime index and
    /// carries its own precise message, asserted in
    /// `scatter_overwrite_with_a_runtime_index_is_refused_for_the_duplicate_reason`.
    /// `IndexPut` left it for the same reason `Scatter` never joined it: its
    /// constant-index case is now lowered, and only the runtime-index case is
    /// refused — see
    /// `index_put_with_a_runtime_index_is_refused_for_the_duplicate_reason`.
    #[test]
    fn refused_misc_ops_stay_refused() {
        let cases: Vec<(TraceOp, &str)> = vec![(
            TraceOp::MoeGating {
                num_experts: 4,
                top_k: 2,
            },
            "MoeGating",
        )];
        for (op, opname) in cases {
            let graph = ComputationGraph::from_nodes(vec![
                node(0, "x", TraceOp::Input, &[], &[4]),
                node(1, "r", op, &[0], &[4]),
            ]);
            let err = translate(&graph).expect_err("refused op must error");
            match err {
                NyError::UnsupportedOp(m) => assert_eq!(
                    m,
                    format!("{opname} not supported in NY trace translation"),
                    "refusal message must keep the catch-all shape"
                ),
                other => panic!("expected UnsupportedOp for {opname}, got {other:?}"),
            }
        }
    }
    /// Compare lowers to a 2-input Compare layer: data + scalar-threshold
    /// constant, with the `compare_op` attribute (NN #2271 parity).
    #[test]
    fn compare_scalar_lowers_to_compare_layer() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[4]),
            node(
                1,
                "c",
                TraceOp::Compare {
                    op: CompareOp::Gt,
                    value: 0.5,
                },
                &[0],
                &[4],
            ),
        ]);
        let model = translate(&graph).expect("Compare translates");
        let cmp = layer(&model, "layer0_trace_1");
        assert_eq!(cmp.layer_type, LayerType::Compare);
        assert_eq!(
            cmp.inputs,
            vec![
                "layer0_trace_0_out".to_string(),
                "layer0_trace_1_threshold".to_string()
            ]
        );
        assert_eq!(
            cmp.attributes.get("compare_op"),
            Some(&AttributeValue::String("Gt".to_string()))
        );
        assert_builds(&model, "Compare");
    }

    /// CompareTensor lowers to a 2-activation-input CompareTensor layer with
    /// the `compare_op` attribute (NN #2271 parity).
    #[test]
    fn compare_tensor_lowers_to_compare_tensor_layer() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[4]),
            node(1, "y", TraceOp::Relu, &[0], &[4]),
            node(
                2,
                "c",
                TraceOp::CompareTensor { op: CompareOp::Le },
                &[0, 1],
                &[4],
            ),
        ]);
        let model = translate(&graph).expect("CompareTensor translates");
        let cmp = layer(&model, "layer0_trace_2");
        assert_eq!(cmp.layer_type, LayerType::CompareTensor);
        assert_eq!(
            cmp.attributes.get("compare_op"),
            Some(&AttributeValue::String("Le".to_string()))
        );
        assert_builds(&model, "CompareTensor");
    }

    // -- WhereCond ---------------------------------------------------------

    /// WhereCond lowers to a single 3-input Where spec in nn's traced order
    /// (condition, on_true, on_false), and graph IBP produces the union hull
    /// of the two branches — the sound enclosure for any realizable mask.
    #[test]
    fn where_cond_lowers_to_ternary_where_and_unions_branches() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[2]),
            node(1, "t", TraceOp::Relu, &[0], &[2]),
            node(2, "f", TraceOp::Neg, &[0], &[2]),
            node(
                3,
                "m",
                TraceOp::Compare {
                    op: CompareOp::Gt,
                    value: 0.0,
                },
                &[0],
                &[2],
            ),
            node(4, "w", TraceOp::WhereCond, &[3, 1, 2], &[2]),
        ]);
        let model = translate(&graph).expect("WhereCond translates");
        let w = layer(&model, "layer0_trace_4");
        assert_eq!(w.layer_type, LayerType::Where);
        assert_eq!(
            w.inputs,
            vec![
                "layer0_trace_3_out".to_string(),
                "layer0_trace_1_out".to_string(),
                "layer0_trace_2_out".to_string()
            ],
            "condition, on_true, on_false — nn's traced input order"
        );
        let network = model
            .build_graph_network(ny_build::GraphNetworkOptions::default())
            .expect("WhereCond builds");
        // x in [1, 2]: on_true = relu(x) = [1, 2], on_false = -x = [-2, -1].
        // Union hull per element: [-2, 2] — exact min/max selection.
        let bounds = ny_tensor::BoundedTensor::new(
            ndarray::arr1(&[1.0f32, 1.0]).into_dyn(),
            ndarray::arr1(&[2.0f32, 2.0]).into_dyn(),
        )
        .expect("bounds");
        let out = network.propagate_ibp(&bounds).expect("WhereCond IBP");
        assert_eq!(out.lower().as_slice().unwrap(), &[-2.0, -2.0]);
        assert_eq!(out.upper().as_slice().unwrap(), &[2.0, 2.0]);
    }

    /// The masked-fill pattern (uniform constant on_true branch, mask from
    /// Compare — exactly what nn's `masked_fill` traces) lowers via the
    /// Clip[v, v] activation carrier: the graph builder wires Where either as
    /// 3 activations or with BOTH branches embedded, so a single constant
    /// branch needs the carrier to survive activation wiring. Bounds stay the
    /// sound union hull of the fill constant and the passthrough values.
    #[test]
    fn where_cond_masked_fill_uses_clip_carrier() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[2]),
            node(
                1,
                "fill",
                TraceOp::ConstantWeight {
                    weight: WeightPayload::f32(vec![9.0, 9.0], vec![2]),
                },
                &[],
                &[2],
            ),
            node(
                2,
                "m",
                TraceOp::Compare {
                    op: CompareOp::Gt,
                    value: 0.5,
                },
                &[0],
                &[2],
            ),
            node(3, "w", TraceOp::WhereCond, &[2, 1, 0], &[2]),
        ]);
        let model = translate(&graph).expect("masked-fill WhereCond translates");
        let carrier = layer(&model, "layer0_trace_3_on_true_carrier");
        assert_eq!(carrier.layer_type, LayerType::Clip);
        assert_eq!(
            carrier.attributes.get("min"),
            Some(&AttributeValue::Float(9.0))
        );
        assert_eq!(
            carrier.attributes.get("max"),
            Some(&AttributeValue::Float(9.0))
        );
        assert_eq!(
            carrier.inputs,
            vec!["layer0_trace_0_out".to_string()],
            "the variable branch anchors the carrier's shape"
        );
        let w = layer(&model, "layer0_trace_3");
        assert_eq!(w.layer_type, LayerType::Where);
        assert_eq!(
            w.inputs,
            vec![
                "layer0_trace_2_out".to_string(),
                "layer0_trace_3_on_true_carrier_out".to_string(),
                "layer0_trace_0_out".to_string()
            ]
        );
        let network = model
            .build_graph_network(ny_build::GraphNetworkOptions::default())
            .expect("masked-fill builds");
        // x in [0, 1] straddles the threshold: union of {9} and [0, 1].
        let bounds = ny_tensor::BoundedTensor::new(
            ndarray::arr1(&[0.0f32, 0.0]).into_dyn(),
            ndarray::arr1(&[1.0f32, 1.0]).into_dyn(),
        )
        .expect("bounds");
        let out = network.propagate_ibp(&bounds).expect("masked-fill IBP");
        assert_eq!(out.lower().as_slice().unwrap(), &[0.0, 0.0]);
        assert_eq!(out.upper().as_slice().unwrap(), &[9.0, 9.0]);
    }

    /// Both branches constant: the spec is emitted verbatim (no carriers) and
    /// the graph builder embeds the constants into the WhereLayer, wiring the
    /// condition as the sole activation. Bounds must enclose the true select
    /// and never exceed the branch hull.
    #[test]
    fn where_cond_both_constant_branches_use_embedded_path() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[2]),
            node(
                1,
                "t",
                TraceOp::ConstantWeight {
                    weight: WeightPayload::f32(vec![9.0, 9.0], vec![2]),
                },
                &[],
                &[2],
            ),
            node(
                2,
                "f",
                TraceOp::ConstantWeight {
                    weight: WeightPayload::f32(vec![-9.0, -9.0], vec![2]),
                },
                &[],
                &[2],
            ),
            node(
                3,
                "m",
                TraceOp::Compare {
                    op: CompareOp::Gt,
                    value: 0.5,
                },
                &[0],
                &[2],
            ),
            node(4, "w", TraceOp::WhereCond, &[3, 1, 2], &[2]),
        ]);
        let model = translate(&graph).expect("both-constant WhereCond translates");
        assert_eq!(count(&model, &LayerType::Clip), 0, "no carriers");
        let network = model
            .build_graph_network(ny_build::GraphNetworkOptions::default())
            .expect("embedded-constant Where builds");
        // x in [1, 2]: the mask is determined true, so the realized output is
        // 9 everywhere. Sound bounds contain 9 and stay within the branch
        // hull [-9, 9] (the embedded path may or may not tighten to the
        // determined select; both are sound).
        let bounds = ny_tensor::BoundedTensor::new(
            ndarray::arr1(&[1.0f32, 1.0]).into_dyn(),
            ndarray::arr1(&[2.0f32, 2.0]).into_dyn(),
        )
        .expect("bounds");
        let out = network.propagate_ibp(&bounds).expect("embedded Where IBP");
        for i in 0..2 {
            let lo = out.lower()[[i]];
            let hi = out.upper()[[i]];
            assert!(
                lo <= 9.0 && hi >= 9.0,
                "must enclose the select: [{lo}, {hi}]"
            );
            assert!(
                lo >= -9.0 && hi <= 9.0,
                "must stay in the hull: [{lo}, {hi}]"
            );
        }
    }

    /// A NON-uniform single constant branch has no exact activation carrier
    /// and the builder embeds constants only pairwise: refused fail-closed at
    /// translate time instead of failing later at graph build.
    #[test]
    fn where_cond_non_uniform_single_constant_refused() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[2]),
            node(
                1,
                "fill",
                TraceOp::ConstantWeight {
                    weight: WeightPayload::f32(vec![1.0, 2.0], vec![2]),
                },
                &[],
                &[2],
            ),
            node(
                2,
                "m",
                TraceOp::Compare {
                    op: CompareOp::Gt,
                    value: 0.5,
                },
                &[0],
                &[2],
            ),
            node(3, "w", TraceOp::WhereCond, &[2, 1, 0], &[2]),
        ]);
        let err = translate(&graph).expect_err("non-uniform single constant must fail closed");
        assert!(
            matches!(err, NyError::UnsupportedOp(ref m)
                if m.contains("WhereCond")
                    && m.contains("uniform")
                    && m.contains("on_true")),
            "got {err:?}"
        );
    }

    /// WhereCond with fewer than 3 inputs cannot select: refused fail-closed.
    #[test]
    fn where_cond_requires_three_inputs() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[2]),
            node(1, "t", TraceOp::Relu, &[0], &[2]),
            node(2, "w", TraceOp::WhereCond, &[0, 1], &[2]),
        ]);
        let err = translate(&graph).expect_err("2-input WhereCond refused");
        assert!(
            matches!(err, NyError::UnsupportedOp(ref m)
                if m.contains("WhereCond requires 3 inputs")),
            "got {err:?}"
        );
    }

    // -- Roll --------------------------------------------------------------

    /// Roll([4], shifts [1], dims [0]) → tail Slice [3,4), head Slice [0,3),
    /// then Concat — matching nn's example
    /// `[1,2,3,4].roll(&[1],&[0]) = [4,1,2,3]` exactly under IBP (pure
    /// reindexing).
    #[test]
    fn roll_lowers_to_rotation_and_is_exact() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[4]),
            node(
                1,
                "r",
                TraceOp::Roll {
                    shifts: vec![1],
                    dims: vec![0],
                },
                &[0],
                &[4],
            ),
        ]);
        let model = translate(&graph).expect("Roll translates");
        assert_eq!(count(&model, &LayerType::Slice), 2, "tail + head");
        let tail = layer(&model, "layer0_trace_1_r0_tail");
        assert_eq!(tail.attributes.get("start"), Some(&AttributeValue::Int(3)));
        assert_eq!(tail.attributes.get("end"), Some(&AttributeValue::Int(4)));
        let head = layer(&model, "layer0_trace_1_r0_head");
        assert_eq!(head.attributes.get("start"), Some(&AttributeValue::Int(0)));
        assert_eq!(head.attributes.get("end"), Some(&AttributeValue::Int(3)));
        let cat = layer(&model, "layer0_trace_1");
        assert_eq!(cat.layer_type, LayerType::Concat);
        assert_eq!(
            cat.inputs,
            vec![
                "layer0_trace_1_r0_tail_out".to_string(),
                "layer0_trace_1_r0_head_out".to_string()
            ]
        );
        let network = model
            .build_graph_network(ny_build::GraphNetworkOptions::default())
            .expect("Roll builds");
        let bounds = ny_tensor::BoundedTensor::new(
            ndarray::arr1(&[1.0f32, 2.0, 3.0, 4.0]).into_dyn(),
            ndarray::arr1(&[1.0f32, 2.0, 3.0, 4.0]).into_dyn(),
        )
        .expect("bounds");
        let out = network.propagate_ibp(&bounds).expect("Roll IBP");
        let expected = [4.0f32, 1.0, 2.0, 3.0];
        assert_eq!(out.lower().as_slice().unwrap(), &expected);
        assert_eq!(out.upper().as_slice().unwrap(), &expected);
    }

    /// Negative shifts normalize mod the dim size and full-cycle shifts are
    /// skipped, mirroring nn's sequential per-pair evaluation: on [2, 3],
    /// shifts [-1, 2] over dims [1, 0] leaves dim 0 untouched (2 ≡ 0 mod 2)
    /// and rotates dim 1 by -1 → each row [a, b, c] becomes [b, c, a].
    #[test]
    fn roll_normalizes_negative_and_full_cycle_shifts() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[2, 3]),
            node(
                1,
                "r",
                TraceOp::Roll {
                    shifts: vec![-1, 2],
                    dims: vec![1, 0],
                },
                &[0],
                &[2, 3],
            ),
        ]);
        let model = translate(&graph).expect("Roll translates");
        assert_eq!(
            count(&model, &LayerType::Slice),
            2,
            "only the effective rotation emits slices"
        );
        assert_eq!(count(&model, &LayerType::Concat), 1);
        let network = model
            .build_graph_network(ny_build::GraphNetworkOptions::default())
            .expect("Roll builds");
        let input = ndarray::arr2(&[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]).into_dyn();
        let bounds = ny_tensor::BoundedTensor::new(input.clone(), input).expect("bounds");
        let out = network.propagate_ibp(&bounds).expect("Roll IBP");
        let expected = ndarray::arr2(&[[2.0f32, 3.0, 1.0], [5.0, 6.0, 4.0]]).into_dyn();
        assert_eq!(out.lower(), &expected);
        assert_eq!(out.upper(), &expected);
    }

    /// Two effective rotations chain sequentially (nn applies pairs in trace
    /// order): [2, 3] rolled by 1 on dim 0 then 1 on dim 1.
    #[test]
    fn roll_chains_multiple_effective_rotations() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[2, 3]),
            node(
                1,
                "r",
                TraceOp::Roll {
                    shifts: vec![1, 1],
                    dims: vec![0, 1],
                },
                &[0],
                &[2, 3],
            ),
        ]);
        let model = translate(&graph).expect("chained Roll translates");
        assert_eq!(count(&model, &LayerType::Slice), 4, "2 slices per rotation");
        assert_eq!(count(&model, &LayerType::Concat), 2);
        let network = model
            .build_graph_network(ny_build::GraphNetworkOptions::default())
            .expect("chained Roll builds");
        let input = ndarray::arr2(&[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]).into_dyn();
        let bounds = ny_tensor::BoundedTensor::new(input.clone(), input).expect("bounds");
        let out = network.propagate_ibp(&bounds).expect("chained Roll IBP");
        // dim 0 by 1: [[4,5,6],[1,2,3]]; then dim 1 by 1: [[6,4,5],[3,1,2]].
        let expected = ndarray::arr2(&[[6.0f32, 4.0, 5.0], [3.0, 1.0, 2.0]]).into_dyn();
        assert_eq!(out.lower(), &expected);
        assert_eq!(out.upper(), &expected);
    }

    /// All shifts vanishing mod their dim sizes (and the empty pair list)
    /// collapse to the Add + 0 identity.
    #[test]
    fn roll_zero_effective_shift_is_identity() {
        for (shifts, dims) in [(vec![0i64], vec![0usize]), (vec![], vec![])] {
            let graph = ComputationGraph::from_nodes(vec![
                node(0, "x", TraceOp::Input, &[], &[4]),
                node(1, "r", TraceOp::Roll { shifts, dims }, &[0], &[4]),
            ]);
            let model = translate(&graph).expect("identity Roll translates");
            assert_eq!(count(&model, &LayerType::Slice), 0);
            assert_eq!(count(&model, &LayerType::Concat), 0);
            // Input identity Add + the Roll identity Add.
            assert_eq!(count(&model, &LayerType::Add), 2);
            assert_builds(&model, "Roll identity");
        }
    }

    /// A shifts/dims length mismatch cannot come from a real nn trace (nn
    /// errors before recording): refused fail-closed.
    #[test]
    fn roll_mismatched_shifts_dims_refused() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[4]),
            node(
                1,
                "r",
                TraceOp::Roll {
                    shifts: vec![1, 2],
                    dims: vec![0],
                },
                &[0],
                &[4],
            ),
        ]);
        let err = translate(&graph).expect_err("mismatched Roll refused");
        assert!(
            matches!(err, NyError::UnsupportedOp(ref m) if m.contains("shifts length")),
            "got {err:?}"
        );
    }

    // -- Selection family (direct translate_misc calls) --------------------
    //
    // The mod.rs dispatch still routes Topk/Argmax/Argmin/ArgSort/Sort to its
    // explicit refusal arm (their U32 index outputs also need a dtype-gate
    // carve-out there), so these arms are exercised by calling translate_misc
    // directly and assembling the GraphModel the way translate_impl does.

    /// Build a GraphModel for one misc-family node fed by a single variable
    /// input, bypassing the mod.rs dispatch.
    fn translate_direct(
        op: TraceOp,
        input_shape: &[usize],
        output_shape: &[usize],
    ) -> Result<GraphModel, NyError> {
        use ny_build::{DataType, GraphModelBuilder};
        use std::collections::HashMap;

        let mut ctx = super::super::Ctx::new(false);
        let mut layers: Vec<LayerSpec> = Vec::new();

        // Input node plumbing (mirrors translate_impl).
        let in_out = "layer0_trace_0_out";
        layers.extend(
            super::ops_core::translate_input("layer0_trace_0", in_out, &mut ctx)
                .expect("input identity translates")
                .specs,
        );
        let in_shape_i64: Vec<i64> = input_shape.iter().map(|&d| d as i64).collect();
        ctx.tensor_shapes
            .insert(in_out.to_string(), in_shape_i64.clone());

        let trace_node = node(1, "sel", op, &[0], output_shape);
        let out_tensor = "layer0_trace_1_out";
        let translated = super::translate_misc(
            &trace_node,
            "layer0_trace_1",
            &[in_out.to_string()],
            out_tensor,
            &HashMap::new(),
            &mut ctx,
        )?;
        let out_shape_i64: Vec<i64> = output_shape.iter().map(|&d| d as i64).collect();
        ctx.tensor_shapes
            .insert(out_tensor.to_string(), out_shape_i64.clone());
        layers.extend(translated.specs);

        let mut builder = GraphModelBuilder::new("misc_direct")
            .input("layer0_trace_0_in", &in_shape_i64, DataType::Float32)
            .output(out_tensor, &out_shape_i64, DataType::Float32);
        for spec in &layers {
            for out in &spec.outputs {
                if let Some(first) = spec.inputs.first() {
                    builder = builder.tensor_producer(out.clone(), first.clone());
                }
            }
        }
        for spec in layers {
            builder = builder.layer(spec);
        }
        for (wname, arr) in ctx.weights.iter() {
            builder = builder.weight(wname.to_string(), arr.clone());
        }
        for cname in &ctx.constant_tensors {
            builder = builder.constant_tensor(cname.clone());
        }
        for (tname, shape) in &ctx.tensor_shapes {
            builder = builder.tensor_shape(tname.clone(), shape);
        }
        Ok(builder.build())
    }

    /// Propagate IBP through a directly-assembled model.
    fn ibp_direct(
        model: &GraphModel,
        lower: ndarray::ArrayD<f32>,
        upper: ndarray::ArrayD<f32>,
    ) -> ny_tensor::BoundedTensor {
        let network = model
            .build_graph_network(ny_build::GraphNetworkOptions::default())
            .expect("direct model builds");
        let bounds = ny_tensor::BoundedTensor::new(lower, upper).expect("bounds");
        network.propagate_ibp(&bounds).expect("direct IBP")
    }

    /// Argmax lowers to LayerType::Argmax with a trailing-relative axis and
    /// the pinned keepdims=0 / select_last_index=0 attributes; IBP is exact
    /// when one element dominates its lane and the [0, n-1] hull otherwise.
    #[test]
    fn argmax_lowers_to_native_layer_with_pinned_attributes() {
        let model =
            translate_direct(TraceOp::Argmax { dim: 1 }, &[2, 4], &[2]).expect("Argmax translates");
        let spec = layer(&model, "layer0_trace_1");
        assert_eq!(spec.layer_type, LayerType::Argmax);
        // Trailing-relative: dim 1 of the rank-2 DATA tensor -> -1.
        assert_eq!(spec.attributes.get("axis"), Some(&AttributeValue::Int(-1)));
        // nn records the op before the keepdim unsqueeze: always reduced.
        assert_eq!(
            spec.attributes.get("keepdims"),
            Some(&AttributeValue::Int(0))
        );
        // nn's strict-compare first-index tie policy on every wired backend.
        assert_eq!(
            spec.attributes.get("select_last_index"),
            Some(&AttributeValue::Int(0))
        );

        // Lane 0: element 1 dominates ([8,9] vs [1,2],[3,4],[4,6]) -> exact 1.
        // Lane 1: overlap -> sound [0, 3] hull.
        let out = ibp_direct(
            &model,
            ndarray::arr2(&[[1.0f32, 8.0, 3.0, 4.0], [0.0, -2.0, 7.0, 6.0]]).into_dyn(),
            ndarray::arr2(&[[2.0f32, 9.0, 4.0, 6.0], [1.0, 8.5, 9.0, 8.0]]).into_dyn(),
        );
        assert_eq!(out.lower().as_slice().unwrap(), &[1.0, 0.0]);
        assert_eq!(out.upper().as_slice().unwrap(), &[1.0, 3.0]);
    }

    /// Argmin mirrors Argmax through LayerType::ArgMin, and exact ties pin
    /// nn's first-index policy (strict-compare loops on CPU and Metal; CUDA
    /// falls back to CPU): concrete [4, 4, 1] argmax is index 0, never 1.
    #[test]
    fn argmin_and_tie_policy_match_nn() {
        let model =
            translate_direct(TraceOp::Argmin { dim: 1 }, &[2, 4], &[2]).expect("Argmin translates");
        let spec = layer(&model, "layer0_trace_1");
        assert_eq!(spec.layer_type, LayerType::ArgMin);
        let out = ibp_direct(
            &model,
            ndarray::arr2(&[[-5.0f32, 8.0, 3.0, 4.0], [0.0, -2.0, 7.0, 6.0]]).into_dyn(),
            ndarray::arr2(&[[-4.0f32, 9.0, 4.0, 6.0], [1.0, -1.0, 9.0, 8.0]]).into_dyn(),
        );
        // Lane 0: element 0 dominates below; lane 1: element 1 dominates.
        assert_eq!(out.lower().as_slice().unwrap(), &[0.0, 1.0]);
        assert_eq!(out.upper().as_slice().unwrap(), &[0.0, 1.0]);

        // First-index tie pin (rank-2 to keep a rank-1 output).
        let model =
            translate_direct(TraceOp::Argmax { dim: 1 }, &[1, 3], &[1]).expect("Argmax translates");
        let tie = ndarray::arr2(&[[4.0f32, 4.0, 1.0]]).into_dyn();
        let out = ibp_direct(&model, tie.clone(), tie);
        assert_eq!(out.lower().as_slice().unwrap(), &[0.0]);
        assert_eq!(out.upper().as_slice().unwrap(), &[0.0]);
    }

    /// ArgSort lowers to LayerType::ArgSort carrying the serialized
    /// `descending` flag; IBP emits the exact permutation only for strictly
    /// separated lanes and the sound [0, n-1] hull under any overlap (nn's
    /// unstable sort leaves tie order unspecified).
    #[test]
    fn argsort_lowers_to_native_layer() {
        let model = translate_direct(
            TraceOp::ArgSort {
                dim: 0,
                descending: true,
            },
            &[3],
            &[3],
        )
        .expect("ArgSort translates");
        let spec = layer(&model, "layer0_trace_1");
        assert_eq!(spec.layer_type, LayerType::ArgSort);
        assert_eq!(spec.attributes.get("axis"), Some(&AttributeValue::Int(-1)));
        assert_eq!(
            spec.attributes.get("descending"),
            Some(&AttributeValue::Int(1))
        );

        // Concrete separated lane: descending order of [1, 4, 2] is exactly
        // indices [1, 2, 0].
        let concrete = ndarray::arr1(&[1.0f32, 4.0, 2.0]).into_dyn();
        let out = ibp_direct(&model, concrete.clone(), concrete);
        assert_eq!(out.lower().as_slice().unwrap(), &[1.0, 2.0, 0.0]);
        assert_eq!(out.upper().as_slice().unwrap(), &[1.0, 2.0, 0.0]);

        // Fully overlapping intervals: sound uniform [0, 2] hull.
        let out = ibp_direct(
            &model,
            ndarray::arr1(&[0.0f32, 0.0, 0.0]).into_dyn(),
            ndarray::arr1(&[5.0f32, 5.0, 5.0]).into_dyn(),
        );
        assert_eq!(out.lower().as_slice().unwrap(), &[0.0, 0.0, 0.0]);
        assert_eq!(out.upper().as_slice().unwrap(), &[2.0, 2.0, 2.0]);
    }

    /// Sort (descending) lowers to a single full-length Topk values spec —
    /// exact IBP because order statistics are monotone in every input — and
    /// Sort (ascending) appends the slice reversal.
    #[test]
    fn sort_lowers_to_full_length_topk_values() {
        // Descending: one Topk spec, no reversal machinery.
        let model = translate_direct(
            TraceOp::Sort {
                dim: 0,
                descending: true,
            },
            &[4],
            &[4],
        )
        .expect("Sort translates");
        let spec = layer(&model, "layer0_trace_1");
        assert_eq!(spec.layer_type, LayerType::Topk);
        assert_eq!(spec.attributes.get("axis"), Some(&AttributeValue::Int(-1)));
        assert_eq!(spec.attributes.get("k"), Some(&AttributeValue::Int(4)));
        assert_eq!(
            spec.attributes.get("output"),
            Some(&AttributeValue::String("values".to_string()))
        );
        assert_eq!(count(&model, &LayerType::Slice), 0);
        let concrete = ndarray::arr1(&[3.0f32, 1.0, 4.0, 2.0]).into_dyn();
        let out = ibp_direct(&model, concrete.clone(), concrete);
        assert_eq!(out.lower().as_slice().unwrap(), &[4.0, 3.0, 2.0, 1.0]);
        assert_eq!(out.upper().as_slice().unwrap(), &[4.0, 3.0, 2.0, 1.0]);

        // Ascending: Topk + N reversed slices + Concat.
        let model = translate_direct(
            TraceOp::Sort {
                dim: 0,
                descending: false,
            },
            &[4],
            &[4],
        )
        .expect("ascending Sort translates");
        assert_eq!(
            layer(&model, "layer0_trace_1_desc").layer_type,
            LayerType::Topk
        );
        assert_eq!(count(&model, &LayerType::Slice), 4);
        assert_eq!(
            layer(&model, "layer0_trace_1").layer_type,
            LayerType::Concat
        );
        let concrete = ndarray::arr1(&[3.0f32, 1.0, 4.0, 2.0]).into_dyn();
        let out = ibp_direct(&model, concrete.clone(), concrete);
        assert_eq!(out.lower().as_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(out.upper().as_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    }

    /// Interval inputs: sorted bounds are the per-rank order statistics of
    /// the bound vectors — exact even when the realized order is uncertain —
    /// and interior (non-trailing) axes remap correctly.
    #[test]
    fn sort_interval_and_interior_axis_bounds_are_exact() {
        // Ascending on overlapping intervals: rank-p lower bound is the p-th
        // smallest lower, rank-p upper bound the p-th smallest upper.
        let model = translate_direct(
            TraceOp::Sort {
                dim: 0,
                descending: false,
            },
            &[3],
            &[3],
        )
        .expect("Sort translates");
        let out = ibp_direct(
            &model,
            ndarray::arr1(&[1.0f32, 0.0, 2.0]).into_dyn(),
            ndarray::arr1(&[1.5f32, 0.5, 2.5]).into_dyn(),
        );
        assert_eq!(out.lower().as_slice().unwrap(), &[0.0, 1.0, 2.0]);
        assert_eq!(out.upper().as_slice().unwrap(), &[0.5, 1.5, 2.5]);

        // Descending along dim 0 of a rank-2 tensor (axis -2): per-column
        // descending sorts.
        let model = translate_direct(
            TraceOp::Sort {
                dim: 0,
                descending: true,
            },
            &[2, 3],
            &[2, 3],
        )
        .expect("interior-axis Sort translates");
        let spec = layer(&model, "layer0_trace_1");
        assert_eq!(spec.attributes.get("axis"), Some(&AttributeValue::Int(-2)));
        let concrete = ndarray::arr2(&[[3.0f32, 1.0, 2.0], [0.0, 5.0, 1.0]]).into_dyn();
        let out = ibp_direct(&model, concrete.clone(), concrete);
        let expected = ndarray::arr2(&[[3.0f32, 5.0, 2.0], [0.0, 1.0, 1.0]]).into_dyn();
        assert_eq!(out.lower(), &expected);
        assert_eq!(out.upper(), &expected);
    }

    /// Sort fail-closed edges: size-0 dims and an ascending reversal past the
    /// decomposition cap.
    #[test]
    fn sort_fail_closed_edges() {
        let err = translate_direct(
            TraceOp::Sort {
                dim: 0,
                descending: true,
            },
            &[0],
            &[0],
        )
        .expect_err("size-0 Sort must fail closed");
        assert!(
            matches!(err, NyError::UnsupportedOp(ref m) if m.contains("dim size is 0")),
            "got {err:?}"
        );

        let oversized = super::MAX_DECOMPOSE_DIM + 1;
        let err = translate_direct(
            TraceOp::Sort {
                dim: 0,
                descending: false,
            },
            &[oversized],
            &[oversized],
        )
        .expect_err("oversized ascending reversal must fail closed");
        assert!(
            matches!(err, NyError::UnsupportedOp(ref m) if m.contains("decomposition limit")),
            "got {err:?}"
        );
    }

    /// Topk lowers for `largest && sorted` and ONLY that combination.
    ///
    /// One assertion per quadrant, so the decision is pinned in both
    /// directions: a future change that lowers a second combination has to
    /// delete a refusal assertion here and justify it, and one that
    /// re-refuses the lowered combination has to delete the acceptance.
    #[test]
    fn topk_lowers_exactly_the_combination_topk_layer_models() {
        let model = translate_direct(
            TraceOp::Topk {
                k: 2,
                dim: 0,
                largest: true,
                sorted: true,
            },
            &[4],
            &[2],
        )
        .expect("largest && sorted is the function TopkLayer models");
        let spec = layer(&model, "layer0_trace_1");
        assert_eq!(spec.layer_type, LayerType::Topk);
        assert_eq!(
            spec.attributes.get("k"),
            Some(&AttributeValue::Int(2)),
            "k rides the wire"
        );
        assert_eq!(
            spec.attributes.get("axis"),
            Some(&AttributeValue::Int(-1)),
            "dim 0 of rank 1 encodes trailing-relative"
        );
        assert_eq!(
            spec.attributes.get("output"),
            Some(&AttributeValue::String("values".to_string())),
            "nn traces the VALUES tensor only"
        );

        // largest = false: the k smallest. TopkLayer sorts descending, so its
        // interval can exclude the traced value outright.
        for sorted in [true, false] {
            let err = translate_direct(
                TraceOp::Topk {
                    k: 2,
                    dim: 0,
                    largest: false,
                    sorted,
                },
                &[4],
                &[2],
            )
            .expect_err("k-smallest has no layer to land on");
            assert!(
                matches!(err, NyError::UnsupportedOp(ref m)
                    if m.contains("largest=false")
                        && m.contains("DESCENDING")
                        && m.contains("false proof")),
                "sorted={sorted}: got {err:?}"
            );
        }

        // sorted = false: the right SET, an unspecified order. The layer's
        // per-position interval is strictly narrower than the needed hull.
        let err = translate_direct(
            TraceOp::Topk {
                k: 2,
                dim: 0,
                largest: true,
                sorted: false,
            },
            &[4],
            &[2],
        )
        .expect_err("unsorted output order has no layer to land on");
        assert!(
            matches!(err, NyError::UnsupportedOp(ref m)
                if m.contains("sorted=false")
                    && m.contains("UNSPECIFIED order")
                    && m.contains("strictly narrower")),
            "got {err:?}"
        );
    }

    /// THE WIRE PIN for the Topk decision — the successor to the tripwire that
    /// fired when nn serialized the flags.
    ///
    /// The old test asserted the wire carried `{k, dim}` and NOTHING else, with
    /// a deliberately `..`-free pattern so that adding a field would break the
    /// build and force whoever added it back to the translate arm. That is
    /// exactly what happened: nn-core's `TraceOp::Topk` now carries `largest`
    /// and `sorted`, the mirror grew them, and the refusal that had outlived
    /// its reason was replaced by
    /// [`topk_lowers_exactly_the_combination_topk_layer_models`].
    ///
    /// The pattern stays `..`-free, pointed the other way: the four fields are
    /// now load-bearing, and a FIFTH would change which function the wire
    /// names. Breaking the build again is the correct outcome.
    #[test]
    fn topk_wire_carries_the_selection_semantics() {
        let op = TraceOp::Topk {
            k: 3,
            dim: 1,
            largest: false,
            sorted: false,
        };
        // No `..`: adding a field to the variant is a compile error here.
        #[allow(clippy::match_wildcard_for_single_variants)]
        let (k, dim, largest, sorted) = match &op {
            TraceOp::Topk {
                k,
                dim,
                largest,
                sorted,
            } => (*k, *dim, *largest, *sorted),
            other => panic!("constructed a Topk, got {other:?}"),
        };
        assert_eq!(
            (k, dim, largest, sorted),
            (3, 1, false, false),
            "the wire carries the selection SIZE, AXIS, END of the lane, and ORDER — the four \
             fields that pick out which of the four functions ran"
        );
    }

    /// The flags decide, and the payload does not rescue or condemn: the
    /// lowered combination lowers for every `{k, dim}` and the refused ones
    /// stay refused for every `{k, dim}`.
    #[test]
    fn topk_decision_is_the_flags_not_the_payload() {
        for (k, dim, rank_len) in [(1usize, 0usize, 4usize), (2, 0, 4), (4, 0, 4)] {
            translate_direct(
                TraceOp::Topk {
                    k,
                    dim,
                    largest: true,
                    sorted: true,
                },
                &[rank_len],
                &[k],
            )
            .unwrap_or_else(|e| panic!("k={k} dim={dim} must lower: {e:?}"));

            for (largest, sorted) in [(false, true), (true, false), (false, false)] {
                let err = translate_direct(
                    TraceOp::Topk {
                        k,
                        dim,
                        largest,
                        sorted,
                    },
                    &[rank_len],
                    &[k],
                )
                .expect_err("refused combination must stay refused for every payload");
                assert!(
                    matches!(err, NyError::UnsupportedOp(ref m)
                        if m.contains("Only largest=true, sorted=true lowers")),
                    "k={k} largest={largest} sorted={sorted}: got {err:?}"
                );
            }
        }
    }

    /// A lowered Topk builds and propagates, and its interval ENCLOSES the
    /// exact k-largest selection at a point input.
    ///
    /// The enclosure is the whole claim: `TopkLayer` sorts each bound lane
    /// descending, which at a degenerate box is the exact descending sort, so
    /// output position j must equal the (j+1)-th largest input.
    #[test]
    fn topk_ibp_encloses_the_k_largest() {
        let model = translate_direct(
            TraceOp::Topk {
                k: 3,
                dim: 0,
                largest: true,
                sorted: true,
            },
            &[6],
            &[3],
        )
        .expect("Topk translates");

        // Point box: the layer's descending sort of each bound lane is then
        // the exact descending sort, so every output position must be pinned
        // to the corresponding order statistic.
        let values = [3.0_f32, -1.0, 7.0, 0.5, 7.0, -4.0];
        let point = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[6]), values.to_vec()).unwrap();
        let out = ibp_direct(&model, point.clone(), point);
        assert_eq!(out.shape(), &[3]);

        let mut sorted_desc = values;
        sorted_desc.sort_unstable_by(|a, b| b.total_cmp(a));
        for j in 0..3 {
            let (lo, hi) = (out.lower()[j], out.upper()[j]);
            assert!(
                lo <= sorted_desc[j] && sorted_desc[j] <= hi,
                "position {j}: {} not enclosed by [{lo}, {hi}]",
                sorted_desc[j]
            );
        }
        assert_eq!(out.lower().as_slice().unwrap(), &[7.0, 7.0, 3.0]);
        assert_eq!(out.upper().as_slice().unwrap(), &[7.0, 7.0, 3.0]);

        // A genuine box: the enclosure must still hold at every corner it can
        // reach, which for order statistics is the pair of bound-vector sorts.
        let lower = ndarray::ArrayD::from_shape_vec(
            ndarray::IxDyn(&[6]),
            vec![0.0_f32, -2.0, 5.0, 0.0, 6.0, -5.0],
        )
        .unwrap();
        let upper = ndarray::ArrayD::from_shape_vec(
            ndarray::IxDyn(&[6]),
            vec![4.0_f32, 1.0, 9.0, 1.0, 8.0, -3.0],
        )
        .unwrap();
        let out = ibp_direct(&model, lower, upper);
        assert_eq!(out.lower().as_slice().unwrap(), &[6.0, 5.0, 0.0]);
        assert_eq!(out.upper().as_slice().unwrap(), &[9.0, 8.0, 4.0]);
    }

    // -----------------------------------------------------------------------
    // Scatter family
    // -----------------------------------------------------------------------

    /// A `ConstantWeight` node carrying integer-by-semantics values.
    fn const_weight(id: u64, name: &str, values: Vec<f32>, shape: Vec<usize>) -> TraceNode {
        let dims = shape.clone();
        node(
            id,
            name,
            TraceOp::ConstantWeight {
                weight: WeightPayload::f32(values, shape),
            },
            &[],
            &dims,
        )
    }

    fn bounds(lower: &[f32], upper: &[f32], shape: &[usize]) -> ny_tensor::BoundedTensor {
        let lower = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(shape), lower.to_vec())
            .expect("lower shape");
        let upper = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(shape), upper.to_vec())
            .expect("upper shape");
        ny_tensor::BoundedTensor::new(lower, upper).expect("bounds")
    }

    fn network(model: &GraphModel, what: &str) -> ny_propagate::GraphNetwork {
        model
            .build_graph_network(ny_build::GraphNetworkOptions::default())
            .unwrap_or_else(|e| panic!("{what}: graph network must build: {e:?}"))
    }

    /// `ScatterAdd` / `IndexAdd` emit their native layer with the three trace
    /// operands intact and a TRAILING-RELATIVE axis: `dim 0` on a rank-2 data
    /// tensor becomes `-2`, which names the same dimension whether or not the
    /// runtime tensor kept a leading batch axis.
    #[test]
    fn additive_scatter_ops_emit_native_layers_with_trailing_axis() {
        for (op, expected_type) in [
            (TraceOp::ScatterAdd { dim: 0 }, LayerType::ScatterAdd),
            (TraceOp::IndexAdd { dim: 0 }, LayerType::IndexAdd),
        ] {
            let index_values = match expected_type {
                // ScatterAdd's index is shaped like src; IndexAdd's is 1-D.
                LayerType::ScatterAdd => (vec![1.0, 1.0, 1.0, 3.0, 3.0, 3.0], vec![2, 3]),
                _ => (vec![1.0, 3.0], vec![2]),
            };
            let graph = ComputationGraph::from_nodes(vec![
                node(0, "data", TraceOp::Input, &[], &[4, 3]),
                const_weight(1, "index", index_values.0, index_values.1),
                const_weight(2, "src", vec![1.0; 6], vec![2, 3]),
                node(3, "acc", op.clone(), &[0, 1, 2], &[4, 3]),
            ]);
            let model = translate(&graph).unwrap_or_else(|e| panic!("{op:?} translates: {e:?}"));
            let spec = layer(&model, "layer0_trace_3");
            assert_eq!(spec.layer_type, expected_type, "{op:?}");
            assert_eq!(
                spec.attributes.get("axis"),
                Some(&AttributeValue::Int(-2)),
                "{op:?} axis is trailing-relative"
            );
            assert_eq!(spec.inputs.len(), 3, "{op:?} keeps (data, index, src)");
            assert_builds(&model, "additive scatter");
        }
    }

    /// gptoss's MoE combine (`crates/nn-gptoss/src/moe_dispatch.rs`:
    /// `output = output.index_add(0, &ids_tensor, &weighted)`) end-to-end:
    /// `dim 0`, a 1-D token-id index, and a `[num_routed, hidden]` src
    /// accumulated into a `[num_tokens, hidden]` destination. Token 1 is hit by
    /// two routed rows, which must SUM (accumulation is order-independent).
    #[test]
    fn gptoss_shaped_index_add_accumulates_rows_end_to_end() {
        let graph = ComputationGraph::from_nodes(vec![
            // The destination is the variable: the routed expert outputs ride
            // in as a captured constant, as nn's untraced `from_vec` tensors do.
            node(0, "output", TraceOp::Input, &[], &[4, 2]),
            const_weight(1, "ids", vec![1.0, 3.0, 1.0], vec![3]),
            const_weight(
                2,
                "weighted",
                vec![1.0, 2.0, 4.0, 5.0, 10.0, 20.0],
                vec![3, 2],
            ),
            node(
                3,
                "combine",
                TraceOp::IndexAdd { dim: 0 },
                &[0, 1, 2],
                &[4, 2],
            ),
        ]);
        let model = translate(&graph).expect("gptoss-shaped IndexAdd translates");
        let out = network(&model, "gptoss IndexAdd")
            .propagate_ibp(&bounds(&[0.0; 8], &[0.0; 8], &[4, 2]))
            .expect("IndexAdd IBP");

        assert_eq!(
            out.shape(),
            &[4, 2],
            "index_add preserves the destination shape"
        );
        // token 0 untouched; token 1 = row0 + row2; token 2 untouched; token 3 = row1.
        let expected = [0.0, 0.0, 11.0, 22.0, 0.0, 0.0, 4.0, 5.0];
        assert_eq!(out.lower().iter().copied().collect::<Vec<_>>(), expected);
        assert_eq!(out.upper().iter().copied().collect::<Vec<_>>(), expected);
    }

    /// A NON-constant index does not refuse `ScatterAdd`: it stays dynamic,
    /// and the layer widens `data` by the total possible additive contribution
    /// of `src`. Every index assignment routes each src element to exactly one
    /// cell, so the hull encloses all of them — verified here by enumerating
    /// every assignment of the concrete src corners.
    #[test]
    fn scatter_add_with_a_runtime_index_stays_dynamic_and_hulls_conservatively() {
        // The index producer is an F32-dtype activation (a Relu), so the graph
        // dtype gate is not what this test is about.
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[2]),
            node(1, "index", TraceOp::Relu, &[0], &[2]),
            const_weight(2, "data", vec![0.0, 0.0, 0.0], vec![3]),
            node(3, "sa", TraceOp::ScatterAdd { dim: 0 }, &[2, 1, 0], &[3]),
        ]);
        let model = translate(&graph).expect("dynamic-index ScatterAdd translates");
        let spec = layer(&model, "layer0_trace_3");
        assert_eq!(spec.layer_type, LayerType::ScatterAdd);

        // src == x, bounded in [-2, -1] and [1, 4] respectively.
        let out = network(&model, "dynamic ScatterAdd")
            .propagate_ibp(&bounds(&[-2.0, 1.0], &[-1.0, 4.0], &[2]))
            .expect("dynamic ScatterAdd IBP");
        let lower: Vec<f32> = out.lower().iter().copied().collect();
        let upper: Vec<f32> = out.upper().iter().copied().collect();

        for i0 in 0..3usize {
            for i1 in 0..3usize {
                for s0 in [-2.0_f32, -1.0] {
                    for s1 in [1.0_f32, 4.0] {
                        let mut concrete = [0.0_f32; 3];
                        concrete[i0] += s0;
                        concrete[i1] += s1;
                        for (cell, value) in concrete.iter().enumerate() {
                            assert!(
                                lower[cell] <= *value && *value <= upper[cell],
                                "dynamic hull [{}, {}] missed {value} at cell {cell} for \
                                 index ({i0}, {i1})",
                                lower[cell],
                                upper[cell]
                            );
                        }
                    }
                }
            }
        }
    }

    /// A constant-index `Scatter` lowers to `ScatterND` with a SYNTHESIZED
    /// absolute-coordinate index, and `src` passes through unchanged (the
    /// coordinate tensor's prefix is `src.shape()`, so ScatterND's expected
    /// updates shape is exactly `src`'s — no reshape needed).
    #[test]
    fn scatter_overwrite_lowers_a_constant_index_to_scatter_nd() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "data", TraceOp::Input, &[], &[2, 3]),
            // dim 1: row 0 writes column 0 twice, row 1 writes column 2 twice.
            const_weight(1, "index", vec![0.0, 0.0, 2.0, 2.0], vec![2, 2]),
            const_weight(2, "src", vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
            node(3, "sc", TraceOp::Scatter { dim: 1 }, &[0, 1, 2], &[2, 3]),
        ]);
        let model = translate(&graph).expect("constant-index Scatter translates");
        let spec = layer(&model, "layer0_trace_3");
        assert_eq!(spec.layer_type, LayerType::ScatterND);
        assert_eq!(spec.inputs.len(), 3);
        assert_eq!(spec.inputs[1], "layer0_trace_3_nd_indices");
        assert_eq!(
            spec.inputs[2], "layer0_trace_2_out",
            "src feeds ScatterND unchanged — no reshape node"
        );

        // Coordinates are `src_shape ++ [data_rank]` = [2, 2, 2]: each src
        // element's own coordinate with axis 1 replaced by its destination.
        // They ride the f32 weight surface, and every one round-trips exactly
        // (all are far below 2^24 — see `insert_index_constant`).
        let indices = model
            .weights
            .get("layer0_trace_3_nd_indices")
            .expect("the synthesized ND index constant must be in the weight store");
        assert_eq!(indices.shape(), &[2, 2, 2]);
        assert_eq!(
            indices.iter().map(|&v| v as i64).collect::<Vec<_>>(),
            vec![0, 0, /**/ 0, 0, /**/ 1, 2, /**/ 1, 2]
        );
        assert!(
            indices.iter().all(|v| v.fract() == 0.0),
            "coordinates must be exact integers in the f32 mirror"
        );
        assert_builds(&model, "constant-index Scatter");
    }

    /// THE soundness test for the overwrite. With duplicate destinations the
    /// surviving value is whichever write landed LAST, and nn does not fix that
    /// order — its CPU `scatter_loop` walks src row-major while `scatter_gpu`
    /// races parallel writes. So the destination's value is a SET, and the
    /// emitted bounds must contain EVERY candidate, not one of them.
    #[test]
    fn scatter_overwrite_duplicate_indices_enclose_both_write_orders() {
        // index = [1, 1]: both src elements target cell 1 of a [3] destination.
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "src", TraceOp::Input, &[], &[2]),
            const_weight(1, "data", vec![7.0, 7.0, 7.0], vec![3]),
            const_weight(2, "index", vec![1.0, 1.0], vec![2]),
            node(3, "sc", TraceOp::Scatter { dim: 0 }, &[1, 2, 0], &[3]),
        ]);
        let model = translate(&graph).expect("duplicate-index Scatter translates");
        // Exact src values, so the ONLY imprecision left is the write order.
        let out = network(&model, "duplicate-index Scatter")
            .propagate_ibp(&bounds(&[2.0, 5.0], &[2.0, 5.0], &[2]))
            .expect("duplicate-index Scatter IBP");
        let lower: Vec<f32> = out.lower().iter().copied().collect();
        let upper: Vec<f32> = out.upper().iter().copied().collect();

        // Untouched cells keep `data` exactly.
        assert_eq!((lower[0], upper[0]), (7.0, 7.0));
        assert_eq!((lower[2], upper[2]), (7.0, 7.0));
        // The contested cell must enclose BOTH resolutions.
        for candidate in [2.0_f32, 5.0] {
            assert!(
                lower[1] <= candidate && candidate <= upper[1],
                "duplicate-destination bounds [{}, {}] must contain the {candidate} write order",
                lower[1],
                upper[1]
            );
        }
        // ...and it must NOT have kept the overwritten `data` value, which no
        // write order can produce (the cell is certainly written).
        assert_eq!(
            (lower[1], upper[1]),
            (2.0, 5.0),
            "the contested cell is the union of the candidate writes, nothing wider or narrower"
        );
    }

    /// The union holds for interval-valued src too: the cell's bounds are the
    /// hull of the candidate writes' intervals.
    #[test]
    fn scatter_overwrite_duplicate_indices_hull_candidate_intervals() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "src", TraceOp::Input, &[], &[3]),
            const_weight(1, "data", vec![0.0, 0.0], vec![2]),
            // Cells 0 and 0 contested; cell 1 written once.
            const_weight(2, "index", vec![0.0, 0.0, 1.0], vec![3]),
            node(3, "sc", TraceOp::Scatter { dim: 0 }, &[1, 2, 0], &[2]),
        ]);
        let model = translate(&graph).expect("interval Scatter translates");
        let out = network(&model, "interval Scatter")
            .propagate_ibp(&bounds(&[-1.0, 2.0, 4.0], &[1.0, 3.0, 5.0], &[3]))
            .expect("interval Scatter IBP");
        // Contested cell 0: hull of [-1, 1] and [2, 3] = [-1, 3].
        assert_eq!(
            out.lower().iter().copied().collect::<Vec<_>>(),
            vec![-1.0, 4.0]
        );
        assert_eq!(
            out.upper().iter().copied().collect::<Vec<_>>(),
            vec![3.0, 5.0]
        );
    }

    /// A RUNTIME index is refused, and the message states the real reason —
    /// the collision pattern is itself a runtime value, so the union that
    /// makes the overwrite soundly lowerable cannot be formed statically.
    #[test]
    fn scatter_overwrite_with_a_runtime_index_is_refused_for_the_duplicate_reason() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[2]),
            node(1, "index", TraceOp::Relu, &[0], &[2]),
            const_weight(2, "data", vec![0.0, 0.0, 0.0], vec![3]),
            node(3, "sc", TraceOp::Scatter { dim: 0 }, &[2, 1, 0], &[3]),
        ]);
        let err = translate(&graph).expect_err("a runtime Scatter index must fail closed");
        assert!(
            matches!(err, NyError::UnsupportedOp(ref m)
                if m.contains("not a translate-time constant")
                    && m.contains("OVERWRITE")
                    && m.contains("landed last")),
            "the refusal must state the duplicate-write reason, got {err:?}"
        );
    }

    /// Out-of-range destinations fail closed instead of wrapping or clamping
    /// onto a real cell; in-range negatives follow the Python convention.
    #[test]
    fn scatter_overwrite_index_range_is_enforced() {
        let scatter_with = |index: f32| {
            let graph = ComputationGraph::from_nodes(vec![
                node(0, "src", TraceOp::Input, &[], &[1]),
                const_weight(1, "data", vec![0.0, 0.0, 0.0], vec![3]),
                const_weight(2, "index", vec![index], vec![1]),
                node(3, "sc", TraceOp::Scatter { dim: 0 }, &[1, 2, 0], &[3]),
            ]);
            translate(&graph)
        };

        for out_of_range in [3.0_f32, -4.0, 100.0] {
            let err = scatter_with(out_of_range)
                .expect_err("out-of-range Scatter index must fail closed");
            assert!(
                matches!(err, NyError::UnsupportedOp(ref m) if m.contains("out of bounds")),
                "index {out_of_range}: got {err:?}"
            );
        }

        // -1 names the last cell.
        let model = scatter_with(-1.0).expect("in-range negative index translates");
        let out = network(&model, "negative Scatter index")
            .propagate_ibp(&bounds(&[9.0], &[9.0], &[1]))
            .expect("negative-index Scatter IBP");
        assert_eq!(
            out.lower().iter().copied().collect::<Vec<_>>(),
            vec![0.0, 0.0, 9.0]
        );
    }

    /// A non-integral index value is refused rather than rounded: a write
    /// destination that has to be guessed is exactly the fail-closed case.
    #[test]
    fn scatter_overwrite_non_integral_index_is_refused() {
        for value in [
            1.5_f32,
            f32::from_bits(1.0_f32.to_bits() + 1),
            f32::from_bits(1.0_f32.to_bits() - 1),
        ] {
            let graph = ComputationGraph::from_nodes(vec![
                node(0, "src", TraceOp::Input, &[], &[1]),
                const_weight(1, "data", vec![0.0, 0.0, 0.0], vec![3]),
                const_weight(2, "index", vec![value], vec![1]),
                node(3, "sc", TraceOp::Scatter { dim: 0 }, &[1, 2, 0], &[3]),
            ]);
            let err = translate(&graph).expect_err("a non-integral index must fail closed");
            assert!(
                matches!(err, NyError::UnsupportedOp(ref m) if m.contains("not an exact integer")),
                "index {value}: got {err:?}"
            );
        }
    }

    /// THE f32 exact-integer boundary. `2^24` is the FIRST magnitude at which
    /// an index is ambiguous — `2^24 + 1` is not f32-representable and rounds
    /// down onto exactly `2^24` — so the guard has to be `>=`, matching the
    /// "at or beyond" wording it and its `insert_index_constant` sibling share.
    ///
    /// The reviewer probe is the NEGATIVE branch, which is where a `>` guard
    /// does real damage rather than merely failing an out-of-range check: an
    /// authored `-(2^24 + 1)` arrives as `-2^24`, and `normalize_scatter_index`
    /// adds `axis_len` to it. On an axis LONGER than 2^24 both the authored and
    /// the corrupted value normalize into range — to DIFFERENT cells — so the
    /// scatter would silently write the wrong destination with no diagnostic.
    /// Here `axis_len = 2^24 + 4`: the authored index means cell 3, the
    /// captured one would mean cell 4.
    #[test]
    fn scatter_overwrite_index_at_the_f32_exact_integer_boundary_is_refused() {
        const LIMIT: f32 = 16_777_216.0; // 2^24
                                         // The premise, asserted rather than assumed: the authored index
                                         // -(2^24 + 1) is NOT representable and lands on -2^24 by the time it
                                         // reaches the weight surface, so the two are indistinguishable here.
        assert_eq!(-16_777_217.0_f32, -LIMIT, "2^24 + 1 collapses onto 2^24");
        let axis_len = 16_777_220_usize; // 2^24 + 4: both values normalize in range

        // A one-element src keeps the synthesized ND index tiny; the large
        // extent is only a RECORDED shape, so nothing materializes it (and
        // `data` stays a graph input rather than a 64 MB constant weight —
        // hence `translate_multi_input`).
        let scatter_with = |index: f32| {
            let graph = ComputationGraph::from_nodes(vec![
                node(0, "src", TraceOp::Input, &[], &[1]),
                node(1, "data", TraceOp::Input, &[], &[axis_len]),
                const_weight(2, "index", vec![index], vec![1]),
                node(
                    3,
                    "sc",
                    TraceOp::Scatter { dim: 0 },
                    &[1, 2, 0],
                    &[axis_len],
                ),
            ]);
            translate_multi_input(&graph).map(|t| t.model)
        };

        // The negative branch: the corrupted value must fail closed, not
        // normalize onto cell 4.
        for boundary in [-LIMIT, LIMIT, -16_777_217.0_f32, 16_777_217.0_f32] {
            let err =
                scatter_with(boundary).expect_err("an index at or beyond 2^24 must fail closed");
            assert!(
                matches!(err, NyError::UnsupportedOp(ref m)
                    if m.contains("at or beyond") && m.contains("exact-integer limit")),
                "index {boundary}: expected the exact-integer refusal, got {err:?}"
            );
        }

        // ...and the guard is not over-broad: one ULP below the limit every
        // integer is still exact, so `2^24 - 1` gets PAST the exact-integer
        // guard and is accepted as the real destination it names.
        let model = scatter_with(LIMIT - 1.0).expect("2^24 - 1 is an exact, in-range index");
        let indices = model
            .weights
            .get("layer0_trace_3_nd_indices")
            .expect("synthesized ND index constant");
        assert_eq!(
            indices.iter().map(|&v| v as i64).collect::<Vec<_>>(),
            vec![16_777_215],
            "the last exactly-representable index round-trips to its own cell"
        );
    }

    /// Shape disagreements between the index and src operands are refused
    /// (torch requires them equal for `scatter`).
    #[test]
    fn scatter_overwrite_index_src_shape_mismatch_is_refused() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "src", TraceOp::Input, &[], &[2]),
            const_weight(1, "data", vec![0.0; 3], vec![3]),
            const_weight(2, "index", vec![0.0], vec![1]),
            node(3, "sc", TraceOp::Scatter { dim: 0 }, &[1, 2, 0], &[3]),
        ]);
        let err = translate(&graph).expect_err("index/src shape mismatch must fail closed");
        assert!(
            matches!(err, NyError::UnsupportedOp(ref m) if m.contains("must equal src shape")),
            "got {err:?}"
        );
    }

    /// The scatter family's index operand is an integer carrier in real nn
    /// traces. A translate-time-constant integer leaf in position 1 passes the
    /// graph-wide F32 dtype gate for all three ops — the same justification as
    /// the IndexSelect/Gather index slot: the translator resolves the payload
    /// at translate/build time and never lowers the tensor into the runtime
    /// graph.
    #[test]
    fn integer_scatter_index_leaf_passes_the_dtype_gate() {
        for dtype in [DType::U32, DType::I64, DType::I32, DType::U8] {
            for (op, index) in [
                (TraceOp::Scatter { dim: 0 }, vec![1.0, 2.0]),
                (TraceOp::ScatterAdd { dim: 0 }, vec![1.0, 2.0]),
                (TraceOp::IndexAdd { dim: 0 }, vec![1.0, 2.0]),
            ] {
                let mut index_node = const_weight(1, "index", index, vec![2]);
                index_node.output_dtype = dtype;
                let graph = ComputationGraph::from_nodes(vec![
                    node(0, "data", TraceOp::Input, &[], &[3]),
                    index_node,
                    const_weight(2, "src", vec![1.0, 1.0], vec![2]),
                    node(3, "s", op.clone(), &[0, 1, 2], &[3]),
                ]);
                translate(&graph).unwrap_or_else(|e| {
                    panic!("{op:?} with a {dtype:?} index leaf must pass the gate: {e:?}")
                });
            }
        }
    }

    /// The carve-out is POSITION-scoped: an integer constant in the scatter's
    /// DATA slot (position 0) is still refused — that value IS consumed
    /// numerically by the emitted graph.
    #[test]
    fn integer_leaf_in_the_scatter_data_slot_stays_refused() {
        let mut data = const_weight(0, "data", vec![1.0, 2.0, 3.0], vec![3]);
        data.output_dtype = DType::U32;
        let graph = ComputationGraph::from_nodes(vec![
            data,
            const_weight(1, "index", vec![0.0, 1.0], vec![2]),
            node(2, "src", TraceOp::Input, &[], &[2]),
            node(3, "sa", TraceOp::ScatterAdd { dim: 0 }, &[0, 1, 2], &[3]),
        ]);
        let err = translate(&graph).expect_err("an integer DATA leaf must fail closed");
        assert!(
            matches!(err, NyError::UnsupportedOp(ref m)
                if m.contains("output dtype") && m.contains("only soundly supports F32")),
            "got {err:?}"
        );
    }

    /// Wrong operand counts fail closed for all four ops.
    #[test]
    fn scatter_family_requires_three_operands() {
        for op in [
            TraceOp::Scatter { dim: 0 },
            TraceOp::ScatterAdd { dim: 0 },
            TraceOp::IndexAdd { dim: 0 },
            TraceOp::IndexPut { dim: 0 },
        ] {
            let graph = ComputationGraph::from_nodes(vec![
                node(0, "data", TraceOp::Input, &[], &[3]),
                const_weight(1, "index", vec![0.0], vec![1]),
                node(2, "s", op.clone(), &[0, 1], &[3]),
            ]);
            let err = translate(&graph).expect_err("a 2-operand scatter must fail closed");
            assert!(
                matches!(err, NyError::UnsupportedOp(ref m) if m.contains("requires 3 inputs")),
                "{op:?}: got {err:?}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // IndexPut — the constant-index case, rewritten exactly like Scatter
    // -----------------------------------------------------------------------

    /// A constant-index `IndexPut` lowers to `ScatterND` with a SYNTHESIZED
    /// absolute-coordinate index, and `src` passes through unchanged.
    ///
    /// The one place it differs from `Scatter`: the index is 1-D and names a
    /// destination per SLICE, so every src element in slice `j` along the write
    /// axis gets coordinate `index[j]` there. Here `data` is `[2, 3]`,
    /// `index = [2, 0]` and `dim = 1`, so src column 0 lands in data column 2
    /// and src column 1 in data column 0 — for BOTH rows.
    #[test]
    fn index_put_lowers_a_constant_index_to_scatter_nd() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "data", TraceOp::Input, &[], &[2, 3]),
            const_weight(1, "index", vec![2.0, 0.0], vec![2]),
            const_weight(2, "src", vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
            node(3, "ip", TraceOp::IndexPut { dim: 1 }, &[0, 1, 2], &[2, 3]),
        ]);
        let model = translate(&graph).expect("constant-index IndexPut translates");
        let spec = layer(&model, "layer0_trace_3");
        assert_eq!(spec.layer_type, LayerType::ScatterND);
        assert_eq!(spec.inputs.len(), 3);
        assert_eq!(spec.inputs[1], "layer0_trace_3_nd_indices");
        assert_eq!(
            spec.inputs[2], "layer0_trace_2_out",
            "src feeds ScatterND unchanged — no reshape node"
        );

        // `src_shape ++ [data_rank]` = [2, 2, 2]: each src element's own
        // coordinate with axis 1 replaced by its COLUMN's destination.
        let indices = model
            .weights
            .get("layer0_trace_3_nd_indices")
            .expect("the synthesized ND index constant must be in the weight store");
        assert_eq!(indices.shape(), &[2, 2, 2]);
        assert_eq!(
            indices.iter().map(|&v| v as i64).collect::<Vec<_>>(),
            vec![0, 2, /**/ 0, 0, /**/ 1, 2, /**/ 1, 0]
        );
        assert_builds(&model, "constant-index IndexPut");
    }

    /// THE parity test: the emitted bounds must ENCLOSE nn's own `index_put`
    /// evaluation, cell for cell, over a fixture with an interesting
    /// permutation. `src` is the only variable, held to exact points, so the
    /// only thing under test is whether the rewrite writes nn's destinations.
    ///
    /// The expected values replicate nn-core `index_put`'s N-D path
    /// operation-for-operation: `dst_coord = src_coord` with
    /// `dst_coord[dim] = indices[src_coord[dim]]`, everything else keeping
    /// `self`.
    #[test]
    fn index_put_encloses_nn_index_put_on_a_dense_fixture() {
        // data [2, 4] = const 7; src [2, 2] variable; index [3, 0], dim 1.
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "src", TraceOp::Input, &[], &[2, 2]),
            const_weight(1, "data", vec![7.0; 8], vec![2, 4]),
            const_weight(2, "index", vec![3.0, 0.0], vec![2]),
            node(3, "ip", TraceOp::IndexPut { dim: 1 }, &[1, 2, 0], &[2, 4]),
        ]);
        let model = translate(&graph).expect("IndexPut translates");
        let src_values = [10.0_f32, 20.0, 30.0, 40.0];
        let out = network(&model, "IndexPut parity")
            .propagate_ibp(&bounds(&src_values, &src_values, &[2, 2]))
            .expect("IndexPut IBP");
        let lower: Vec<f32> = out.lower().iter().copied().collect();
        let upper: Vec<f32> = out.upper().iter().copied().collect();

        // nn's evaluation, replayed here rather than hand-written.
        let index = [3usize, 0];
        let mut expected = [7.0_f32; 8];
        for row in 0..2usize {
            for col in 0..2usize {
                expected[row * 4 + index[col]] = src_values[row * 2 + col];
            }
        }
        for (cell, value) in expected.iter().enumerate() {
            assert!(
                lower[cell] <= *value && *value <= upper[cell],
                "cell {cell}: bounds [{}, {}] must enclose nn's {value}",
                lower[cell],
                upper[cell]
            );
        }
        // No duplicates in this fixture, so the enclosure is EXACT — a looser
        // bound here would mean the rewrite lost precision it did not have to.
        assert_eq!(lower, expected.to_vec());
        assert_eq!(upper, expected.to_vec());
    }

    /// THE soundness test for the overwrite semantics. With a duplicated
    /// destination the surviving value is one of the candidate writes, and the
    /// wire does not fix which: nn documents last-write-wins, but nn-import
    /// emits this same op for `aten.index_put` with `accumulate=false`, which
    /// PyTorch documents as nondeterministic on duplicate indices. The bounds
    /// must therefore contain EVERY candidate — union, never pick-one.
    #[test]
    fn index_put_duplicate_destinations_enclose_every_write_order() {
        // index = [1, 1]: both src slices target column 1 of a [3] destination.
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "src", TraceOp::Input, &[], &[2]),
            const_weight(1, "data", vec![7.0, 7.0, 7.0], vec![3]),
            const_weight(2, "index", vec![1.0, 1.0], vec![2]),
            node(3, "ip", TraceOp::IndexPut { dim: 0 }, &[1, 2, 0], &[3]),
        ]);
        let model = translate(&graph).expect("duplicate-index IndexPut translates");
        let out = network(&model, "duplicate-index IndexPut")
            .propagate_ibp(&bounds(&[2.0, 5.0], &[2.0, 5.0], &[2]))
            .expect("duplicate-index IndexPut IBP");
        let lower: Vec<f32> = out.lower().iter().copied().collect();
        let upper: Vec<f32> = out.upper().iter().copied().collect();

        // Untouched cells keep `data` exactly.
        assert_eq!((lower[0], upper[0]), (7.0, 7.0));
        assert_eq!((lower[2], upper[2]), (7.0, 7.0));
        // The contested cell encloses both resolutions — including nn's own
        // "last wins" answer, 5.0 — and nothing wider or narrower.
        for candidate in [2.0_f32, 5.0] {
            assert!(
                lower[1] <= candidate && candidate <= upper[1],
                "duplicate-destination bounds [{}, {}] must contain the {candidate} write order",
                lower[1],
                upper[1]
            );
        }
        assert_eq!(
            (lower[1], upper[1]),
            (2.0, 5.0),
            "the contested cell is the union of the candidate writes"
        );
        // ...and it does NOT keep the overwritten `data` value, which no write
        // order can produce: the cell is certainly written.
        assert!(
            !(lower[1] <= 7.0 && 7.0 <= upper[1]),
            "a certainly-overwritten cell must not still admit the old data value"
        );
    }

    /// A RUNTIME index is refused, and the message states the real reason —
    /// the collision pattern is itself a runtime value, so the union that makes
    /// the overwrite soundly lowerable cannot be formed statically. This is the
    /// half of the old blanket refusal that survives.
    #[test]
    fn index_put_with_a_runtime_index_is_refused_for_the_duplicate_reason() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[2]),
            node(1, "index", TraceOp::Relu, &[0], &[2]),
            const_weight(2, "data", vec![0.0, 0.0, 0.0], vec![3]),
            node(3, "ip", TraceOp::IndexPut { dim: 0 }, &[2, 1, 0], &[3]),
        ]);
        let err = translate(&graph).expect_err("a runtime IndexPut index must fail closed");
        assert!(
            matches!(err, NyError::UnsupportedOp(ref m)
                if m.contains("not a translate-time constant")
                    && m.contains("OVERWRITE")
                    && m.contains("landed last")),
            "the refusal must state the duplicate-write reason, got {err:?}"
        );
    }

    /// Unlike `Scatter`, `IndexPut` has NO negative-index convention: nn-core's
    /// `index_put` rejects a negative `I64` index outright and its `U32` path
    /// cannot express one, so a negative value here is a tensor nn would have
    /// refused to run. Wrapping it would bound a computation the traced program
    /// cannot perform, so it fails closed — and the contrast with `Scatter`,
    /// where `-1` names the last cell, is asserted in the same test so the two
    /// conventions cannot quietly converge.
    #[test]
    fn index_put_refuses_negative_indices_where_scatter_wraps_them() {
        let put_with = |index: f32| {
            let graph = ComputationGraph::from_nodes(vec![
                node(0, "src", TraceOp::Input, &[], &[1]),
                const_weight(1, "data", vec![0.0, 0.0, 0.0], vec![3]),
                const_weight(2, "index", vec![index], vec![1]),
                node(3, "ip", TraceOp::IndexPut { dim: 0 }, &[1, 2, 0], &[3]),
            ]);
            translate(&graph)
        };

        let err = put_with(-1.0).expect_err("a negative IndexPut index must fail closed");
        assert!(
            matches!(err, NyError::UnsupportedOp(ref m)
                if m.contains("is negative") && m.contains("no negative-index convention")),
            "got {err:?}"
        );

        // The sibling op DOES wrap -1 onto the last cell — same magnitude,
        // different op, different answer, because nn's two kernels differ.
        let scatter = ComputationGraph::from_nodes(vec![
            node(0, "src", TraceOp::Input, &[], &[1]),
            const_weight(1, "data", vec![0.0, 0.0, 0.0], vec![3]),
            const_weight(2, "index", vec![-1.0], vec![1]),
            node(3, "sc", TraceOp::Scatter { dim: 0 }, &[1, 2, 0], &[3]),
        ]);
        translate(&scatter).expect("Scatter's Python convention still accepts -1");

        // Out-of-range positives fail closed on both.
        for out_of_range in [3.0_f32, 100.0] {
            let err =
                put_with(out_of_range).expect_err("out-of-range IndexPut index must fail closed");
            assert!(
                matches!(err, NyError::UnsupportedOp(ref m) if m.contains("out of bounds")),
                "index {out_of_range}: got {err:?}"
            );
        }
    }

    /// A non-integral index value is refused rather than rounded — the shared
    /// `constant_index_values` guard, exercised through `IndexPut` so a future
    /// divergence between the two callers cannot go unnoticed.
    #[test]
    fn index_put_non_integral_index_is_refused() {
        for value in [1.5_f32, f32::from_bits(1.0_f32.to_bits() + 1)] {
            let graph = ComputationGraph::from_nodes(vec![
                node(0, "src", TraceOp::Input, &[], &[1]),
                const_weight(1, "data", vec![0.0, 0.0, 0.0], vec![3]),
                const_weight(2, "index", vec![value], vec![1]),
                node(3, "ip", TraceOp::IndexPut { dim: 0 }, &[1, 2, 0], &[3]),
            ]);
            let err = translate(&graph).expect_err("a non-integral index must fail closed");
            assert!(
                matches!(err, NyError::UnsupportedOp(ref m) if m.contains("not an exact integer")),
                "index {value}: got {err:?}"
            );
        }
    }

    /// nn-core's own preconditions are RE-CHECKED, and the reason is not
    /// pedantry: nn-import's `map_index_put` takes `index_names[0]` and
    /// discards the rest, so PyTorch's `x[i, j] = v` reaches the wire as
    /// `IndexPut { dim: 0 }` carrying only `i`. That trace does not describe
    /// the model — and it also does not satisfy `index_put`'s shape contract,
    /// so the checks turn a silent bound on the wrong function into a refusal.
    #[test]
    fn index_put_enforces_nn_core_shape_preconditions() {
        // (a) A multi-axis advanced write, shaped the way the dropped-index
        //     import leaves it: data [4, 5], index [3], values [3] — src rank
        //     1 against data rank 2.
        let dropped_index = ComputationGraph::from_nodes(vec![
            node(0, "data", TraceOp::Input, &[], &[4, 5]),
            const_weight(1, "index", vec![0.0, 1.0, 2.0], vec![3]),
            const_weight(2, "src", vec![1.0, 2.0, 3.0], vec![3]),
            node(3, "ip", TraceOp::IndexPut { dim: 0 }, &[0, 1, 2], &[4, 5]),
        ]);
        let err = translate(&dropped_index).expect_err("a rank mismatch must fail closed");
        assert!(
            matches!(err, NyError::UnsupportedOp(ref m)
                if m.contains("src rank") && m.contains("advanced index")),
            "got {err:?}"
        );

        // (b) src's write-axis extent must equal the index length.
        let wrong_extent = ComputationGraph::from_nodes(vec![
            node(0, "data", TraceOp::Input, &[], &[4]),
            const_weight(1, "index", vec![0.0, 1.0], vec![2]),
            const_weight(2, "src", vec![1.0, 2.0, 3.0], vec![3]),
            node(3, "ip", TraceOp::IndexPut { dim: 0 }, &[0, 1, 2], &[4]),
        ]);
        let err = translate(&wrong_extent).expect_err("a src/index length mismatch must refuse");
        assert!(
            matches!(err, NyError::UnsupportedOp(ref m) if m.contains("src dim 0 size 3")),
            "got {err:?}"
        );

        // (c) src must match data on every OTHER axis.
        let wrong_other_axis = ComputationGraph::from_nodes(vec![
            node(0, "data", TraceOp::Input, &[], &[4, 5]),
            const_weight(1, "index", vec![0.0], vec![1]),
            const_weight(2, "src", vec![1.0, 2.0], vec![1, 2]),
            node(3, "ip", TraceOp::IndexPut { dim: 0 }, &[0, 1, 2], &[4, 5]),
        ]);
        let err = translate(&wrong_other_axis).expect_err("a non-write-axis mismatch must refuse");
        assert!(
            matches!(err, NyError::UnsupportedOp(ref m) if m.contains("src dim 1 size 2")),
            "got {err:?}"
        );

        // (d) a non-1-D index is not this op.
        let index_2d = ComputationGraph::from_nodes(vec![
            node(0, "data", TraceOp::Input, &[], &[2, 3]),
            const_weight(1, "index", vec![0.0, 1.0], vec![1, 2]),
            const_weight(2, "src", vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
            node(3, "ip", TraceOp::IndexPut { dim: 1 }, &[0, 1, 2], &[2, 3]),
        ]);
        let err = translate(&index_2d).expect_err("a rank-2 index must fail closed");
        assert!(
            matches!(err, NyError::UnsupportedOp(ref m) if m.contains("requires a 1-D index")),
            "got {err:?}"
        );

        // (e) the write axis must exist.
        let bad_dim = ComputationGraph::from_nodes(vec![
            node(0, "data", TraceOp::Input, &[], &[4]),
            const_weight(1, "index", vec![0.0], vec![1]),
            const_weight(2, "src", vec![1.0], vec![1]),
            node(3, "ip", TraceOp::IndexPut { dim: 2 }, &[0, 1, 2], &[4]),
        ]);
        let err = translate(&bad_dim).expect_err("an out-of-range dim must fail closed");
        assert!(
            matches!(err, NyError::UnsupportedOp(ref m) if m.contains("out of range for data rank")),
            "got {err:?}"
        );
    }

    /// TRIPWIRE for the one piece of this lowering that is NOT yet wired, and
    /// that this file cannot wire itself.
    ///
    /// Real nn traces record an index tensor as `U32`, and the graph-wide F32
    /// dtype gate in `translate/mod.rs` only exempts an integer constant leaf
    /// consumed in a known value-independent slot —
    /// `is_value_independent_integer_leaf`, whose list is
    /// `Embedding` position 0 and position 1 of
    /// `IndexSelect`/`Gather`/`RepeatInterleave`/`Scatter`/`ScatterAdd`/`IndexAdd`.
    /// `IndexPut` is missing from it, so a REAL trace is refused by the gate
    /// before reaching the arm above; only an F32-typed index constant (what
    /// the other tests here build) gets through today.
    ///
    /// `IndexPut` earns the same exemption on exactly `Scatter`'s argument —
    /// [`translate_index_put`] resolves the payload at translate time and emits
    /// a synthesized ND-coordinate constant in its place, so no index VALUE
    /// ever becomes a runtime graph edge — but that edit belongs to `mod.rs`.
    /// When it lands, this test starts failing: swap the `expect_err` for an
    /// `expect` and the assertion for the ScatterND check.
    #[test]
    fn index_put_u32_index_leaf_is_still_blocked_by_the_dtype_gate() {
        let mut index_node = const_weight(1, "index", vec![1.0, 2.0], vec![2]);
        index_node.output_dtype = DType::U32;
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "data", TraceOp::Input, &[], &[3]),
            index_node,
            const_weight(2, "src", vec![1.0, 1.0], vec![2]),
            node(3, "ip", TraceOp::IndexPut { dim: 0 }, &[0, 1, 2], &[3]),
        ]);
        let err = translate(&graph).expect_err(
            "EXPECTED FAILURE ONCE mod.rs's is_value_independent_integer_leaf lists IndexPut \
             at position 1 — flip this test to assert the ScatterND lowering instead",
        );
        assert!(
            matches!(err, NyError::UnsupportedOp(ref m)
                if m.contains("output dtype") && m.contains("only soundly supports F32")),
            "the block must be the dtype GATE, not the IndexPut arm: {err:?}"
        );
        // The sibling ops the gate already exempts, for contrast: same graph
        // shape, same U32 index leaf, accepted.
        let mut exempt_index = const_weight(1, "index", vec![1.0, 2.0], vec![2]);
        exempt_index.output_dtype = DType::U32;
        let exempt = ComputationGraph::from_nodes(vec![
            node(0, "data", TraceOp::Input, &[], &[3]),
            exempt_index,
            const_weight(2, "src", vec![1.0, 1.0], vec![2]),
            node(3, "sa", TraceOp::IndexAdd { dim: 0 }, &[0, 1, 2], &[3]),
        ]);
        translate(&exempt).expect("IndexAdd's U32 index leaf is exempt today");
    }

    /// `IndexPut`'s refusal is still classified `Soundness` — the runtime-index
    /// case must stay hard in permissive mode. Lowering the CONSTANT case does
    /// not soften that: the two live in the same arm, and it is the one the
    /// taxonomy has to assume.
    #[test]
    fn index_put_refusal_stays_non_degradable() {
        assert_eq!(
            crate::coverage::refusal_class(&TraceOp::IndexPut { dim: 0 }),
            crate::coverage::RefusalClass::Soundness
        );
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[2]),
            node(1, "index", TraceOp::Relu, &[0], &[2]),
            const_weight(2, "data", vec![0.0, 0.0, 0.0], vec![3]),
            node(3, "ip", TraceOp::IndexPut { dim: 0 }, &[2, 1, 0], &[3]),
        ]);
        let err = super::super::translate_with_options(
            &graph,
            crate::translate::TranslateOptions::permissive(),
        )
        .expect_err("a runtime IndexPut index must fail even in permissive mode");
        assert!(matches!(err, NyError::UnsupportedOp(_)), "got {err:?}");
    }
}
