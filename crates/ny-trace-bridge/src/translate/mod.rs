// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Lower a serialized [`schema::ComputationGraph`] into an `ny_build::GraphModel`.
//!
//! This is the NY-owned port of NN's `nn-verify` `trace_to_graph` translator
//! (~11k lines across 27 files). The dispatch maps each [`schema::TraceOp`] to
//! the corresponding `ny_build` `LayerSpec` / `AttributeValue` / `WeightRef`,
//! accumulates weights and tensor metadata, and assembles a `GraphModel` via
//! [`GraphModelBuilder`]. The translation logic now lives once in NY; framework
//! intake paths serialize their trace to the schema and call [`translate`].
//!
//! ## Sound by construction
//!
//! Every [`schema::TraceOp`] the translator does not yet handle returns
//! [`NyError::UnsupportedOp`] naming the op. The translator never emits a
//! vacuous or passthrough layer for an unrecognized op — an unsupported op is a
//! hard error, not silent best-effort. [`crate::coverage`] makes the supported
//! set explicit at build time.
//!
//! One deliberate, documented exception: [`TraceOp::Custom`] — the trace
//! format's *explicit* opaque escape hatch — lowers to the graph builder's
//! `OpaqueSkip` substitution (`LayerType::Unknown`), whose bounds are the
//! genuinely sound `[-inf, +inf]` over-approximation (never an identity;
//! see ny-propagate `layers/misc/skip_merge.rs`). That is not an
//! unrecognized-op fallback: the op *says* it is opaque, ±inf is the only
//! sound treatment, and refusing would reject whole models over one
//! escape-hatch node. Known-but-unported ops still hard-error.
//!
//! ## Opt-in permissive degradation
//!
//! [`translate_with_options`] / [`translate_multi_input_with_options`] accept
//! [`TranslateOptions::permissive`], which extends the same `OpaqueSkip`
//! substitution to a node whose op simply has no lowering yet, instead of
//! failing the whole graph over one unported op (`ny_build`'s
//! `GraphNetworkOptions::permissive()` does exactly this one layer down). The
//! degradation is deliberately narrow:
//!
//! - STRICT REMAINS THE DEFAULT. [`translate`], [`translate_with_metadata`],
//!   [`translate_multi_input`] and [`translate_segmented`] are unchanged; the
//!   caller must ask.
//! - Only [`crate::coverage::RefusalClass::Unported`] refusals degrade. Ops
//!   refused for SOUNDNESS — data-dependent routing (`Scatter`, `IndexPut`,
//!   `MoeGating`, `GridSample`), unmodeled cast/exponent endpoints (`ToDtype`,
//!   beyond-cap `Powf`) — stay HARD refusals in permissive mode. (`Topk` was
//!   in this list while its `largest`/`sorted` flags were off the wire; they
//!   are serialized now, so its remaining refusals are missing-LAYER refusals
//!   and DO degrade.)
//! - So do the graph-wide dtype gate, the single-variable-input guard, and
//!   every non-`UnsupportedOp` error (numerical-instability gates, malformed
//!   payloads, broken topology).
//! - Every degraded node is reported in [`Translation::skipped_nodes`] with
//!   its trace name, emitted layer name, op and original refusal, counted by
//!   [`Translation::skipped_count`] — vacuous nodes must be impossible to
//!   overlook.
//!
//! ## Dry-run refusal diagnostics
//!
//! [`refusals`] answers the question neither [`crate::coverage::is_translatable`]
//! nor [`translate`] answers: *which nodes of THIS graph would a strict
//! translation refuse, and why* — all of them, not just the first. `coverage`
//! answers "does a sound lowering exist in principle" (so it says `true` for a
//! `Topk` configuration `translate` refuses by name); `translate` answers the
//! right question but stops at the first refused node and reports it only as
//! stringly-typed error text. A caller deciding whether a permissive
//! translation is worth attempting needs the whole list up front. See
//! [`refusals`] for the exact guarantee and its three caveats.
//!
//! ## Conventions mirrored from NN
//!
//! NN's driver hardcodes the unbatched (`is_batched = false`) convention: trace
//! dims are data dims, and axis attributes are emitted TRAILING-RELATIVE
//! (negative) via [`trailing_axis`] — correct in every ny-build conversion
//! regime; see its docs for why the historic pretend-batched `+1` encoding is
//! gone. Node names use the `layer{block}_trace_{id}` form NN relies on for block-wise
//! CROWN, with the block index advancing after each normalization boundary.
//! Output tensors are `{name}_out`; input nodes register a `{name}_in`
//! `TensorSpec` and emit an `Add(input, 0)` identity layer.
//!
//! ## Single-input aliasing and the variable-input guard
//!
//! The graph-build backend maps every input `TensorSpec` to the same network
//! input, so in single-input mode all [`TraceOp::Input`] nodes alias one
//! tensor. That aliasing is UNSOUND for genuinely independent inputs (it can
//! produce a false "holds"), so [`translate`] hard-errors when more than one
//! reachable *variable* input exists — mirroring NN's
//! `validate_single_input_mode` / `MultipleVariableInputs` guard.
//! [`translate_multi_input`] is the sanctioned multi-variable path: it stacks
//! all variable inputs into one flat 1-D tensor and splits it back per
//! variable with generated Slice + Reshape layers.
//!
//! ## Reachable-node filtering
//!
//! Nodes not reachable (backwards over input edges) from the graph's marked
//! output nodes are skipped, mirroring NN. Such dead nodes are primitive ops
//! shadowed by composite ops (e.g. the MatMul/Add a `Linear::forward()` emits
//! before the composite `Linear` node is recorded); an unported op in dead
//! code must not fail an otherwise-translatable graph.
//!
//! ## Dtype-cast modeling
//!
//! A [`TraceOp::ToDtype`] node records only its target dtype, but the source
//! dtype is still recoverable from the wire format: the cast's input tensor is
//! produced by a node whose `output_dtype` field IS the source dtype. The
//! translator records those dtypes alongside tensor shapes and lowers three
//! resolved families: the proven F32 → F32 identity and the lossless
//! U8/Bool → F32 embedding, both as `Add(x, 0)`; and F32 → I32/I64 as the
//! GUARDED round-toward-zero `Trunc` ny-onnx already emits for the identical
//! ONNX `Cast` (`to = 6`/`7`), whose in-range/finite proof obligation every
//! verdict-bearing propagation discharges. Every other resolved combination is
//! refused (the rounding semantics of the non-F32 endpoint are not modeled, so
//! an identity lowering would not enclose the true cast), and a cast whose
//! producer cannot be resolved fails closed.
//! Translation also rejects every reachable node whose declared output dtype
//! is not F32, because the emitted graph contract is F32 — with four narrow
//! carve-outs, each justified by the emitted layers carrying no semantics the
//! integer dtype would add: value-independent integer index/count leaves
//! (Embedding indices, IndexSelect/Gather indices, RepeatInterleave counts —
//! see [`is_value_independent_integer_leaf`]), boolean mask carriers from
//! Compare/CompareTensor (see [`is_boolean_mask_op`]), the exact-f32
//! integer indices Argmax/Argmin/ArgSort emit (see
//! [`is_exact_integer_index_op`]), and the guarded integer cast's own
//! integer-declared output, whose f32 mirror is bit-exact (see
//! [`is_guarded_integer_cast`]).
//!
//! ## Constant folding
//!
//! NN folds unary / binary / matmul ops whose operands are all constant tensors
//! (the graph-build backend filters constant tensors out of activation inputs,
//! so an all-constant op would otherwise build with zero activation inputs).
//! That folding is ported here verbatim in semantics.//!
//! ## Module layout
//!
//! `translate_node` (below) is the single dispatch point. Implemented op arms
//! live in `ops_core`; each not-yet-ported model family owns exactly one stub
//! module (`ops_conv_linear`, `ops_pooling`, `ops_attention`, `ops_lstm`,
//! `ops_kokoro`, `ops_pad_resample`, `ops_misc`) so porting a family only ever
//! touches that one file. Shared state (`Ctx`, `NodeOutput`) and helpers
//! (`weight_f32`, `simple_spec`, `first_input`, `op_name`, ...) live here in
//! `mod.rs` and are visible to every child module via `super::`. `diagnostics`
//! owns the dry-run [`refusals`] surface, which drives the very same node loop
//! in a private mode rather than restating the dispatch.

mod diagnostics;
pub use diagnostics::{refusals, refusals_multi_input, NodeRefusal};

mod ops_attention;
mod ops_conv_linear;
mod ops_core;
mod ops_kokoro;
mod ops_lstm;
mod ops_misc;
pub(crate) use ops_misc::MAX_POW_EXPONENT_ABS;
mod ops_pad_resample;
mod ops_pooling;

use std::collections::{HashMap, HashSet, VecDeque};

use ndarray::{ArrayD, IxDyn};
use ny_build::{AttributeValue, DataType, GraphModel, GraphModelBuilder, LayerSpec, WeightStore};
use ny_core::{checked_shape_product, LayerType, NyError, Result};

use crate::schema::{
    ComputationGraph, DType, NodeId, SegmentedGraph, TraceNode, TraceOp, WeightData, WeightPayload,
};

// Name of the stacked 1-D input tensor emitted by [`translate_multi_input`].
// Matches NN's multi-input producer contract so bounds layouts stay portable.
const MULTI_INPUT_TENSOR: &str = "multi_in";

/// Policy for a node whose op the bridge has no lowering for.
///
/// STRICT ([`Self::Refuse`]) is the default everywhere. Degradation must be
/// asked for explicitly, because it trades a loud failure for a silently
/// vacuous node — see [`TranslateOptions::permissive`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum UnportedOpPolicy {
    /// Any refusal fails the whole translation. The historic behaviour, and
    /// the default.
    #[default]
    Refuse,
    /// A refusal classified [`crate::coverage::RefusalClass::Unported`] emits
    /// the `LayerType::Unknown` carrier — which the graph builder substitutes
    /// with the `[-inf, +inf]` `OpaqueSkip` layer — and records the node in
    /// [`Translation::skipped_nodes`] instead of failing.
    ///
    /// A refusal classified [`crate::coverage::RefusalClass::Soundness`] is
    /// STILL a hard failure under this policy: those ops report that the trace
    /// cannot be read unambiguously, not that a port is missing.
    DegradeToOpaqueSkip,
}

/// Options controlling a translation.
///
/// `TranslateOptions::default()` is strict — identical to [`translate`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TranslateOptions {
    /// What to do with a node whose op has no lowering.
    pub unported_op_policy: UnportedOpPolicy,
}

impl TranslateOptions {
    /// The strict default: any refusal fails the whole translation.
    #[must_use]
    pub const fn strict() -> Self {
        Self {
            unported_op_policy: UnportedOpPolicy::Refuse,
        }
    }

    /// Opt in to degrading UNPORTED ops to the vacuous `OpaqueSkip` carrier.
    ///
    /// Mirrors `ny_build::GraphNetworkOptions::permissive()`, which does the
    /// same thing one layer down for a `LayerSpec` its converter cannot lower.
    /// Soundness-class refusals (data-dependent routing, ambiguous wire
    /// semantics, unmodeled cast rounding) stay hard failures — see
    /// [`UnportedOpPolicy::DegradeToOpaqueSkip`].
    ///
    /// Every degraded node is listed in [`Translation::skipped_nodes`]. Bounds
    /// at a skipped node are `[-inf, +inf]`, so a verification that depends on
    /// one cannot succeed — but a caller that ignores the list can still be
    /// surprised by where the imprecision came from, which is why the list
    /// carries the node names and the original refusal.
    #[must_use]
    pub const fn permissive() -> Self {
        Self {
            unported_op_policy: UnportedOpPolicy::DegradeToOpaqueSkip,
        }
    }

    /// Whether an [`RefusalClass::Unported`](crate::coverage::RefusalClass)
    /// refusal may be degraded under these options.
    const fn degrades_unported(self) -> bool {
        matches!(
            self.unported_op_policy,
            UnportedOpPolicy::DegradeToOpaqueSkip
        )
    }
}

/// One trace node a permissive translation replaced with the vacuous
/// `OpaqueSkip` carrier.
///
/// Deliberately verbose: a vacuous node must be impossible to overlook, so the
/// record carries the trace's own node id and name, the generated layer name
/// that appears in the emitted `GraphModel`, the op, and the refusal the
/// translator would otherwise have returned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkippedNode {
    /// `TraceNode::id` of the degraded node.
    pub node_id: u64,
    /// `TraceNode::name` — the producer's own name for the node.
    pub trace_name: String,
    /// Name of the emitted `LayerType::Unknown` spec (`layer{block}_trace{id}`
    /// form), i.e. how to find the node in [`Translation::model`].
    pub layer_name: String,
    /// Stable display name of the refused [`TraceOp`].
    pub op: &'static str,
    /// The `UnsupportedOp` message the strict translation would have failed
    /// with.
    pub reason: String,
}

/// Result of translating a computation graph: the lowered model plus
/// translation metadata.
///
/// Callers that only need the model can use [`translate`] directly.
#[derive(Debug)]
pub struct Translation {
    /// The lowered `ny_build::GraphModel` producer contract.
    pub model: GraphModel,
    /// Compatibility metadata, always zero for successful translations.
    ///
    /// Precision-changing casts are refused because the bridge has no sound
    /// lowering for them.
    pub dtype_cast_count: usize,
    /// Nodes a permissive translation replaced with the vacuous
    /// `[-inf, +inf]` `OpaqueSkip` carrier, in graph order.
    ///
    /// ALWAYS EMPTY under the strict default — a non-empty list means the
    /// caller asked for [`TranslateOptions::permissive`] and the model now
    /// contains nodes whose bounds carry no information. Check
    /// [`Translation::skipped_count`] before trusting a verdict.
    pub skipped_nodes: Vec<SkippedNode>,
}

impl Translation {
    /// Number of nodes degraded to the vacuous `OpaqueSkip` carrier; `0` for
    /// every strict translation.
    #[must_use]
    pub fn skipped_count(&self) -> usize {
        self.skipped_nodes.len()
    }
}

/// Translate a serialized computation graph into an `ny_build::GraphModel`.
///
/// Mirrors NN's single-input `trace_to_graph_model`: all [`TraceOp::Input`]
/// nodes alias the same network input. The last marked output node (or the last
/// node) becomes the graph output. Nodes unreachable from the marked outputs
/// are skipped.
///
/// Convenience wrapper over [`translate_with_metadata`] for callers that only
/// need the model.
///
/// # Errors
///
/// Returns [`NyError::UnsupportedConfiguration`] when the graph has more than
/// one reachable variable input (single-input aliasing would be unsound — use
/// [`translate_multi_input`]), [`NyError::UnsupportedOp`] for any op outside
/// the supported core set, and [`NyError::InternalError`] /
/// [`NyError::ModelLoad`] for malformed graphs (empty graph, broken topology,
/// dangling input references, non-finite or shape-only weights).
pub fn translate(graph: &ComputationGraph) -> Result<GraphModel> {
    translate_with_metadata(graph).map(|t| t.model)
}

/// Like [`translate`], but also returns compatibility metadata.
///
/// # Errors
///
/// Same conditions as [`translate`].
pub fn translate_with_metadata(graph: &ComputationGraph) -> Result<Translation> {
    translate_with_options(graph, TranslateOptions::strict())
}

/// Single-input translation under explicit [`TranslateOptions`].
///
/// `TranslateOptions::strict()` reproduces [`translate_with_metadata`] exactly.
/// `TranslateOptions::permissive()` additionally degrades UNPORTED ops to the
/// vacuous `OpaqueSkip` carrier, listing each one in
/// [`Translation::skipped_nodes`]; soundness-class refusals, the graph-wide
/// dtype gate, and the single-variable-input guard stay hard failures in both
/// modes.
///
/// # Errors
///
/// Same conditions as [`translate`], minus the unported-op refusals a
/// permissive policy degrades.
pub fn translate_with_options(
    graph: &ComputationGraph,
    options: TranslateOptions,
) -> Result<Translation> {
    let analysis = analyze_graph(graph)?;
    validate_single_input_mode(&analysis)?;
    translate_impl(graph, false, &analysis, options, None)
}

/// Multi-input variant: each distinct reachable variable [`TraceOp::Input`]
/// node gets its own slice of a stacked 1-D input tensor.
///
/// Mirrors NN's `trace_to_graph_model_multi_input`. Use this when the model
/// has genuinely independent input variables. The stacked input is a single
/// `TensorSpec` named `multi_in` of shape `[sum of all variable-input
/// elements]` (flattened in node order); each variable is recovered with a
/// generated Slice + Reshape pair. Callers therefore provide bounds as one
/// flat 1-D tensor. Weight-only `Input` nodes (consumed only as parameters of
/// composite ops) get no slot in the stacked tensor.
///
/// With one or zero variable inputs this degrades to exactly the single-input
/// translation (no stacking, no guard needed).
///
/// # Errors
///
/// Same conditions as [`translate`], minus the variable-input guard.
pub fn translate_multi_input(graph: &ComputationGraph) -> Result<Translation> {
    translate_multi_input_with_options(graph, TranslateOptions::strict())
}

/// Multi-input translation under explicit [`TranslateOptions`].
///
/// The multi-input twin of [`translate_with_options`]; see it for what a
/// permissive policy does and does not degrade.
///
/// # Errors
///
/// Same conditions as [`translate_multi_input`], minus the unported-op
/// refusals a permissive policy degrades.
pub fn translate_multi_input_with_options(
    graph: &ComputationGraph,
    options: TranslateOptions,
) -> Result<Translation> {
    let analysis = analyze_graph(graph)?;
    translate_impl(graph, true, &analysis, options, None)
}

/// Indexed graph facts shared by validation and translation.
struct GraphAnalysis {
    node_indices: HashMap<NodeId, usize>,
    reachable: HashSet<NodeId>,
    variable_inputs: HashSet<NodeId>,
}

/// Validate invariants required by every translation mode at the wire boundary
/// and build the graph indexes used by reachability and consumer analysis.
///
/// `ComputationGraph` is serde-deserializable and its public fields can also be
/// assembled directly, so source-tracer invariants cannot be assumed here.
fn analyze_graph(graph: &ComputationGraph) -> Result<GraphAnalysis> {
    if graph.is_empty() {
        return Err(NyError::InternalError(
            "computation graph is empty (no nodes)".to_string(),
        ));
    }

    let mut node_indices = HashMap::with_capacity(graph.nodes.len());
    for (index, node) in graph.nodes.iter().enumerate() {
        if node_indices.insert(node.id, index).is_some() {
            return Err(NyError::InternalError(format!(
                "duplicate trace node id {} at index {index}",
                node.id.get()
            )));
        }
    }

    graph
        .validate_topology()
        .map_err(|e| NyError::InternalError(format!("topology validation failed: {e}")))?;

    for output_id in &graph.output_nodes {
        if !node_indices.contains_key(output_id) {
            return Err(NyError::InternalError(format!(
                "marked output node {} is not present in the computation graph",
                output_id.get()
            )));
        }
    }

    let reachable = reachable_node_ids(graph, &node_indices);
    for node in graph
        .nodes
        .iter()
        .filter(|node| reachable.contains(&node.id))
    {
        if node.output_dtype != DType::F32
            && !is_value_independent_integer_leaf(node, graph, &reachable)
            && !is_boolean_mask_op(node)
            && !is_exact_integer_index_op(node)
            && !is_guarded_integer_cast(node, graph, &node_indices)
        {
            return Err(NyError::UnsupportedOp(format!(
                "trace node {} ({}) has output dtype {:?}; the bridge only soundly supports F32 tensors",
                node.id.get(),
                node.name,
                node.output_dtype
            )));
        }
    }

    let mut variable_inputs = HashSet::new();
    for node in graph
        .nodes
        .iter()
        .filter(|node| reachable.contains(&node.id))
    {
        let variable_edges = if is_composite_op(&node.op) {
            node.inputs.get(..1).unwrap_or(&[])
        } else {
            node.inputs.as_slice()
        };
        for input_id in variable_edges {
            if node_indices
                .get(input_id)
                .is_some_and(|&index| matches!(&graph.nodes[index].op, TraceOp::Input))
            {
                variable_inputs.insert(*input_id);
            }
        }
    }

    Ok(GraphAnalysis {
        node_indices,
        reachable,
        variable_inputs,
    })
}

/// Compare/CompareTensor nodes whose declared output dtype is a boolean
/// carrier (Bool/U8/I32/I64/U32) pass the F32 gate: the lowering
/// (`e00772c39` — the sound Compare port that closed the kokoro corpus gap)
/// models the output as the exact `{0, 1}` value set with `[0, 1]`-hull IBP
/// regardless of how the tracer STORED the mask, so the dtype label carries
/// no semantics the emitted F32 layers mismodel. Downstream consumers see a
/// sound `[0, 1]` F32 mask — the same values nn's own arithmetic promotes
/// the mask to. Float non-F32 carriers (F16/BF16/F64) stay refused: their
/// rounding story is the general one the gate exists for.
fn is_boolean_mask_op(node: &TraceNode) -> bool {
    matches!(
        node.op,
        TraceOp::Compare { .. } | TraceOp::CompareTensor { .. }
    ) && matches!(
        node.output_dtype,
        DType::Bool | DType::U8 | DType::I32 | DType::I64 | DType::U32
    )
}

/// A `ToDtype` node that [`ops_core::translate_to_dtype`] lowers to the
/// GUARDED round-toward-zero `Trunc` (`F32 -> I32`/`I64`) passes the F32 gate,
/// even though its declared output dtype is an integer.
///
/// # Why the integer label carries nothing the F32 graph mismodels
///
/// The other three carve-outs rest on the VALUE never being read numerically.
/// This one rests on something stronger: the F32 mirror of the cast's result is
/// EXACT. `trunc(x)` of an f32 is itself an f32, and inside the domain the
/// guarded `TruncLayer` certifies (finite and inside `[-2^31, 2^31)` /
/// `[-2^63, 2^63)`) the mathematical integer `x as i64` equals that f32
/// bit-for-bit. So every downstream layer reads precisely the integer the
/// producer computed — no rounding is introduced by the mirror, which is the
/// hazard the graph-wide F32 contract exists to prevent. (See
/// [`ops_core::translate_to_dtype`] for the pointwise agreement with nn's
/// `f32_to_i64` and the exactness of `Trunc`'s IBP.)
///
/// # Why no extra consumer restriction is needed — verified, not assumed
///
/// The question the audit demands an answer to is whether the truncated value
/// can reach downstream f32 arithmetic in a way this gate would otherwise
/// catch. It cannot, and the argument does not depend on enumerating consumers:
///
/// * a consumer that reads the operand as a NUMBER sees exactly the integer
///   the producer computed (the mirror is exact), so its emitted lowering is
///   exactly as correct as it is for any other F32 operand. There is no
///   rounding for the F32 contract to mismodel;
/// * the only semantics that genuinely differ for an integer tensor are
///   INTEGER-SPECIFIC ARITHMETIC — exactness past 2²⁴ where f32 rounds, and
///   wraparound at the type's width. Every op with those semantics is
///   dtype-preserving, so its node declares an integer `output_dtype` and is
///   refused by THIS SAME GATE with no extra rule. The one way back to a
///   declared F32 dtype is a cast, and `ToDtype I64 -> F32` is refused by the
///   cast arm itself (only `U8`/`Bool -> F32` embeds losslessly), so the round
///   trip cannot launder the value either.
///
/// Two families would pass the gate on their own carve-outs, and both are
/// exact on an exact operand rather than merely admissible:
/// `Compare`/`CompareTensor` ([`is_boolean_mask_op`]) compares exact values, so
/// the mask is unchanged; `Argmax`/`Argmin`/`ArgSort`
/// ([`is_exact_integer_index_op`]) take an extremum over exact values, so the
/// index is unchanged.
///
/// The remaining case is an INDEX slot (`Gather`/`IndexSelect` position 1,
/// `Embedding` position 0, the scatter family), which declares the DATA
/// tensor's F32 dtype and so is not caught here — and does not need to be.
/// `Embedding`'s surrogate is index-value-independent by construction;
/// ny-propagate's `GatherLayer` with non-constant indices takes the min/max
/// hull across the gather axis, which encloses every row any index could
/// select; the additive scatter layers' dynamic rule likewise never reads the
/// index. And a `Gather` node fed a RUNTIME index keeps two activation edges,
/// which ny-propagate's dispatch rejects (`require_unary_input`) — an error,
/// never a silent mis-wiring.
///
/// # Legs
///
/// (a) the op is `ToDtype` with a target [`ops_core::guarded_integer_cast_target`]
///     admits (`I32`/`I64`) — the single source of truth shared with the arm
///     that emits the layer, so the two can never drift apart;
/// (b) the node's own declared `output_dtype` IS that target. A trace whose
///     cast node claims some other dtype is internally inconsistent; fail
///     closed rather than trust half of it;
/// (c) the cast has exactly one input and its producer node is present with a
///     declared `output_dtype` of `F32`. The arm resolves the source dtype the
///     same way and refuses anything else, so admitting a non-F32-sourced cast
///     here would only trade a precise refusal for a confusing one; and
/// (d) it is not the graph's effective output (marked, or the last-node
///     fallback when nothing is marked) — the same leg
///     [`is_value_independent_integer_leaf`] carries, and the same rule
///     ny-onnx's `loader::convert` enforces for an integer `Cast` on an
///     authored output ("ny's verifier API exposes FLOAT32 outputs"). The
///     mirror is exact, so this leg costs only coverage; it is kept so the
///     bridge and the ONNX intake path refuse the same shapes.
fn is_guarded_integer_cast(
    node: &TraceNode,
    graph: &ComputationGraph,
    node_indices: &HashMap<NodeId, usize>,
) -> bool {
    // (a) + (b): a cast to a target with a modeled guarded lowering, whose own
    // declared output dtype agrees with that target.
    let TraceOp::ToDtype { target_dtype } = node.op else {
        return false;
    };
    if ops_core::guarded_integer_cast_target(target_dtype).is_none()
        || node.output_dtype != target_dtype
    {
        return false;
    }

    // (c) exactly one input, produced by a node recorded as F32.
    let [source_id] = node.inputs.as_slice() else {
        return false;
    };
    let source_is_f32 = node_indices
        .get(source_id)
        .and_then(|&index| graph.nodes.get(index))
        .is_some_and(|producer| producer.output_dtype == DType::F32);
    if !source_is_f32 {
        return false;
    }

    // (d) not the effective graph output.
    let is_effective_output = if graph.output_nodes.is_empty() {
        graph.nodes.last().is_some_and(|last| last.id == node.id)
    } else {
        graph.output_nodes.contains(&node.id)
    };
    !is_effective_output
}

/// Index-producing selection nodes (`Argmax` / `Argmin` / `ArgSort`) whose
/// declared output dtype is an integer carrier pass the F32 gate.
///
/// Real nn traces record these with a U32 output dtype (the index tensor's
/// storage type), which the graph-wide F32 contract would otherwise refuse.
/// The carve-out is sound because the emitted layers model the indices as
/// *exact f32 integers*: ny-propagate's `reduction/selection.rs` builds each
/// certain index bound with `index_interval`, which returns the degenerate
/// point `(i as f32, i as f32)` whenever the index round-trips exactly and an
/// OUTWARD two-endpoint enclosure `(next_down, rounded)` / `(rounded,
/// next_up)` when it does not — i.e. above 2²⁴, where consecutive integers
/// stop being f32-representable, the interval widens outward rather than
/// snapping to a wrong integer. Undetermined index positions get the full
/// `[0, n-1]` hull. So the integer dtype label carries no semantics the F32
/// graph mismodels: every value the integer tensor could hold is enclosed by
/// the f32 interval the emitted graph propagates.
///
/// `Sort` and `Topk` need no carve-out — both lower to the VALUES tensor,
/// which nn traces as F32. Float non-F32 carriers stay refused here exactly as
/// in [`is_boolean_mask_op`]: their rounding story is the general one the gate
/// exists for.
fn is_exact_integer_index_op(node: &TraceNode) -> bool {
    matches!(
        node.op,
        TraceOp::Argmax { .. } | TraceOp::Argmin { .. } | TraceOp::ArgSort { .. }
    ) && matches!(
        node.output_dtype,
        DType::U8 | DType::I32 | DType::I64 | DType::U32
    )
}

/// Integer-index carve-out for the graph-wide F32 gate: `true` iff this
/// reachable non-F32 node is an integer-typed leaf whose VALUE provably never
/// reaches the emitted graph's arithmetic.
///
/// The Embedding surrogate (zero-weight midpoint Linear + EpsilonExpand
/// half-spread widening; see `ops_conv_linear::translate_embedding`) is
/// index-value-independent: its output interval is the per-dimension
/// vocabulary hull, which encloses every row the lookup could select for ANY
/// index value. The index tensor's F32 stand-in feeds only the surrogate's
/// zeros weight matrix, so no arithmetic in the emitted graph ever consumes
/// the index numerically — which is exactly why the integer dtype (whose
/// rounding/overflow semantics the F32 graph contract cannot model) is
/// irrelevant to soundness here: a value nothing reads cannot round,
/// truncate, or alias anything. The carve-out therefore admits a non-F32
/// node IFF:
///
/// (a) it is a [`TraceOp::Input`] or [`TraceOp::Constant`] node with an
///     integer dtype (I32/I64/U32/U8) — a leaf carrying no modeled
///     arithmetic of its own (float and Bool dtypes stay refused: they are
///     not index types and carry real-valued/logical semantics the contract
///     cannot enclose). [`TraceOp::ConstantWeight`] is admitted for the
///     translate-time-constant index slots below — nn's kokoro sinegen
///     traces capture U32 `index_select` indices exactly that way;
/// (b) every reachable consumer edge referencing it is a value-independent
///     index position: the FIRST input (index operand) of a
///     [`TraceOp::Embedding`] node (whose surrogate never reads index
///     values), or — for Constant/ConstantWeight leaves only — the SECOND
///     input of a [`TraceOp::IndexSelect`] / [`TraceOp::Gather`] (index
///     operand), [`TraceOp::RepeatInterleave`] (per-element counts), or
///     [`TraceOp::Scatter`] / [`TraceOp::ScatterAdd`] / [`TraceOp::IndexAdd`]
///     (scatter destinations — nn records all three as `(self, index, src)`)
///     node, which the translator resolves from the captured weight payload
///     at translate/build time and never lowers into the runtime graph.
///     Unreachable consumers are never translated (dead code is skipped
///     wholesale), so they cannot leak the value; and
/// (c) it is not itself the graph's effective output (marked, or the
///     last-node fallback when no outputs are marked) — an output's value IS
///     consumed, by the caller.
///
/// Every node failing any leg keeps the standing F32 refusal verbatim.
fn is_value_independent_integer_leaf(
    node: &TraceNode,
    graph: &ComputationGraph,
    reachable: &HashSet<NodeId>,
) -> bool {
    // (a) integer-typed leaf. Input/Constant may feed Embedding (its
    // surrogate never reads the index values); Constant/ConstantWeight may
    // additionally feed IndexSelect/Gather index slots, whose values the
    // translator resolves from the weight payload at translate/build time
    // (nn's tracer captures U32 index tensors as f32 weight data — see
    // nn-core `to_weight_ref`) and never lowers into the runtime graph. A
    // runtime `Input` index for Gather stays refused: it is not
    // translate-time-resolvable.
    let integer_dtype = matches!(
        node.output_dtype,
        DType::I32 | DType::I64 | DType::U32 | DType::U8
    );
    let is_leaf = matches!(
        node.op,
        TraceOp::Input | TraceOp::Constant { .. } | TraceOp::ConstantWeight { .. }
    );
    if !integer_dtype || !is_leaf {
        return false;
    }
    let leaf_is_translate_time_constant = matches!(
        node.op,
        TraceOp::Constant { .. } | TraceOp::ConstantWeight { .. }
    );

    // (c) not the effective graph output. Reachability seeds from exactly
    // these nodes, so every reachable node NOT caught here has at least one
    // reachable consumer for (b) to inspect — the all() below is never
    // vacuously true.
    let is_effective_output = if graph.output_nodes.is_empty() {
        graph.nodes.last().is_some_and(|last| last.id == node.id)
    } else {
        graph.output_nodes.contains(&node.id)
    };
    if is_effective_output {
        return false;
    }

    // (b) consumed only in value-independent index positions: Embedding's
    // first input (any admissible leaf), or the second input of
    // IndexSelect/Gather (indices), RepeatInterleave (per-element counts), or
    // Scatter/ScatterAdd/IndexAdd (scatter destinations) when the leaf is a
    // translate-time constant. RepeatInterleave's counts ride the same
    // justification as the Gather index slot:
    // `ops_misc::translate_repeat_interleave` reads them out of the captured
    // weight payload to pick the Slice/Concat (or Tile) decomposition and
    // never emits the counts tensor as a runtime graph input, so their
    // integer dtype is never consumed numerically by the F32 graph.
    //
    // The scatter family's index slot rides it too, and nn puts the index at
    // position 1 for all three (`(self, index, src)` —  nn-core
    // `dyn_tensor/selection/accumulate.rs`). What each lowering does with the
    // captured payload differs, but neither variant lets an index VALUE reach
    // F32 arithmetic:
    //
    //   - `Scatter` — `ops_misc::translate_scatter_overwrite` reads the
    //     destinations at translate time and emits a SYNTHESIZED ND-coordinate
    //     constant in their place; the original index tensor never becomes a
    //     graph edge at all. (A non-constant index is refused outright, so
    //     this arm is the only way a Scatter index reaches translation.)
    //   - `ScatterAdd` / `IndexAdd` — ny-build's converter embeds the resolved
    //     indices into the layer, which then carries no activation edge for
    //     them (`activation_input_count()` drops the slot). Should the constant
    //     ever fail to materialize, the converter refuses rather than leaving
    //     a dangling operand.
    //
    // A RUNTIME (Input) index stays refused for all three, matching the Gather
    // rule. The additive layers do have a sound dynamic-index path that
    // provably never reads the tensor, but admitting an integer runtime input
    // on that basis would widen the carve-out past what is needed here, so the
    // narrower translate-time-constant rule stands.
    graph
        .nodes
        .iter()
        .filter(|consumer| reachable.contains(&consumer.id))
        .all(|consumer| {
            consumer
                .inputs
                .iter()
                .enumerate()
                .filter(|(_, input_id)| **input_id == node.id)
                .all(|(position, _)| match consumer.op {
                    TraceOp::Embedding { .. } => position == 0,
                    TraceOp::IndexSelect { .. }
                    | TraceOp::Gather { .. }
                    | TraceOp::RepeatInterleave { .. }
                    | TraceOp::Scatter { .. }
                    | TraceOp::ScatterAdd { .. }
                    | TraceOp::IndexAdd { .. } => position == 1 && leaf_is_translate_time_constant,
                    _ => false,
                })
        })
}

/// Shared implementation for single-input and multi-input translation — and,
/// via `diagnostics`, for the dry run behind [`refusals`].
///
/// `diagnostics` is the private diagnostic-mode switch, and it is deliberately
/// a parameter of the ONE node loop rather than a second traversal: a parallel
/// classifier over [`TraceOp`] would drift away from the dispatch it claims to
/// predict, which is the exact failure [`refusals`] exists to remove. `None` is
/// every production caller. `Some(sink)` means:
///
/// - every [`NyError::UnsupportedOp`] from [`translate_node`] is recorded in
///   `sink` and the node is replaced with the `LayerType::Unknown` carrier
///   UNCONDITIONALLY — ignoring both the
///   [`RefusalClass::Unported`](crate::coverage::RefusalClass) soundness gate
///   and the [`TranslateOptions::degrades_unported`] policy gate — purely so the
///   walk can continue and find the refusals AFTER this one;
/// - **the resulting [`Translation`] is therefore NOT SOUND and MUST NOT be
///   handed to any caller.** `diagnostics::collect` is the only code that can
///   pass `Some`, it discards the `Translation` unread, and it returns nothing
///   but the `Vec<NodeRefusal>`. That containment — not the carrier's own
///   `[-inf, +inf]` semantics — is what makes bypassing the gates acceptable
///   here. Do not widen `Some(..)` to a public entry point;
/// - any other error class, and any refusal whose carrier cannot be emitted,
///   still returns `Err` and stops the walk. The sink keeps whatever it
///   collected up to that point, because the caller owns it.
fn translate_impl(
    graph: &ComputationGraph,
    enable_multi_input: bool,
    analysis: &GraphAnalysis,
    options: TranslateOptions,
    mut diagnostics: Option<&mut Vec<NodeRefusal>>,
) -> Result<Translation> {
    let reachable = &analysis.reachable;
    let mut skipped_nodes: Vec<SkippedNode> = Vec::new();
    // Batch convention: hardcoded false, matching NN's driver. The +1 axis
    // convention works for all current traces.
    let mut ctx = Ctx::new(false);
    let mut node_names: HashMap<u64, String> = HashMap::new();
    let mut all_layers: Vec<LayerSpec> = Vec::new();
    let mut input_specs: Vec<(String, Vec<i64>)> = Vec::new();
    let mut tensor_producer: HashMap<String, String> = HashMap::new();

    // Multi-input mode: collect the reachable variable Input nodes (weight-only
    // Input nodes get no slot) and stack them into a single 1-D tensor split
    // back per variable with Slice + Reshape. The graph-build backend maps ALL
    // input TensorSpecs to the same network input, so multiple inputs alias one
    // tensor — wrong for independent variables; the stacked layout is how NN
    // keeps them independent.
    let input_node_data: Vec<(NodeId, Vec<usize>)> = if enable_multi_input {
        graph
            .nodes
            .iter()
            .filter(|n| reachable.contains(&n.id) && matches!(n.op, TraceOp::Input))
            .filter(|n| analysis.variable_inputs.contains(&n.id))
            .map(|n| (n.id, n.output_shape.clone()))
            .collect()
    } else {
        Vec::new()
    };
    let multi_input = enable_multi_input && input_node_data.len() > 1;

    // For multi-input: one stacked 1-D input TensorSpec; each variable's
    // elements are flattened and concatenated in node order.
    let multi_input_offsets: HashMap<NodeId, (usize, usize)> = if multi_input {
        let mut offsets = HashMap::new();
        let mut total_flat: usize = 0;
        for (id, shape) in &input_node_data {
            let flat = checked_shape_product(shape).ok_or_else(|| {
                NyError::InternalError(format!(
                    "multi-input: shape product overflows for node {}",
                    id.get()
                ))
            })?;
            offsets.insert(*id, (total_flat, flat));
            total_flat = total_flat.checked_add(flat).ok_or_else(|| {
                NyError::InternalError("multi-input: total flat size overflow".to_string())
            })?;
        }
        let total_i64 = dim_as_i64(total_flat, "multi-input total")?;
        input_specs.push((MULTI_INPUT_TENSOR.to_string(), vec![total_i64]));
        ctx.tensor_shapes
            .insert(MULTI_INPUT_TENSOR.to_string(), vec![total_i64]);
        offsets
    } else {
        HashMap::new()
    };

    // Block index counter for NY block-wise CROWN propagation. Norm ops are
    // block boundaries; the block index increments after each one.
    let mut block_index: usize = 0;

    for node in &graph.nodes {
        if !reachable.contains(&node.id) {
            continue;
        }
        let name = format!("layer{block_index}_trace_{}", node.id.get());

        // Advance the block index after normalization boundaries.
        if is_norm_boundary(&node.op) {
            block_index += 1;
        }

        // Multi-input path: emit Slice + Reshape instead of the identity layer.
        if matches!(node.op, TraceOp::Input) && multi_input {
            let &(offset, flat_size) = multi_input_offsets.get(&node.id).ok_or_else(|| {
                NyError::InternalError("multi-input: node not in offset map".to_string())
            })?;
            let end = offset.checked_add(flat_size).ok_or_else(|| {
                NyError::InternalError("multi-input: offset+size overflow".to_string())
            })?;

            // Slice: extract [offset, offset+flat_size) from the stacked input.
            // Axis 0, VERBATIM: the stacked model's only input is the rank-1
            // `multi_in` tensor, so ny-build's `model_is_unbatched` always
            // classifies it unbatched and converts ONNX axes verbatim (no
            // legacy batch-dim subtraction). The historic `axis=1` encoding
            // relied on the legacy blanket `axis-1` shift and now fails
            // fail-closed ("Slice: axis 1 out of range for 1D tensor").
            let slice_name = format!("{name}_mslice");
            let slice_out = format!("{slice_name}_out");
            let mut slice_attrs = HashMap::new();
            slice_attrs.insert("axis".to_string(), AttributeValue::Int(0));
            slice_attrs.insert(
                "start".to_string(),
                AttributeValue::Int(dim_as_i64(offset, "multi-input offset")?),
            );
            slice_attrs.insert(
                "end".to_string(),
                AttributeValue::Int(dim_as_i64(end, "multi-input end")?),
            );
            all_layers.push(simple_spec(
                &slice_name,
                LayerType::Slice,
                vec![MULTI_INPUT_TENSOR.to_string()],
                &slice_out,
                slice_attrs,
            ));
            ctx.tensor_shapes.insert(
                slice_out.clone(),
                vec![dim_as_i64(flat_size, "multi-input slice out")?],
            );
            tensor_producer.insert(slice_out.clone(), MULTI_INPUT_TENSOR.to_string());

            // Reshape: [flat_size] → the variable's original (unbatched) shape.
            // Attribute-based Reshape does NOT strip a batch dim, so unbatch
            // here (a no-op under the hardcoded unbatched convention).
            let output_tensor = format!("{name}_out");
            let orig_shape_i64 =
                shape_to_i64(ctx.unbatch_shape(&node.output_shape), "multi-input reshape")?;
            let mut reshape_attrs = HashMap::new();
            reshape_attrs.insert(
                "shape".to_string(),
                AttributeValue::Ints(orig_shape_i64.clone()),
            );
            all_layers.push(simple_spec(
                &name,
                LayerType::Reshape,
                vec![slice_out.clone()],
                &output_tensor,
                reshape_attrs,
            ));
            ctx.tensor_shapes
                .entry(output_tensor.clone())
                .or_insert(orig_shape_i64);
            ctx.tensor_dtypes
                .insert(output_tensor.clone(), node.output_dtype);
            tensor_producer.insert(output_tensor, slice_out);

            node_names.insert(node.id.get(), name);
            continue;
        }

        let output = match translate_node(node, &name, &node_names, &mut ctx) {
            Ok(output) => output,
            Err(error) => {
                if let Some(sink) = diagnostics.as_deref_mut() {
                    // DIAGNOSTIC DRY RUN (`refusals`). The model assembled on
                    // this branch is deliberately UNSOUND — the carrier is
                    // emitted for soundness-class refusals too — and is
                    // discarded unread by `diagnostics::collect`, the only
                    // caller that can get here. Nothing built below may escape.
                    if !diagnostics::is_refusal(&error) {
                        // Not a refusal at all: a numerical-instability gate, a
                        // malformed payload or a broken topology. Walking past
                        // one would report refusals derived from a graph that
                        // is already invalid, so stop and keep what we have.
                        return Err(error);
                    }
                    let carrier = opaque_skip_carrier(node, &name, &node_names);
                    sink.push(NodeRefusal {
                        node_id: node.id.get(),
                        trace_name: node.name.clone(),
                        op: op_name(&node.op),
                        reason: error.to_string(),
                        class: crate::coverage::refusal_class(&node.op),
                    });
                    match carrier {
                        Some(specs) => specs,
                        // A refused LEAF, or a dangling input edge: the carrier
                        // cannot stand in for this node, so the walk cannot
                        // continue past it. The refusal itself is already
                        // recorded; the caller's list is simply not exhaustive.
                        None => return Err(error),
                    }
                } else {
                    match degrade_to_opaque_skip(node, &name, &node_names, &error, options) {
                        // Permissive mode, unported op: emit the vacuous carrier
                        // and record the node so the caller cannot overlook it.
                        Some(specs) => {
                            skipped_nodes.push(SkippedNode {
                                node_id: node.id.get(),
                                trace_name: node.name.clone(),
                                layer_name: name.clone(),
                                op: op_name(&node.op),
                                reason: error.to_string(),
                            });
                            specs
                        }
                        // Strict mode, or a refusal that must stay hard in
                        // every mode.
                        None => return Err(error),
                    }
                }
            }
        };

        // Register the input tensor spec for single-input mode.
        if matches!(node.op, TraceOp::Input) {
            let shape = shape_to_i64(&node.output_shape, "Input")?;
            input_specs.push((format!("{name}_in"), shape));
        }

        // Record the output tensor shape and dtype from the traced node.
        let out_shape = shape_to_i64(&node.output_shape, "output shape")?;
        ctx.tensor_shapes.insert(format!("{name}_out"), out_shape);
        ctx.tensor_dtypes
            .insert(format!("{name}_out"), node.output_dtype);

        // tensor_producer: map each spec output to its first input tensor.
        for spec in &output.specs {
            for out_t in &spec.outputs {
                if let Some(first_in) = spec.inputs.first() {
                    tensor_producer.insert(out_t.clone(), first_in.clone());
                }
            }
        }

        all_layers.extend(output.specs);
        node_names.insert(node.id.get(), name);
    }

    // Determine the output tensor spec.
    let output_node = if let Some(output_id) = graph.output_nodes.last() {
        analysis
            .node_indices
            .get(output_id)
            .and_then(|&index| graph.nodes.get(index))
    } else {
        graph.nodes.last()
    }
    .ok_or_else(|| NyError::InternalError("no output node in computation graph".to_string()))?;
    let output_name = node_names.get(&output_node.id.get()).ok_or_else(|| {
        NyError::InternalError("output node not found in translated graph".to_string())
    })?;
    let output_shape = shape_to_i64(&output_node.output_shape, "output node")?;
    let output_tensor = format!("{output_name}_out");

    // Assemble the GraphModel via the builder.
    let mut builder = GraphModelBuilder::new(format!("trace_graph_{}", output_node.name));
    for (in_name, in_shape) in &input_specs {
        builder = builder.input(in_name.clone(), in_shape, DataType::Float32);
    }
    builder = builder.output(output_tensor, &output_shape, DataType::Float32);
    for spec in all_layers {
        builder = builder.layer(spec);
    }
    // Move accumulated weights/metadata into the builder.
    for (wname, arr) in ctx.weights.iter() {
        builder = builder.weight(wname.to_string(), arr.clone());
    }
    for (tname, pname) in &tensor_producer {
        builder = builder.tensor_producer(tname.clone(), pname.clone());
    }
    for cname in &ctx.constant_tensors {
        builder = builder.constant_tensor(cname.clone());
    }
    for (tname, shape) in &ctx.tensor_shapes {
        builder = builder.tensor_shape(tname.clone(), shape);
    }

    Ok(Translation {
        model: builder.build(),
        dtype_cast_count: 0,
        skipped_nodes,
    })
}

/// Decide whether `error` from `node` may be degraded to the vacuous
/// `OpaqueSkip` carrier, and if so build the carrier's specs.
///
/// Returns `None` when the refusal must stay hard — which is the answer for
/// EVERY refusal under the strict default. Under
/// [`UnportedOpPolicy::DegradeToOpaqueSkip`] four conditions must all hold:
///
/// 1. the error is an `UnsupportedOp`. A `NumericalInstability` (the norm
///    epsilon gates, non-finite weights), a `ModelLoad` (a malformed or
///    oversized payload) or an `InternalError` (a broken topology) is not a
///    missing port and is never degraded — silently vacuous-ing a numerically
///    invalid model would hide a real defect in the trace;
/// 2. [`crate::coverage::refusal_class`] classifies the op
///    [`RefusalClass::Unported`](crate::coverage::RefusalClass) — so the
///    data-dependent-routing ops (`Scatter`, `IndexPut`, `MoeGating`,
///    `GridSample`), the unmodeled cast/exponent residues (`ToDtype`,
///    beyond-cap `Powf`) and the pipeline marker `SegmentBoundary` stay hard
///    refusals in permissive mode too;
/// 3. the carrier can actually be emitted for this node — see
///    [`opaque_skip_carrier`] for the two structural preconditions (at least
///    one input, every input edge resolvable) and why each is fail-closed.
///
/// The carrier itself is byte-identical to the [`TraceOp::Custom`] arm's:
/// `LayerType::Unknown` with the op name in `custom_op_name`, which
/// `ny_build`'s graph builder substitutes with `Layer::OpaqueSkip` — IBP
/// `[-inf, +inf]` in the declared output shape, CROWN
/// `LinearBounds::conservative`. That is a genuine over-approximation of ANY
/// function, never an identity, which is what makes opting into it sound at
/// all; see the `Custom` arm in [`translate_node`] for the full argument.
fn degrade_to_opaque_skip(
    node: &TraceNode,
    name: &str,
    node_names: &HashMap<u64, String>,
    error: &NyError,
    options: TranslateOptions,
) -> Option<NodeOutput> {
    if !options.degrades_unported() || !diagnostics::is_refusal(error) {
        return None;
    }
    if !crate::coverage::refusal_class(&node.op).is_degradable() {
        return None;
    }
    opaque_skip_carrier(node, name, node_names)
}

/// Build the `LayerType::Unknown` / `OpaqueSkip` carrier that stands in for
/// `node`, or `None` when the graph's own structure forbids it.
///
/// Split out of [`degrade_to_opaque_skip`] so the STRUCTURAL preconditions are
/// stated once and shared with the [`refusals`] dry run, which deliberately
/// bypasses that function's POLICY gates (see [`translate_impl`]'s
/// `diagnostics` parameter) but cannot bypass these — they are facts about the
/// graph, not choices:
///
/// 1. the node has at least one input. `OpaqueSkip` derives its output shape
///    from its first input edge and ny-propagate's dispatch rejects a
///    zero-input one, so a refused LEAF would build an invalid graph rather
///    than a conservative one — fail closed instead; and
/// 2. every input edge resolves to an already-translated producer. A dangling
///    edge is a malformed graph, not a missing port.
fn opaque_skip_carrier(
    node: &TraceNode,
    name: &str,
    node_names: &HashMap<u64, String>,
) -> Option<NodeOutput> {
    if node.inputs.is_empty() {
        return None;
    }
    let input_tensors: Vec<String> = match node
        .inputs
        .iter()
        .map(|id| resolve_input(id.get(), node_names))
        .collect::<Result<_>>()
    {
        Ok(tensors) => tensors,
        // A dangling input edge is a malformed graph; keep the original
        // refusal rather than papering over it.
        Err(_) => return None,
    };
    let mut attrs = HashMap::new();
    attrs.insert(
        "custom_op_name".to_string(),
        AttributeValue::String(op_name(&node.op).to_string()),
    );
    Some(NodeOutput::one(simple_spec(
        name,
        LayerType::Unknown,
        input_tensors,
        &format!("{name}_out"),
        attrs,
    )))
}

/// Translate each segment of a [`SegmentedGraph`] independently.
///
/// Mirrors NN's `trace_to_graph_segmented`: data-dependent boundary markers
/// split the graph into self-contained sub-graphs, each lowered on its own. The
/// returned models are in segment order; callers compose bounds across segments
/// (output bounds of segment `N` feed segment `N+1`).
///
/// Each segment goes through [`translate`], so the single-input variable-input
/// guard applies per segment (fail closed: a segment whose independent inputs
/// would alias is refused rather than translated unsoundly).
///
/// # Errors
///
/// Propagates any [`translate`] error from any segment.
pub fn translate_segmented(segmented: &SegmentedGraph) -> Result<Vec<GraphModel>> {
    segmented
        .segments
        .iter()
        .map(|segment| translate(&segment.graph))
        .collect()
}

// ---------------------------------------------------------------------------
// Translation context (mirrors NN's TranslateContext)
// ---------------------------------------------------------------------------

/// Accumulated state shared across node translations.
struct Ctx {
    weights: WeightStore,
    constant_tensors: HashSet<String>,
    tensor_shapes: HashMap<String, Vec<i64>>,
    /// Declared output dtype of each translated node's output tensor, keyed
    /// like `tensor_shapes`. Lets dtype-sensitive arms (ToDtype) resolve the
    /// SOURCE dtype of an edge from the producer's recorded `output_dtype`
    /// instead of guessing; fail closed on a missing entry.
    tensor_dtypes: HashMap<String, DType>,
    /// Whether the trace uses the batched convention (batch=1 at dim 0).
    is_batched: bool,
}

impl Ctx {
    fn new(is_batched: bool) -> Self {
        Self {
            weights: WeightStore::new(),
            constant_tensors: HashSet::new(),
            tensor_shapes: HashMap::new(),
            tensor_dtypes: HashMap::new(),
            is_batched,
        }
    }

    /// Strip the leading batch dim from a shape for NY's unbatched convention.
    fn unbatch_shape<'a>(&self, shape: &'a [usize]) -> &'a [usize] {
        if self.is_batched && shape.len() > 1 {
            &shape[1..]
        } else {
            shape
        }
    }

    /// Insert a weight into the store and mark its tensor as constant.
    fn insert_weight(&mut self, tensor_name: &str, data: ArrayD<f32>) -> Result<()> {
        let shape = shape_to_i64(data.shape(), tensor_name)?;
        self.weights.insert(tensor_name.to_string(), data);
        self.constant_tensors.insert(tensor_name.to_string());
        self.tensor_shapes.insert(tensor_name.to_string(), shape);
        Ok(())
    }
}

/// Output of translating a single trace node: zero or more LayerSpecs.
struct NodeOutput {
    specs: Vec<LayerSpec>,
}

impl NodeOutput {
    fn one(spec: LayerSpec) -> Self {
        Self { specs: vec![spec] }
    }

    fn none() -> Self {
        Self { specs: vec![] }
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Convert a `&[usize]` shape to `Vec<i64>` with overflow checks.
fn shape_to_i64(shape: &[usize], context: &str) -> Result<Vec<i64>> {
    shape.iter().map(|&d| dim_as_i64(d, context)).collect()
}

/// Convert `usize` to `i64`, rejecting overflow.
fn dim_as_i64(val: usize, context: &str) -> Result<i64> {
    i64::try_from(val)
        .map_err(|_| NyError::InternalError(format!("{context}: dimension {val} exceeds i64::MAX")))
}

/// Trailing-relative (negative) axis encoding for a trace dim indexing a
/// tensor of rank `rank`: `dim - rank` (e.g. dim 1 of rank 2 → `-1`).
///
/// This is THE axis convention for every emitted axis attribute (mirrors
/// NN's post-rework `TranslateContext::trailing_axis`, nn d7144ea7):
/// ny-build's July-2026 axis-conversion rework (`model_unbatched` verbatim
/// conversion + recorded-rank `remap_axis_trailing`) interprets positive
/// emitted axes against the RECORDED tensor shapes, so the historic
/// pretend-batched `+1` encoding is rejected fail-closed (e.g. "ONNX axis 2
/// out of range for input ... of recorded rank 2") or silently selects the
/// wrong interior dim. Trailing-negative axes are correct in EVERY ny-build
/// conversion regime (unbatched: verbatim, runtime-resolved; batched known
/// rank: range-checked passthrough; unknown rank: passthrough). `rank` is
/// the rank of the tensor the axis indexes per the op's ONNX semantics
/// (e.g. Reduce/Squeeze/Unfold: input rank; Unsqueeze: output rank;
/// rank-preserving ops: output rank).
pub(super) fn trailing_axis(dim: usize, rank: usize, context: &str) -> Result<i64> {
    if dim >= rank {
        return Err(NyError::InternalError(format!(
            "{context}: dim {dim} out of range for rank {rank}"
        )));
    }
    Ok(dim as i64 - rank as i64)
}

/// Checked `f64` → `f32` cast that rejects overflow to infinity.
fn checked_f64_to_f32(val: f64, context: &str) -> Result<f32> {
    if !val.is_finite() {
        return Err(NyError::NumericalInstability(format!(
            "{context}: value is non-finite ({val})"
        )));
    }
    let val_f32 = val as f32;
    if !val_f32.is_finite() {
        return Err(NyError::NumericalInstability(format!(
            "{context}: value {val} overflows f32 (becomes {val_f32})"
        )));
    }
    Ok(val_f32)
}

/// Build a LayerSpec with no weights.
fn simple_spec(
    name: &str,
    layer_type: LayerType,
    inputs: Vec<String>,
    output: &str,
    attrs: HashMap<String, AttributeValue>,
) -> LayerSpec {
    LayerSpec {
        name: name.to_string(),
        layer_type,
        inputs,
        outputs: vec![output.to_string()],
        weights: None,
        attributes: attrs,
    }
}

/// Resolve the input tensor name for a trace-node input id.
///
/// The output tensor name is `{node_name}_out` by convention.
fn resolve_input(input_id: u64, node_names: &HashMap<u64, String>) -> Result<String> {
    let node_name = node_names.get(&input_id).ok_or_else(|| {
        NyError::InternalError(format!(
            "trace node {input_id} not found in translated graph"
        ))
    })?;
    Ok(format!("{node_name}_out"))
}

/// Extract a finite `Vec<f32>` from a [`WeightPayload`], dequantizing as needed.
///
/// Rejects shape-only placeholders and non-finite elements. F16/Bf16 values
/// widen exactly to f32. F64/integer payloads are accepted only when every
/// element round-trips through f32 exactly; silently rounding a deployed
/// parameter would verify a different model.
fn weight_f32(payload: &WeightPayload, context: &str) -> Result<Vec<f32>> {
    let data: Vec<f32> = match &payload.data {
        WeightData::F32(v) => v.clone(),
        WeightData::F64(v) => v
            .iter()
            .enumerate()
            .map(|(index, &value)| {
                if !value.is_finite() {
                    return Err(NyError::NumericalInstability(format!(
                        "{context}: F64 weight contains non-finite value ({value}) at index {index}"
                    )));
                }
                let narrowed = value as f32;
                if narrowed as f64 != value {
                    return Err(NyError::ModelLoad(format!(
                        "{context}: F64 weight value {value} at index {index} is not exactly representable as f32"
                    )));
                }
                Ok(narrowed)
            })
            .collect::<Result<Vec<_>>>()?,
        WeightData::F16(v) => v.iter().map(|&x| x.to_f32()).collect(),
        WeightData::Bf16(v) => v.iter().map(|&x| x.to_f32()).collect(),
        WeightData::I32(v) => v
            .iter()
            .enumerate()
            .map(|(index, &value)| {
                let narrowed = value as f32;
                if narrowed as i64 != i64::from(value) {
                    return Err(NyError::ModelLoad(format!(
                        "{context}: I32 weight value {value} at index {index} is not exactly representable as f32"
                    )));
                }
                Ok(narrowed)
            })
            .collect::<Result<Vec<_>>>()?,
        WeightData::I64(v) => v
            .iter()
            .enumerate()
            .map(|(index, &value)| {
                let narrowed = value as f32;
                if narrowed as i128 != i128::from(value) {
                    return Err(NyError::ModelLoad(format!(
                        "{context}: I64 weight value {value} at index {index} is not exactly representable as f32"
                    )));
                }
                Ok(narrowed)
            })
            .collect::<Result<Vec<_>>>()?,
        WeightData::Placeholder => {
            return Err(NyError::ModelLoad(format!(
                "{context}: weight data is shape-only (placeholder). \
                 Use CPU tracing to capture weight data."
            )));
        }
    };
    if data.is_empty() {
        return Err(NyError::ModelLoad(format!(
            "{context}: weight data is empty (shape-only capture)."
        )));
    }
    for (i, &val) in data.iter().enumerate() {
        if !val.is_finite() {
            return Err(NyError::NumericalInstability(format!(
                "{context}: weight contains non-finite value ({val}) at index {i}"
            )));
        }
    }
    Ok(data)
}

/// Convert a [`WeightPayload`] to an `ArrayD<f32>` and insert it into the store.
fn insert_payload(
    ctx: &mut Ctx,
    payload: &WeightPayload,
    tensor_name: &str,
    context: &str,
) -> Result<()> {
    let data = weight_f32(payload, context)?;
    let arr = ArrayD::from_shape_vec(IxDyn(&payload.shape), data)
        .map_err(|e| NyError::ModelLoad(format!("{context}: shape mismatch: {e}")))?;
    ctx.insert_weight(tensor_name, arr)
}

/// Insert a finite scalar constant weight tensor.
fn insert_scalar_constant(ctx: &mut Ctx, tensor_name: &str, value: f32) -> Result<()> {
    if !value.is_finite() {
        return Err(NyError::NumericalInstability(format!(
            "scalar constant non-finite ({value})"
        )));
    }
    ctx.insert_weight(tensor_name, ArrayD::from_elem(IxDyn(&[]), value))
}

/// Returns `true` for ops that act as block boundaries (normalizations).
fn is_norm_boundary(op: &TraceOp) -> bool {
    matches!(
        op,
        TraceOp::LayerNorm { .. }
            | TraceOp::RmsNorm { .. }
            | TraceOp::GroupNorm { .. }
            | TraceOp::InstanceNorm { .. }
            | TraceOp::BatchNorm { .. }
    )
}

// ---------------------------------------------------------------------------
// Graph-analysis predicates (mirror NN's trace_to_graph_predicates)
// ---------------------------------------------------------------------------

/// Reject graphs with multiple reachable variable inputs in single-input mode.
///
/// Single-input mode aliases ALL [`TraceOp::Input`] nodes to the same network
/// input, which produces unsound bounds (possible false "holds") when the
/// inputs are genuinely independent. Mirrors NN's `validate_single_input_mode`
/// / `MultipleVariableInputs` hard error.
fn validate_single_input_mode(analysis: &GraphAnalysis) -> Result<()> {
    let variable_input_count = analysis.variable_inputs.len();
    if variable_input_count > 1 {
        return Err(NyError::UnsupportedConfiguration(format!(
            "translate() found {variable_input_count} variable inputs but expects exactly 1; \
             use translate_multi_input() for multi-input models"
        )));
    }
    Ok(())
}

/// Compute the set of node IDs reachable from the graph's marked outputs.
///
/// Walks backward over input edges using BFS. Nodes not reachable from any
/// output are primitive ops shadowed by composite ops (e.g. MatMul/Add nodes a
/// framework's `Linear::forward()` emits before the composite `Linear` node is
/// recorded) and must be skipped, not translated. NN seeds from its single
/// primary output; the schema carries a list of marked outputs, so every entry
/// seeds the walk (a superset — keeping more nodes can only surface more
/// errors, never weaken bounds). With no marked outputs, the last node seeds,
/// matching the output-selection fallback in [`translate`].
fn reachable_node_ids(
    graph: &ComputationGraph,
    node_indices: &HashMap<NodeId, usize>,
) -> HashSet<NodeId> {
    let mut reachable: HashSet<NodeId> = HashSet::new();
    let mut queue: VecDeque<NodeId> = VecDeque::new();

    let seeds: Vec<NodeId> = if graph.output_nodes.is_empty() {
        graph.nodes.last().map(|n| n.id).into_iter().collect()
    } else {
        graph.output_nodes.clone()
    };
    for seed in seeds {
        if reachable.insert(seed) {
            queue.push_back(seed);
        }
    }

    while let Some(id) = queue.pop_front() {
        if let Some(node) = node_indices
            .get(&id)
            .and_then(|&index| graph.nodes.get(index))
        {
            for &input_id in &node.inputs {
                if reachable.insert(input_id) {
                    queue.push_back(input_id);
                }
            }
        }
    }

    reachable
}

/// Returns `true` for composite ops that embed weight tensors in their
/// [`TraceOp`] variant, so only `inputs[0]` is the data input.
fn is_composite_op(op: &TraceOp) -> bool {
    matches!(
        op,
        TraceOp::Conv1d { .. }
            | TraceOp::Conv2d { .. }
            | TraceOp::Conv3d { .. }
            | TraceOp::ConvTranspose1d { .. }
            | TraceOp::ConvTranspose2d { .. }
            | TraceOp::Linear { .. }
            | TraceOp::QLinear { .. }
            | TraceOp::Embedding { .. }
            | TraceOp::Lstm { .. }
            | TraceOp::BatchNorm { .. }
            | TraceOp::LayerNorm { .. }
            | TraceOp::RmsNorm { .. }
            | TraceOp::InstanceNorm { .. }
            | TraceOp::GroupNorm { .. }
    )
}

/// Smallest normalization epsilon any NY norm layer accepts.
///
/// MIRROR of `ny_propagate::layers::NORMALIZATION_MIN_EPS`. ny-propagate is a
/// dev-dependency of this crate (the runtime bridge needs only ny-build), so
/// the value is restated here rather than imported; the unit test
/// `eps_gate_mirrors_ny_propagate_floor` asserts the two agree bit-for-bit
/// and fails if ny-propagate ever moves its floor.
pub(super) const NORMALIZATION_MIN_EPS: f32 = 1e-12;

/// Epsilon gate for the DATA-VARIANCE normalizations — LayerNorm, RmsNorm,
/// InstanceNorm, GroupNorm, and the AdaIN/AdaLayerNorm/FusedAdainResBlock
/// forms in `ops_kokoro`.
///
/// # The three gates this reconciles
///
/// | layer | rule before | why it was wrong |
/// | --- | --- | --- |
/// | nn (`nn-core/src/layers` `validate_eps`) | finite, `eps >= 0` | authoritative for what a real trace can CARRY, so `eps == 0` reaches the bridge |
/// | this bridge | finite, `eps > 0` | admitted `0 < eps < 1e-12`, which every ny norm layer then refuses |
/// | ny-propagate (`validate_norm_eps`) / ny-build (`normalization_fusion`) | finite, `eps >= NORMALIZATION_MIN_EPS` | the gate the bound derivations are actually validated at |
///
/// The old bridge rule was the odd one out in BOTH directions: it refused
/// `eps == 0` (fine) but admitted the `(0, 1e-12)` window, so a trace in that
/// window translated "successfully" and then failed deep inside
/// `GraphModel::build_graph_network` — a refusal at the wrong boundary, with
/// a message about a converted layer rather than the traced op. Aligning to
/// the propagate floor moves that refusal to translate time.
///
/// # Why the floor, and not merely `eps > 0`
///
/// Read the divisions. Every layer in this family divides by
/// `sqrt(var + eps)` where `var` is computed FROM THE DATA
/// (`rms_norm/math.rs`: `let rms = (mean_sq + eps).sqrt()` then `x_i / rms`;
/// LayerNorm/InstanceNorm/GroupNorm/AdaIN the same shape). An input box is
/// free to contain a constant vector, for which `var == 0` exactly — so at
/// `eps == 0` the represented function is `0/0` at a reachable point and no
/// enclosure of it exists. That rules out `eps == 0` outright. Below
/// `NORMALIZATION_MIN_EPS` the denominator is positive but the layers'
/// derivations are not validated there (`RmsNormLayer`'s L2 sphere argument,
/// the CROWN slope bounds, the f32 fused paths), and ny-propagate refuses
/// rather than silently rounding eps upward, because changing eps changes the
/// represented function and can flip a verdict at a tight property boundary.
/// So the weakest gate that is sound for this family end-to-end IS the
/// propagate floor — restated here so the refusal happens at the trace
/// boundary that can name the offending op.
///
/// BatchNorm does NOT belong to this family: its variance is a translate-time
/// constant, so `eps == 0` is admissible exactly when every channel's
/// `var + eps` is strictly positive. See
/// [`validate_constant_variance_eps`].
fn validate_eps(eps: f64, context: &str) -> Result<f32> {
    if !eps.is_finite() || eps <= 0.0 {
        return Err(NyError::NumericalInstability(format!(
            "{context}: eps must be finite and at least {NORMALIZATION_MIN_EPS} (the floor \
             ny-propagate's normalization bound derivations are validated at), got {eps} — \
             this family divides by sqrt(var + eps) with a DATA-dependent var, so a box \
             containing a constant vector makes the denominator sqrt(eps)"
        )));
    }
    let eps_f32 = eps as f32;
    if !eps_f32.is_finite() || eps_f32 < NORMALIZATION_MIN_EPS {
        return Err(NyError::NumericalInstability(format!(
            "{context}: eps {eps} narrows to {eps_f32}, below the normalization floor \
             {NORMALIZATION_MIN_EPS} that every ny norm layer enforces — refused here, at the \
             traced op, rather than deep in graph construction"
        )));
    }
    Ok(eps_f32)
}

/// Epsilon gate for BatchNorm — the one normalization whose variance is a
/// TRANSLATE-TIME CONSTANT.
///
/// BatchNorm's denominator is `sqrt(running_var + eps)` with `running_var`
/// a captured weight, so unlike the data-variance family there is nothing
/// data-dependent that can drive it to zero: `eps == 0` is perfectly sound
/// whenever every channel already has `running_var > 0`. nn accepts `eps == 0`
/// (`nn-core` `validate_eps`: finite and non-negative) and so does
/// ny-propagate's `BatchNormLayer::new`, which enforces exactly
///
/// ```text
/// let variance_sum_f32 = variance + epsilon;
/// if !variance_sum_f32.is_finite() || variance_sum_f32 <= 0.0 { refuse }
/// ```
///
/// per channel. That is the WEAKEST SOUND GATE for this layer, and this
/// function is its verbatim translate-time twin: same f32 addition, same
/// finite-and-strictly-positive test, so a trace that ny-propagate would
/// accept is not refused here and one it would reject is refused HERE, naming
/// the offending channel and the traced op instead of a converted layer.
///
/// Not replicated: `BatchNormLayer::new`'s finiteness checks on the derived
/// `scale`/`bias`. Those are weight-magnitude properties, not epsilon
/// properties, and remain ny-build's to enforce.
fn validate_constant_variance_eps(eps: f64, running_var: &[f32], context: &str) -> Result<f32> {
    if !eps.is_finite() || eps < 0.0 {
        return Err(NyError::NumericalInstability(format!(
            "{context}: eps must be finite and non-negative, got {eps}"
        )));
    }
    let eps_f32 = eps as f32;
    if !eps_f32.is_finite() {
        return Err(NyError::NumericalInstability(format!(
            "{context}: eps {eps} overflows f32 (becomes {eps_f32})"
        )));
    }
    for (channel, &variance) in running_var.iter().enumerate() {
        // Exactly BatchNormLayer::new's f32 test, channel by channel.
        let variance_sum = variance + eps_f32;
        if !variance_sum.is_finite() || variance_sum <= 0.0 {
            return Err(NyError::NumericalInstability(format!(
                "{context}: channel {channel} has running_var {variance} and eps {eps_f32}, \
                 whose sum {variance_sum} is not finite and strictly positive — the \
                 denominator sqrt(var + eps) is undefined or unbounded there"
            )));
        }
    }
    Ok(eps_f32)
}

// ---------------------------------------------------------------------------
// Per-node dispatch
// ---------------------------------------------------------------------------

/// Translate a single trace node into one or more LayerSpecs.
fn translate_node(
    node: &TraceNode,
    name: &str,
    node_names: &HashMap<u64, String>,
    ctx: &mut Ctx,
) -> Result<NodeOutput> {
    let op = &node.op;
    let output_shape = &node.output_shape;
    let output_tensor = format!("{name}_out");

    // Resolve input tensor names.
    let input_tensors: Vec<String> = node
        .inputs
        .iter()
        .map(|id| resolve_input(id.get(), node_names))
        .collect::<Result<_>>()?;
    match op {
        TraceOp::Input => ops_core::translate_input(name, &output_tensor, ctx),
        TraceOp::Constant { value } => {
            ops_core::translate_constant(*value, output_shape, &output_tensor, ctx)
        }
        TraceOp::ConstantWeight { weight } => {
            ops_core::translate_constant_weight(weight, &output_tensor, ctx)
        }

        // -- Unary activations (constant-fold when sole input is constant) --
        TraceOp::Relu
        | TraceOp::Sigmoid
        | TraceOp::Tanh
        | TraceOp::Exp
        | TraceOp::Silu
        | TraceOp::Sqrt
        | TraceOp::Abs
        | TraceOp::Recip
        | TraceOp::Log
        | TraceOp::Sin
        | TraceOp::Cos
        | TraceOp::Floor
        | TraceOp::Round
        | TraceOp::Neg
        | TraceOp::Softplus
        | TraceOp::Tan
        | TraceOp::Ceil
        | TraceOp::Sign
        | TraceOp::Mish
        | TraceOp::HardSigmoid
        | TraceOp::HardSwish
        | TraceOp::Selu
        | TraceOp::Softsign => {
            if let Some(result) =
                ops_core::try_constant_fold_unary(op, &output_tensor, &input_tensors, ctx)
            {
                return result;
            }
            ops_core::translate_unary_activation(op, name, &input_tensors, &output_tensor)
        }

        // -- GELU variants --
        TraceOp::Gelu => Ok(ops_core::translate_gelu(
            name,
            &input_tensors,
            &output_tensor,
            "tanh",
        )),
        TraceOp::GeluErf => Ok(ops_core::translate_gelu(
            name,
            &input_tensors,
            &output_tensor,
            "none",
        )),

        // -- Sqr: Pow(2) --
        TraceOp::Sqr => ops_core::translate_sqr(name, &input_tensors, &output_tensor, ctx),

        // -- Binary elementwise (constant-fold when both inputs constant) --
        TraceOp::Add | TraceOp::Mul | TraceOp::Div | TraceOp::Maximum | TraceOp::Minimum => {
            if let Some(result) =
                ops_core::try_constant_fold_binary(op, &output_tensor, &input_tensors, ctx)
            {
                return result;
            }
            ops_core::translate_binary(op, name, &input_tensors, &output_tensor)
        }
        TraceOp::Sub => {
            if let Some(result) =
                ops_core::try_constant_fold_binary(op, &output_tensor, &input_tensors, ctx)
            {
                return result;
            }
            ops_core::translate_sub(name, &input_tensors, &output_tensor)
        }

        TraceOp::MatMul => {
            if input_tensors.len() != 2 {
                return Err(NyError::UnsupportedOp(format!(
                    "MatMul requires exactly 2 inputs, got {}",
                    input_tensors.len()
                )));
            }
            if let Some(result) =
                ops_core::try_constant_fold_matmul(&output_tensor, &input_tensors, ctx)
            {
                return result;
            }
            // build_graph_network picks the right layer: both-variable inputs
            // become a bilinear (McCormick) layer, one-constant becomes Linear.
            Ok(NodeOutput::one(simple_spec(
                name,
                LayerType::MatMul,
                input_tensors,
                &output_tensor,
                HashMap::new(),
            )))
        }

        // -- Linear --
        TraceOp::Linear { weight, bias } => {
            ops_core::translate_linear(name, weight, bias, &input_tensors, &output_tensor, ctx)
        }

        // -- Convolutions --
        TraceOp::Conv1d {
            weight,
            bias,
            padding,
            stride,
            dilation,
            groups,
        } => ops_core::translate_conv1d(
            name,
            weight,
            bias,
            *padding,
            *stride,
            *dilation,
            *groups,
            &input_tensors,
            &output_tensor,
            ctx,
        ),
        TraceOp::Conv2d {
            weight,
            bias,
            padding,
            stride,
            dilation,
            groups,
        } => ops_core::translate_conv2d(
            name,
            weight,
            bias,
            padding,
            stride,
            dilation,
            *groups,
            &input_tensors,
            &output_tensor,
            ctx,
        ),

        // -- Reductions --
        TraceOp::ReduceSum { .. }
        | TraceOp::ReduceMean { .. }
        | TraceOp::ReduceMax { .. }
        | TraceOp::ReduceMin { .. } => {
            ops_core::translate_reduce(op, name, &input_tensors, &output_tensor, ctx)
        }

        // -- Shape ops --
        TraceOp::Reshape { target_shape } => {
            let unbatched = ctx.unbatch_shape(target_shape).to_vec();
            ops_core::translate_reshape(name, &unbatched, input_tensors, &output_tensor)
        }
        TraceOp::Transpose { dim0, dim1 } => ops_core::translate_transpose(
            name,
            *dim0,
            *dim1,
            output_shape,
            input_tensors,
            &output_tensor,
        ),
        // Axis values are encoded TRAILING-RELATIVE (negative) via
        // trailing_axis(dim, rank): ny-build passes negative axes through
        // every conversion regime and the runtime layers resolve them against
        // the actual tensor rank. Mirrors NN's post-rework dispatch (nn
        // d7144ea7). ONNX Unsqueeze axes index the OUTPUT rank; Squeeze axes
        // index the INPUT rank (output rank + 1); Cat preserves rank.
        TraceOp::Unsqueeze { dim } => ops_core::translate_unsqueeze(
            name,
            trailing_axis(*dim, output_shape.len(), "Unsqueeze axis")?,
            input_tensors,
            &output_tensor,
        ),
        TraceOp::Squeeze { dim } => ops_core::translate_squeeze(
            name,
            trailing_axis(*dim, output_shape.len() + 1, "Squeeze axis")?,
            input_tensors,
            &output_tensor,
        ),
        TraceOp::Permute { axes } => {
            ops_core::translate_permute(name, axes, input_tensors, &output_tensor)
        }
        TraceOp::Cat { dim, .. } => ops_core::translate_cat(
            name,
            trailing_axis(*dim, output_shape.len(), "Cat axis")?,
            input_tensors,
            &output_tensor,
        ),

        // -- Softmax / LogSoftmax (raw dim; backend adjusts axis) --
        TraceOp::Softmax { dim } => ops_core::translate_softmax(
            name,
            LayerType::Softmax,
            *dim,
            input_tensors,
            &output_tensor,
        ),
        TraceOp::LogSoftmax { dim } => ops_core::translate_softmax(
            name,
            LayerType::LogSoftmax,
            *dim,
            input_tensors,
            &output_tensor,
        ),

        // -- Normalization --
        TraceOp::LayerNorm { eps, weight, bias } => ops_core::translate_layer_norm(
            name,
            *eps,
            weight,
            bias,
            &input_tensors,
            &output_tensor,
            ctx,
        ),
        TraceOp::RmsNorm { eps, weight } => {
            ops_core::translate_rms_norm(name, *eps, weight, &input_tensors, &output_tensor, ctx)
        }
        TraceOp::InstanceNorm { eps } => ops_core::translate_instance_norm(
            name,
            *eps,
            &input_tensors,
            &output_tensor,
            output_shape,
            ctx,
        ),
        TraceOp::BatchNorm {
            eps,
            weight,
            bias,
            running_mean,
            running_var,
        } => ops_core::translate_batch_norm(
            name,
            *eps,
            weight,
            bias,
            running_mean,
            running_var,
            &input_tensors,
            &output_tensor,
            ctx,
        ),
        TraceOp::GroupNorm {
            num_groups,
            eps,
            weight,
            bias,
        } => ops_core::translate_group_norm(
            name,
            *num_groups,
            *eps,
            weight,
            bias,
            &input_tensors,
            &output_tensor,
            output_shape,
            ctx,
        ),

        // -- Clamp / Clip --
        TraceOp::Clamp { min, max } => {
            ops_core::translate_clamp(name, *min, *max, input_tensors, &output_tensor)
        }

        // -- Dropout (identity at inference) --
        TraceOp::Dropout => ops_core::translate_dropout(name, &input_tensors, &output_tensor, ctx),

        // -- Dtype cast (source dtype resolved from the producer's recorded
        //    output dtype; only the exact F32 -> F32 identity lowers) --
        TraceOp::ToDtype { target_dtype } => {
            ops_core::translate_to_dtype(name, *target_dtype, &input_tensors, &output_tensor, ctx)
        }

        // -- Parameterized activations --
        TraceOp::Elu { alpha } => {
            ops_core::translate_elu(name, *alpha, input_tensors, &output_tensor)
        }
        TraceOp::Celu { alpha } => {
            ops_core::translate_celu(name, *alpha, input_tensors, &output_tensor)
        }
        TraceOp::LeakyRelu { slope } => {
            ops_core::translate_leaky_relu(name, *slope, &input_tensors, &output_tensor, ctx)
        }
        TraceOp::PRelu { slope } => {
            ops_core::translate_prelu(name, slope, &input_tensors, &output_tensor, ctx)
        }

        // -- Named activation fallback --
        //
        // Mish note (reconciled at INC-FINAL): the bridge accepts
        // `Activation { kind: Mish }`, lowering it to the same unmodified
        // LayerType::Mish emission used for the dedicated `TraceOp::Mish`
        // variant. Verified sound: ny-propagate's `MishLayer` is a first-class
        // sound relaxation (critical-point-aware IBP for the non-monotonic
        // minimum, directed rounding on every intermediate, f64 chord slopes
        // with slope-error widening). Unlike the Elu/LeakyRelu named-path
        // refusals (which exist because a PARAMETER would be silently
        // defaulted), Mish is parameterless, so nothing can be lost through
        // the named path. NN's direct `translate_named_activation` was taught
        // the same Mish arm at INC-FINAL, so both translators agree;
        // Elu/LeakyRelu remain refused on both.
        TraceOp::Activation { kind } => {
            ops_core::translate_named_activation(name, *kind, &input_tensors, &output_tensor)
        }

        // ------------------------------------------------------------------
        // Pre-wired family dispatch. Each arm routes to the module that owns
        // the op's model family. The family fns are unimplemented stubs today
        // and return the same sound-by-construction UnsupportedOp refusal the
        // old catch-all arm produced; porting an op touches only its module.
        // ------------------------------------------------------------------

        // -- Extended conv / linear / embedding family --
        TraceOp::Conv3d { .. }
        | TraceOp::ConvTranspose1d { .. }
        | TraceOp::ConvTranspose2d { .. }
        | TraceOp::QLinear { .. }
        | TraceOp::Embedding { .. }
        | TraceOp::Narrow { .. }
        | TraceOp::Expand { .. } => ops_conv_linear::translate_conv_linear(
            node,
            name,
            &input_tensors,
            &output_tensor,
            node_names,
            ctx,
        ),

        // -- Pooling family --
        TraceOp::AvgPool1d { .. }
        | TraceOp::AvgPool2d { .. }
        | TraceOp::MaxPool1d { .. }
        | TraceOp::MaxPool2d { .. }
        | TraceOp::AdaptiveAvgPool1d { .. }
        | TraceOp::AdaptiveAvgPool2d { .. }
        | TraceOp::AdaptiveMaxPool2d { .. } => ops_pooling::translate_pooling(
            node,
            name,
            &input_tensors,
            &output_tensor,
            node_names,
            ctx,
        ),

        // -- Attention family --
        //
        // `MultiHeadAttention` routes here too: the family module owns the
        // refusal so the precise wire-format-gap message (the fused op
        // serializes no projection weights) surfaces end-to-end instead of the
        // generic catch-all text.
        TraceOp::Sdpa { .. }
        | TraceOp::SdpaCausal { .. }
        | TraceOp::RotaryEmbedding { .. }
        | TraceOp::MultiHeadAttention { .. } => ops_attention::translate_attention(
            node,
            name,
            &input_tensors,
            &output_tensor,
            node_names,
            ctx,
        ),

        // -- Recurrent (LSTM) family --
        TraceOp::Lstm { .. } => {
            ops_lstm::translate_lstm(node, name, &input_tensors, &output_tensor, node_names, ctx)
        }

        // -- Kokoro fused-op family --
        TraceOp::KokoroFused(_) => ops_kokoro::translate_kokoro(
            node,
            name,
            &input_tensors,
            &output_tensor,
            node_names,
            ctx,
        ),

        // -- Padding / resampling family --
        TraceOp::ReflectionPad1d { .. }
        | TraceOp::ReflectionPad2d { .. }
        | TraceOp::ConstantPadNd { .. }
        | TraceOp::PixelShuffle { .. }
        | TraceOp::PixelUnshuffle { .. }
        | TraceOp::Upsample1d { .. }
        | TraceOp::Upsample2d { .. }
        | TraceOp::ResizeBilinear { .. }
        | TraceOp::GridSample { .. } => ops_pad_resample::translate_pad_resample(
            node,
            name,
            &input_tensors,
            &output_tensor,
            node_names,
            ctx,
        ),

        // -- Misc elementwise / indexing / masking / selection family --
        //
        // The selection ops (`Argmax` / `Argmin` / `ArgSort` / `Sort` / `Topk`)
        // route here as well: `ops_misc` lowers all five. `Topk` lowers for the
        // one `largest`/`sorted` combination `TopkLayer` models — the flags
        // ride the wire now — and owns the refusals of the other three, whose
        // true reason only that module can state.
        TraceOp::SwiGlu
        | TraceOp::Powf { .. }
        | TraceOp::Fract
        | TraceOp::Atan2
        | TraceOp::Cumsum { .. }
        | TraceOp::Flip { .. }
        | TraceOp::Roll { .. }
        | TraceOp::RepeatInterleave { .. }
        | TraceOp::Arange { .. }
        | TraceOp::Triu { .. }
        | TraceOp::Tril { .. }
        | TraceOp::SliceSet { .. }
        | TraceOp::Unfold { .. }
        | TraceOp::IndexSelect { .. }
        | TraceOp::Gather { .. }
        | TraceOp::Compare { .. }
        | TraceOp::CompareTensor { .. }
        | TraceOp::WhereCond
        | TraceOp::Argmax { .. }
        | TraceOp::Argmin { .. }
        | TraceOp::ArgSort { .. }
        | TraceOp::Sort { .. }
        | TraceOp::Topk { .. }
        | TraceOp::Scatter { .. }
        | TraceOp::ScatterAdd { .. }
        | TraceOp::IndexAdd { .. }
        | TraceOp::IndexPut { .. }
        | TraceOp::MoeGating { .. } => {
            ops_misc::translate_misc(node, name, &input_tensors, &output_tensor, node_names, ctx)
        }

        // -- Custom: sound conservative OpaqueSkip mirror (NN #4349) --
        //
        // INC-FINAL reconciliation decision (investigated from code, not from
        // the layer's name): ny-propagate's `OpaqueSkipLayer`
        // (`layers/misc/skip_merge.rs`) is NOT an identity passthrough — its
        // IBP rule returns `[-inf, +inf]` in the declared output shape and
        // its linear rule returns `LinearBounds::conservative` (zero
        // coefficients, ±inf bias). That is a genuinely SOUND
        // over-approximation of an arbitrary unknown op (vacuous, maximally
        // loose — verification only succeeds if nothing downstream depends on
        // this value), so the bridge mirrors NN's lowering byte-for-byte:
        // emit `LayerType::Unknown` with the op name retained in a
        // `custom_op_name` attribute for diagnostics; the shared graph
        // builder auto-substitutes `Layer::OpaqueSkip`. Refusing here would
        // reject entire models over one explicitly-opaque escape-hatch op
        // for which ±inf is the only possible sound treatment.
        //
        // Contrast with `Roll` (ops_misc): a KNOWN op with computable
        // semantics got no vacuous fallback — its refusal forced the real
        // Slice+Concat rotation lowering it now has, instead of silently
        // destroying precision.
        TraceOp::Custom { name: custom_name } => {
            let mut attrs = HashMap::new();
            attrs.insert(
                "custom_op_name".to_string(),
                AttributeValue::String(custom_name.clone()),
            );
            Ok(NodeOutput::one(simple_spec(
                name,
                LayerType::Unknown,
                input_tensors,
                &output_tensor,
                attrs,
            )))
        }

        // -- Genuinely refused forever: `SegmentBoundary` (a pipeline marker,
        //    not a computation). Sound-by-construction refusal; explicit arm
        //    (no wildcard) so adding a TraceOp variant forces a routing
        //    decision here at compile time.
        //
        //    NOT here any more, and deliberately:
        //
        //    - The selection ops (`Topk` / `Argmax` / `Argmin` / `ArgSort` /
        //      `Sort`) never had data-dependent output SHAPES — every one of
        //      them is statically shaped — so they route to `ops_misc`, which
        //      lowers all five, `Topk` for the one largest/sorted combination
        //      `TopkLayer` models (the flags ride the wire now) and refusing
        //      the other three by name.
        //    - The fused `MultiHeadAttention` routes to `ops_attention`, which
        //      lowers it: the projections ride the wire now, so the
        //      wire-format gap that forced a blanket refusal is closed.
        //    - `Scatter` left this arm when the refusal's premise was checked
        //      rather than assumed. "The destination indices are runtime
        //      values" is true only when the index operand IS a runtime value;
        //      nn traces capture untraced index tensors as `ConstantWeight`
        //      payloads, and for a translate-time-constant index every
        //      destination is known at translation. `ops_misc` rewrites those
        //      into absolute ND coordinates for `ScatterNdLayer`, whose
        //      duplicate-destination min/max combine is the UNION of the
        //      candidate writes (the overwrite's last-writer order is
        //      genuinely unfixed — nn's CPU loop and its racing GPU kernel
        //      disagree by construction — so a union is the only sound
        //      enclosure). A runtime index is still refused, now with that
        //      precise reason instead of a blanket one. --
        TraceOp::SegmentBoundary { .. } => Err(NyError::UnsupportedOp(format!(
            "{} not supported in NY trace translation",
            op_name(op)
        ))),
    }
}

// ---------------------------------------------------------------------------
// Misc helpers
// ---------------------------------------------------------------------------

/// Borrow the first input tensor name, or error if there are none.
fn first_input(input_tensors: &[String], op: &str) -> Result<String> {
    input_tensors
        .first()
        .cloned()
        .ok_or_else(|| NyError::InternalError(format!("{op} has no inputs")))
}

/// Stable display name for a [`TraceOp`] used in error messages.
///
/// Faithful to the source variant identifiers so an `UnsupportedOp` error names
/// the exact op the producer would recognize.
fn op_name(op: &TraceOp) -> &'static str {
    match op {
        TraceOp::Input => "Input",
        TraceOp::ConstantWeight { .. } => "ConstantWeight",
        TraceOp::Add => "Add",
        TraceOp::Sub => "Sub",
        TraceOp::Mul => "Mul",
        TraceOp::Div => "Div",
        TraceOp::Maximum => "Maximum",
        TraceOp::Minimum => "Minimum",
        TraceOp::MatMul => "MatMul",
        TraceOp::Relu => "Relu",
        TraceOp::Gelu => "Gelu",
        TraceOp::GeluErf => "GeluErf",
        TraceOp::Silu => "Silu",
        TraceOp::Tanh => "Tanh",
        TraceOp::Sigmoid => "Sigmoid",
        TraceOp::Exp => "Exp",
        TraceOp::Log => "Log",
        TraceOp::Sqrt => "Sqrt",
        TraceOp::Sqr => "Sqr",
        TraceOp::Abs => "Abs",
        TraceOp::Neg => "Neg",
        TraceOp::Recip => "Recip",
        TraceOp::Sin => "Sin",
        TraceOp::Cos => "Cos",
        TraceOp::Tan => "Tan",
        TraceOp::Floor => "Floor",
        TraceOp::Ceil => "Ceil",
        TraceOp::Round => "Round",
        TraceOp::Sign => "Sign",
        TraceOp::Fract => "Fract",
        TraceOp::ReduceSum { .. } => "ReduceSum",
        TraceOp::ReduceMean { .. } => "ReduceMean",
        TraceOp::ReduceMax { .. } => "ReduceMax",
        TraceOp::ReduceMin { .. } => "ReduceMin",
        TraceOp::Reshape { .. } => "Reshape",
        TraceOp::Transpose { .. } => "Transpose",
        TraceOp::Narrow { .. } => "Narrow",
        TraceOp::Unsqueeze { .. } => "Unsqueeze",
        TraceOp::Squeeze { .. } => "Squeeze",
        TraceOp::Permute { .. } => "Permute",
        TraceOp::Cat { .. } => "Cat",
        TraceOp::LayerNorm { .. } => "LayerNorm",
        TraceOp::RmsNorm { .. } => "RmsNorm",
        TraceOp::GroupNorm { .. } => "GroupNorm",
        TraceOp::InstanceNorm { .. } => "InstanceNorm",
        TraceOp::BatchNorm { .. } => "BatchNorm",
        TraceOp::Linear { .. } => "Linear",
        TraceOp::Conv1d { .. } => "Conv1d",
        TraceOp::Conv2d { .. } => "Conv2d",
        TraceOp::Conv3d { .. } => "Conv3d",
        TraceOp::ConvTranspose1d { .. } => "ConvTranspose1d",
        TraceOp::ConvTranspose2d { .. } => "ConvTranspose2d",
        TraceOp::Softmax { .. } => "Softmax",
        TraceOp::LogSoftmax { .. } => "LogSoftmax",
        TraceOp::Sdpa { .. } => "Sdpa",
        TraceOp::SdpaCausal { .. } => "SdpaCausal",
        TraceOp::RotaryEmbedding { .. } => "RotaryEmbedding",
        TraceOp::MultiHeadAttention { .. } => "MultiHeadAttention",
        TraceOp::Embedding { .. } => "Embedding",
        TraceOp::Lstm { .. } => "Lstm",
        TraceOp::MaxPool1d { .. } => "MaxPool1d",
        TraceOp::AvgPool2d { .. } => "AvgPool2d",
        TraceOp::MaxPool2d { .. } => "MaxPool2d",
        TraceOp::AdaptiveAvgPool2d { .. } => "AdaptiveAvgPool2d",
        TraceOp::AvgPool1d { .. } => "AvgPool1d",
        TraceOp::AdaptiveAvgPool1d { .. } => "AdaptiveAvgPool1d",
        TraceOp::AdaptiveMaxPool2d { .. } => "AdaptiveMaxPool2d",
        TraceOp::Activation { .. } => "Activation",
        TraceOp::Elu { .. } => "Elu",
        TraceOp::LeakyRelu { .. } => "LeakyRelu",
        TraceOp::Softplus => "Softplus",
        TraceOp::Selu => "Selu",
        TraceOp::Celu { .. } => "Celu",
        TraceOp::Mish => "Mish",
        TraceOp::HardSigmoid => "HardSigmoid",
        TraceOp::HardSwish => "HardSwish",
        TraceOp::Softsign => "Softsign",
        TraceOp::PRelu { .. } => "PRelu",
        TraceOp::KokoroFused(_) => "KokoroFused",
        TraceOp::SwiGlu => "SwiGlu",
        TraceOp::Dropout => "Dropout",
        TraceOp::PixelShuffle { .. } => "PixelShuffle",
        TraceOp::PixelUnshuffle { .. } => "PixelUnshuffle",
        TraceOp::Upsample1d { .. } => "Upsample1d",
        TraceOp::Upsample2d { .. } => "Upsample2d",
        TraceOp::ResizeBilinear { .. } => "ResizeBilinear",
        TraceOp::Triu { .. } => "Triu",
        TraceOp::Tril { .. } => "Tril",
        TraceOp::GridSample { .. } => "GridSample",
        TraceOp::QLinear { .. } => "QLinear",
        TraceOp::Topk { .. } => "Topk",
        TraceOp::Argmax { .. } => "Argmax",
        TraceOp::Argmin { .. } => "Argmin",
        TraceOp::ArgSort { .. } => "ArgSort",
        TraceOp::Sort { .. } => "Sort",
        TraceOp::IndexSelect { .. } => "IndexSelect",
        TraceOp::Gather { .. } => "Gather",
        TraceOp::WhereCond => "WhereCond",
        TraceOp::Expand { .. } => "Expand",
        TraceOp::Compare { .. } => "Compare",
        TraceOp::CompareTensor { .. } => "CompareTensor",
        TraceOp::Cumsum { .. } => "Cumsum",
        TraceOp::RepeatInterleave { .. } => "RepeatInterleave",
        TraceOp::Powf { .. } => "Powf",
        TraceOp::ToDtype { .. } => "ToDtype",
        TraceOp::Flip { .. } => "Flip",
        TraceOp::Roll { .. } => "Roll",
        TraceOp::Unfold { .. } => "Unfold",
        TraceOp::SliceSet { .. } => "SliceSet",
        TraceOp::Scatter { .. } => "Scatter",
        TraceOp::ScatterAdd { .. } => "ScatterAdd",
        TraceOp::IndexAdd { .. } => "IndexAdd",
        TraceOp::IndexPut { .. } => "IndexPut",
        TraceOp::Clamp { .. } => "Clamp",
        TraceOp::Constant { .. } => "Constant",
        TraceOp::ReflectionPad1d { .. } => "ReflectionPad1d",
        TraceOp::ReflectionPad2d { .. } => "ReflectionPad2d",
        TraceOp::ConstantPadNd { .. } => "ConstantPadNd",
        TraceOp::Atan2 => "Atan2",
        TraceOp::Arange { .. } => "Arange",
        TraceOp::SegmentBoundary { .. } => "SegmentBoundary",
        TraceOp::MoeGating { .. } => "MoeGating",
        TraceOp::Custom { .. } => "Custom",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{ComputationGraph, DType, NodeId, TraceNode, TraceOp, WeightPayload};

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

    fn layer_types(model: &GraphModel) -> Vec<LayerType> {
        model
            .network
            .layers
            .iter()
            .map(|l| l.layer_type.clone())
            .collect()
    }

    fn count(model: &GraphModel, lt: &LayerType) -> usize {
        model
            .network
            .layers
            .iter()
            .filter(|l| &l.layer_type == lt)
            .count()
    }

    /// Tiny MLP: Input → Linear → ReLU → Linear.
    #[test]
    fn translates_tiny_mlp() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[4]),
            node(
                1,
                "fc1",
                TraceOp::Linear {
                    weight: WeightPayload::f32(vec![0.1; 8], vec![2, 4]),
                    bias: Some(WeightPayload::f32(vec![0.0, 0.0], vec![2])),
                },
                &[0],
                &[2],
            ),
            node(2, "act", TraceOp::Relu, &[1], &[2]),
            node(
                3,
                "fc2",
                TraceOp::Linear {
                    weight: WeightPayload::f32(vec![0.2; 6], vec![3, 2]),
                    bias: None,
                },
                &[2],
                &[3],
            ),
        ]);

        let model = translate(&graph).expect("MLP translates");

        // Input emits Add(x, 0); fc1 → Linear; act → ReLU; fc2 → Linear.
        assert_eq!(count(&model, &LayerType::Linear), 2, "two Linear layers");
        assert_eq!(count(&model, &LayerType::ReLU), 1, "one ReLU layer");
        // The input identity is an Add(x, 0).
        assert_eq!(count(&model, &LayerType::Add), 1, "input identity Add");
        assert_eq!(
            layer_types(&model).len(),
            4,
            "Add(input) + Linear + ReLU + Linear"
        );

        // Weights for both Linear layers (+ fc1 bias) are in the store.
        assert!(model.weights.contains_key("layer0_trace_1_weight"));
        assert!(model.weights.contains_key("layer0_trace_1_bias"));
        assert!(model.weights.contains_key("layer0_trace_3_weight"));

        // The graph output tensor name resolves and the network builds.
        assert_eq!(model.network.outputs.len(), 1);
        model
            .build_graph_network(ny_build::GraphNetworkOptions::default())
            .expect("MLP GraphModel builds a graph network");
    }

    /// Conv1d translates to a Conv1d layer with the right attributes.
    #[test]
    fn translates_conv() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[3, 8]),
            node(
                1,
                "conv",
                TraceOp::Conv1d {
                    weight: WeightPayload::f32(vec![0.1; 4 * 3 * 3], vec![4, 3, 3]),
                    bias: Some(WeightPayload::f32(vec![0.0; 4], vec![4])),
                    padding: 1,
                    stride: 1,
                    dilation: 1,
                    groups: 1,
                },
                &[0],
                &[4, 8],
            ),
        ]);

        let model = translate(&graph).expect("conv translates");
        assert_eq!(count(&model, &LayerType::Conv1d), 1, "one Conv1d layer");

        let conv = model
            .network
            .layers
            .iter()
            .find(|l| l.layer_type == LayerType::Conv1d)
            .expect("conv layer present");
        assert_eq!(
            conv.attributes.get("pads"),
            Some(&AttributeValue::Ints(vec![1, 1]))
        );
        assert_eq!(
            conv.attributes.get("strides"),
            Some(&AttributeValue::Ints(vec![1]))
        );
        assert!(conv.weights.is_some(), "conv references its kernel weight");
        // Data input + weight + bias.
        assert_eq!(conv.inputs.len(), 3);
        model
            .build_graph_network(ny_build::GraphNetworkOptions::default())
            .expect("conv GraphModel builds a graph network");
    }

    #[test]
    fn dilated_conv1d_materialization_cap_fails_before_allocation() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[1, 2]),
            node(
                1,
                "conv",
                TraceOp::Conv1d {
                    weight: WeightPayload::f32(vec![1.0, 1.0], vec![1, 1, 2]),
                    bias: None,
                    padding: 0,
                    stride: 1,
                    dilation: 10_000_001,
                    groups: 1,
                },
                &[0],
                &[1, 1],
            ),
        ]);

        let err = translate(&graph)
            .expect_err("oversized dilated Conv1d expansion must fail before allocation");
        assert!(
            matches!(err, NyError::ModelLoad(ref message)
                if message.contains("Conv1d dilation expansion")
                    && message.contains("materialization limit")),
            "got {err:?}"
        );
    }

    /// Conv2d translates with 2D pads/strides/dilations.
    #[test]
    fn translates_conv2d() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[3, 8, 8]),
            node(
                1,
                "conv",
                TraceOp::Conv2d {
                    weight: WeightPayload::f32(vec![0.1; 4 * 3 * 3 * 3], vec![4, 3, 3, 3]),
                    bias: None,
                    padding: [1, 1],
                    stride: [2, 2],
                    dilation: [1, 1],
                    groups: 1,
                },
                &[0],
                &[4, 4, 4],
            ),
        ]);
        let model = translate(&graph).expect("conv2d translates");
        let conv = model
            .network
            .layers
            .iter()
            .find(|l| l.layer_type == LayerType::Conv2d)
            .expect("conv2d layer present");
        assert_eq!(
            conv.attributes.get("strides"),
            Some(&AttributeValue::Ints(vec![2, 2]))
        );
        assert_eq!(
            conv.attributes.get("pads"),
            Some(&AttributeValue::Ints(vec![1, 1, 1, 1]))
        );
    }

    /// A refused op returns UnsupportedOp naming the op — and, now that the
    /// dispatch routes the selection family to `ops_misc`, carrying that
    /// module's TRUE refusal reason end-to-end.
    ///
    /// The witness used to be a bare `Topk`, refused because the wire withheld
    /// nn's `largest`/`sorted` flags. Both are serialized now, so `Topk` lowers
    /// for `largest && sorted` and the end-to-end refusal to pin is one of the
    /// combinations `TopkLayer` does not model — still owned by `ops_misc`,
    /// still reaching the caller without the generic catch-all text.
    #[test]
    fn unhandled_op_returns_unsupported() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[8]),
            node(
                1,
                "topk",
                TraceOp::Topk {
                    k: 3,
                    dim: 0,
                    largest: false,
                    sorted: true,
                },
                &[0],
                &[3],
            ),
        ]);
        let err = translate(&graph).expect_err("k-smallest Topk is unsupported");
        match err {
            NyError::UnsupportedOp(msg) => {
                assert!(msg.contains("Topk"), "error names the op: {msg}");
                assert!(
                    msg.contains("largest=false") && msg.contains("k smallest"),
                    "the message states the real reason (the flag combination \
                     TopkLayer does not model): {msg}"
                );
                assert!(
                    !msg.contains("not supported in NY trace translation"),
                    "the generic catch-all text is gone now that ops_misc \
                     owns the refusal: {msg}"
                );
            }
            other => panic!("expected UnsupportedOp, got {other:?}"),
        }

        // And the sibling combination the layer DOES model lowers end-to-end.
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[8]),
            node(
                1,
                "topk",
                TraceOp::Topk {
                    k: 3,
                    dim: 0,
                    largest: true,
                    sorted: true,
                },
                &[0],
                &[3],
            ),
        ]);
        translate(&graph).expect("largest && sorted lowers through the same dispatch");
    }

    /// A second unsupported op (Gather) also refuses rather than passthrough.
    #[test]
    fn gather_returns_unsupported() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[8]),
            node(1, "g", TraceOp::Gather { dim: 0 }, &[0], &[8]),
        ]);
        let err = translate(&graph).expect_err("Gather is unsupported");
        assert!(matches!(err, NyError::UnsupportedOp(m) if m.contains("Gather")));
    }

    /// `Custom` — the explicit opaque escape hatch — lowers to the sound
    /// conservative OpaqueSkip (`LayerType::Unknown` + `custom_op_name`
    /// attribute, mirroring NN #4349), and IBP over the built network widens
    /// to `[-inf, +inf]`: a genuine over-approximation of any op, never an
    /// identity passthrough (which would be unsound). INC-FINAL
    /// reconciliation of the Custom divergence.
    #[test]
    fn custom_lowers_to_sound_opaque_skip() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[2, 4]),
            node(
                1,
                "c",
                TraceOp::Custom {
                    name: "nn_fancy_op".into(),
                },
                &[0],
                &[2, 4],
            ),
        ]);
        let model = translate(&graph).expect("Custom translates to OpaqueSkip");
        let unknown = model
            .network
            .layers
            .iter()
            .find(|l| l.layer_type == LayerType::Unknown)
            .expect("Unknown layer present (graph builder substitutes OpaqueSkip)");
        assert_eq!(
            unknown.attributes.get("custom_op_name"),
            Some(&AttributeValue::String("nn_fancy_op".into())),
            "op name retained for diagnostics"
        );

        let network = model
            .build_graph_network(ny_build::GraphNetworkOptions::default())
            .expect("builds with the OpaqueSkip substitution");
        let input = ny_tensor::BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[2, 4]), -1.0_f32),
            ArrayD::from_elem(IxDyn(&[2, 4]), 1.0_f32),
        )
        .expect("valid input box");
        let out = network.propagate_ibp(&input).expect("IBP over OpaqueSkip");
        assert!(
            out.lower()
                .iter()
                .all(|v| v.is_infinite() && v.is_sign_negative()),
            "lower must widen to -inf (sound over-approximation, not identity)"
        );
        assert!(
            out.upper()
                .iter()
                .all(|v| v.is_infinite() && v.is_sign_positive()),
            "upper must widen to +inf"
        );
    }

    /// Activations map to the expected NY LayerTypes.
    #[test]
    fn activations_map_to_expected_layer_types() {
        let cases = [
            (TraceOp::Sigmoid, LayerType::Sigmoid),
            (TraceOp::Tanh, LayerType::Tanh),
            (TraceOp::Silu, LayerType::SiLU),
            (TraceOp::Gelu, LayerType::GELU),
            (TraceOp::GeluErf, LayerType::GELU),
        ];
        for (op, expected) in cases {
            let graph = ComputationGraph::from_nodes(vec![
                node(0, "x", TraceOp::Input, &[], &[4]),
                node(1, "act", op.clone(), &[0], &[4]),
            ]);
            let model = translate(&graph).expect("activation translates");
            assert_eq!(
                count(&model, &expected),
                1,
                "{} maps to {:?}",
                op_name(&op),
                expected
            );
        }
    }

    /// The bridge never clips an `Exp` input, and `Exp(100)` — whose true
    /// value 2.69e43 is far outside the f32 range — comes back as a sound
    /// SATURATING enclosure rather than a refusal or a silently-shrunk
    /// `Exp(88)`.
    ///
    /// This pins the current `ExpLayer` monotone-limit contract (the IBP rule
    /// in ny-propagate `layers/activations/exp.rs`): only NaN input bounds
    /// fail closed; a finite input above the f32 overflow threshold saturates
    /// the UPPER endpoint to +inf (the sound outward direction) while the
    /// LOWER endpoint stays FINITE at the largest representable value —
    /// +inf there would claim `e^100 = +inf` for a finite quantity, which is
    /// exactly the false-proof shape the layer's `f32::MAX` branch exists to
    /// prevent. The bridge's job is unchanged and is what this test guards:
    /// it emits the bare `Exp` with no input-domain rewriting, so whatever
    /// enclosure the layer computes is the one verification sees.
    #[test]
    fn exp_above_f32_threshold_saturates_without_clipping() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[1]),
            node(1, "e", TraceOp::Exp, &[0], &[1]),
        ]);
        let model = translate(&graph).expect("exp translates");
        assert_eq!(count(&model, &LayerType::Clip), 0, "input is not altered");
        assert_eq!(count(&model, &LayerType::Exp), 1, "exp present");

        let network = model
            .build_graph_network(ny_build::GraphNetworkOptions::default())
            .expect("Exp model builds");
        let input = ny_tensor::BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[1]), 100.0),
            ArrayD::from_elem(IxDyn(&[1]), 100.0),
        )
        .expect("valid point bounds");
        let out = network
            .propagate_ibp(&input)
            .expect("Exp(100) saturates outward rather than failing closed");
        let lower = out.lower()[0];
        let upper = out.upper()[0];
        assert!(
            lower.is_finite() && lower > 0.0,
            "lower endpoint must stay finite and positive (a +inf lower would \
             be a false proof about a finite e^100): got {lower}"
        );
        assert!(
            lower >= f32::MAX * 0.5,
            "lower endpoint must saturate near f32::MAX, not collapse to an \
             exp(88)-shaped value: got {lower}"
        );
        assert!(
            upper.is_infinite() && upper.is_sign_positive(),
            "upper endpoint saturates to +inf (the sound outward direction): got {upper}"
        );
    }

    #[test]
    fn softplus_and_mish_above_88_preserve_large_values() {
        for (op, layer_type) in [
            (TraceOp::Softplus, LayerType::Softplus),
            (TraceOp::Mish, LayerType::Mish),
        ] {
            let graph = ComputationGraph::from_nodes(vec![
                node(0, "x", TraceOp::Input, &[], &[1]),
                node(1, "act", op, &[0], &[1]),
            ]);
            let model = translate(&graph).expect("activation translates");
            assert_eq!(count(&model, &LayerType::Clip), 0, "input is not altered");
            assert_eq!(count(&model, &layer_type), 1);

            let network = model
                .build_graph_network(ny_build::GraphNetworkOptions::default())
                .expect("activation model builds");
            let input = ny_tensor::BoundedTensor::new(
                ArrayD::from_elem(IxDyn(&[1]), 100.0),
                ArrayD::from_elem(IxDyn(&[1]), 100.0),
            )
            .expect("valid point bounds");
            let output = network
                .propagate_ibp(&input)
                .expect("large finite activation propagates");
            assert!(
                output.lower()[[0]] > 99.0,
                "{layer_type:?}(100) must not be truncated to 88: {:?}",
                output.lower()
            );
        }
    }

    /// LeakyRelu decomposes into Mul/ReLU/Mul/Add.
    #[test]
    fn leaky_relu_decomposes() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[4]),
            node(1, "lr", TraceOp::LeakyRelu { slope: 0.1 }, &[0], &[4]),
        ]);
        let model = translate(&graph).expect("leaky relu translates");
        // Input Add(x,0) + decomposition Add → 2 Adds; 2 Muls; 1 ReLU.
        assert_eq!(count(&model, &LayerType::ReLU), 1);
        assert_eq!(count(&model, &LayerType::Mul), 2);
        assert_eq!(count(&model, &LayerType::Add), 2);
    }

    /// Sub becomes Neg + Add.
    #[test]
    fn sub_becomes_neg_add() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "a", TraceOp::Input, &[], &[4]),
            node(
                1,
                "w",
                TraceOp::ConstantWeight {
                    weight: WeightPayload::f32(vec![1.0; 4], vec![4]),
                },
                &[],
                &[4],
            ),
            node(2, "s", TraceOp::Sub, &[0, 1], &[4]),
        ]);
        let model = translate(&graph).expect("sub translates");
        assert_eq!(count(&model, &LayerType::Neg), 1, "sub emits a Neg");
        // input identity Add + sub Add.
        assert_eq!(count(&model, &LayerType::Add), 2);
    }

    /// LayerNorm maps to a LayerNorm layer with epsilon + normalized_shape.
    #[test]
    fn layer_norm_maps_with_attrs() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[4]),
            node(
                1,
                "ln",
                TraceOp::LayerNorm {
                    eps: 1e-5,
                    weight: WeightPayload::f32(vec![1.0; 4], vec![4]),
                    bias: WeightPayload::f32(vec![0.0; 4], vec![4]),
                },
                &[0],
                &[4],
            ),
        ]);
        let model = translate(&graph).expect("layernorm translates");
        let ln = model
            .network
            .layers
            .iter()
            .find(|l| l.layer_type == LayerType::LayerNorm)
            .expect("layernorm present");
        assert_eq!(
            ln.attributes.get("epsilon"),
            Some(&AttributeValue::Float(1e-5))
        );
        assert_eq!(
            ln.attributes.get("normalized_shape"),
            Some(&AttributeValue::Ints(vec![4]))
        );
    }

    /// GroupNorm decomposes into Reshape → InstanceNorm → Reshape → Mul → Add.
    #[test]
    fn group_norm_decomposes() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[1, 4, 6]),
            node(
                1,
                "gn",
                TraceOp::GroupNorm {
                    num_groups: 2,
                    eps: 1e-5,
                    weight: WeightPayload::f32(vec![1.0; 4], vec![4]),
                    bias: WeightPayload::f32(vec![0.0; 4], vec![4]),
                },
                &[0],
                &[1, 4, 6],
            ),
        ]);
        let model = translate(&graph).expect("groupnorm translates");
        assert_eq!(count(&model, &LayerType::InstanceNorm), 1);
        assert_eq!(count(&model, &LayerType::Reshape), 2);
        assert_eq!(count(&model, &LayerType::Mul), 1);
    }

    #[test]
    fn synthetic_norm_parameters_obey_materialization_cap() {
        let channels = 10_000_001;
        let instance_graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[1, channels, 1]),
            node(
                1,
                "instance_norm",
                TraceOp::InstanceNorm { eps: 1e-5 },
                &[0],
                &[1, channels, 1],
            ),
        ]);
        let err = translate(&instance_graph)
            .expect_err("oversized InstanceNorm parameters must fail before allocation");
        assert!(
            matches!(err, NyError::ModelLoad(ref message)
                if message.contains("InstanceNorm gamma")
                    && message.contains("materialization limit")),
            "got {err:?}"
        );

        let group_graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[1, channels, 1]),
            node(
                1,
                "group_norm",
                TraceOp::GroupNorm {
                    num_groups: 1,
                    eps: 1e-5,
                    weight: WeightPayload::f32(vec![1.0], vec![channels]),
                    bias: WeightPayload::f32(vec![0.0], vec![channels]),
                },
                &[0],
                &[1, channels, 1],
            ),
        ]);
        let err = translate(&group_graph)
            .expect_err("oversized GroupNorm parameters must fail before allocation");
        assert!(
            matches!(err, NyError::ModelLoad(ref message)
                if message.contains("GroupNorm synthetic InstanceNorm gamma")
                    && message.contains("materialization limit")),
            "got {err:?}"
        );
    }

    /// Shape ops: Reshape, Transpose, Permute, Squeeze, Unsqueeze, Cat.
    #[test]
    fn shape_ops_map() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[2, 3]),
            node(
                1,
                "t",
                TraceOp::Transpose { dim0: 0, dim1: 1 },
                &[0],
                &[3, 2],
            ),
            node(
                2,
                "r",
                TraceOp::Reshape {
                    target_shape: vec![6],
                },
                &[1],
                &[6],
            ),
            node(3, "u", TraceOp::Unsqueeze { dim: 0 }, &[2], &[1, 6]),
            node(4, "sq", TraceOp::Squeeze { dim: 0 }, &[3], &[6]),
        ]);
        let model = translate(&graph).expect("shape ops translate");
        // Transpose + Permute both map to LayerType::Transpose; here just one.
        assert_eq!(count(&model, &LayerType::Transpose), 1);
        assert_eq!(count(&model, &LayerType::Reshape), 1);
        assert_eq!(count(&model, &LayerType::Unsqueeze), 1);
        assert_eq!(count(&model, &LayerType::Squeeze), 1);
    }

    /// Softmax carries the axis attribute (raw dim).
    #[test]
    fn softmax_carries_axis() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[4]),
            node(1, "sm", TraceOp::Softmax { dim: 0 }, &[0], &[4]),
        ]);
        let model = translate(&graph).expect("softmax translates");
        let sm = model
            .network
            .layers
            .iter()
            .find(|l| l.layer_type == LayerType::Softmax)
            .expect("softmax present");
        assert_eq!(sm.attributes.get("axis"), Some(&AttributeValue::Int(0)));
    }

    /// An empty graph is rejected.
    #[test]
    fn empty_graph_rejected() {
        let graph = ComputationGraph::from_nodes(vec![]);
        assert!(matches!(translate(&graph), Err(NyError::InternalError(_))));
    }

    /// Duplicate node ids make id-based edge/output resolution ambiguous and
    /// therefore must be rejected at the wire boundary.
    #[test]
    fn duplicate_node_ids_are_rejected() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[1]),
            node(0, "duplicate", TraceOp::Relu, &[0], &[1]),
        ]);
        let err = translate(&graph).expect_err("duplicate ids must be refused");
        assert!(
            matches!(err, NyError::InternalError(ref m) if m.contains("duplicate trace node id 0")),
            "got {err:?}"
        );
    }

    /// A malformed explicit output list must not silently fall back to the
    /// physical last node when its primary (last-marked) output is dangling.
    #[test]
    fn dangling_marked_output_is_rejected_without_fallback() {
        let mut graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[1]),
            node(1, "relu", TraceOp::Relu, &[0], &[1]),
        ]);
        graph.output_nodes = vec![NodeId(1), NodeId(999)];
        let err = translate(&graph).expect_err("dangling marked output must be refused");
        assert!(
            matches!(err, NyError::InternalError(ref m)
                if m.contains("marked output node 999")),
            "got {err:?}"
        );
    }

    /// A shape-only placeholder weight is rejected (no silent vacuous layer).
    #[test]
    fn placeholder_weight_rejected() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[4]),
            node(
                1,
                "fc",
                TraceOp::Linear {
                    weight: WeightPayload::placeholder(vec![2, 4]),
                    bias: None,
                },
                &[0],
                &[2],
            ),
        ]);
        assert!(matches!(translate(&graph), Err(NyError::ModelLoad(_))));
    }

    /// translate_segmented lowers each segment independently.
    ///
    /// Each post-boundary segment re-roots at its own `Input` node (matching how
    /// real segmented traces are captured), so every split segment is a
    /// self-contained, topologically valid sub-graph.
    #[test]
    fn segmented_translation_yields_one_model_per_segment() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[4]),
            node(1, "r1", TraceOp::Relu, &[0], &[4]),
            node(
                10,
                "boundary",
                TraceOp::SegmentBoundary {
                    reason: "test".into(),
                    input_bounds: None,
                },
                &[1],
                &[4],
            ),
            // Segment 2 re-roots at a fresh Input.
            node(2, "x2", TraceOp::Input, &[], &[4]),
            node(3, "r2", TraceOp::Relu, &[2], &[4]),
        ]);
        let segmented = graph.split_at_segment_boundaries();
        let models = translate_segmented(&segmented).expect("segments translate");
        assert_eq!(models.len(), 2, "two segments → two models");
        // Each segment lowered to exactly one input identity + one ReLU.
        for m in &models {
            assert_eq!(
                m.network
                    .layers
                    .iter()
                    .filter(|l| l.layer_type == LayerType::ReLU)
                    .count(),
                1
            );
        }
    }

    /// Two variable inputs in single-input mode are refused (aliasing them to
    /// one network input could produce a false "holds").
    #[test]
    fn single_input_guard_rejects_two_variable_inputs() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "a", TraceOp::Input, &[], &[2, 3]),
            node(1, "b", TraceOp::Input, &[], &[2, 3]),
            node(2, "sum", TraceOp::Add, &[0, 1], &[2, 3]),
        ]);
        let err = translate(&graph).expect_err("two variable inputs must be refused");
        match err {
            NyError::UnsupportedConfiguration(msg) => {
                assert!(
                    msg.contains("2 variable inputs") && msg.contains("translate_multi_input"),
                    "error names the count and the escape hatch: {msg}"
                );
            }
            other => panic!("expected UnsupportedConfiguration, got {other:?}"),
        }
    }

    /// A second Input consumed only as a composite op's weight edge is not a
    /// variable input, so the guard passes.
    #[test]
    fn single_input_guard_ignores_weight_only_input() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[4]),
            node(1, "w", TraceOp::Input, &[], &[2, 4]),
            node(
                2,
                "fc",
                TraceOp::Linear {
                    weight: WeightPayload::f32(vec![0.1; 8], vec![2, 4]),
                    bias: None,
                },
                &[0, 1],
                &[2],
            ),
        ]);
        translate(&graph).expect("weight-only second Input passes the guard");
    }

    /// A dead node carrying an unported op is skipped, not refused: only nodes
    /// reachable from the marked outputs are translated.
    #[test]
    fn dead_node_with_unported_op_is_skipped() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[8]),
            // Dead: consumes x but feeds nothing on the output path.
            node(
                1,
                "topk",
                TraceOp::Topk {
                    k: 3,
                    dim: 0,
                    largest: true,
                    sorted: true,
                },
                &[0],
                &[3],
            ),
            node(2, "act", TraceOp::Relu, &[0], &[8]),
        ]);
        let model = translate(&graph).expect("dead Topk must not fail translation");
        // Input identity Add + ReLU only; nothing emitted for the dead node.
        assert_eq!(count(&model, &LayerType::ReLU), 1);
        assert_eq!(count(&model, &LayerType::Add), 1);
        assert_eq!(layer_types(&model).len(), 2);
        model
            .build_graph_network(ny_build::GraphNetworkOptions::default())
            .expect("dead-node GraphModel builds a graph network");
    }

    /// Multi-input mode stacks two variable inputs into `multi_in` and splits
    /// them back with Slice + Reshape; the GraphModel builds.
    #[test]
    fn multi_input_translates_and_builds() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "a", TraceOp::Input, &[], &[2, 3]),
            node(1, "b", TraceOp::Input, &[], &[2, 3]),
            node(2, "sum", TraceOp::Add, &[0, 1], &[2, 3]),
        ]);
        let translation = translate_multi_input(&graph).expect("multi-input translates");
        let model = &translation.model;
        assert_eq!(translation.dtype_cast_count, 0);

        // One stacked input of shape [6 + 6].
        assert_eq!(model.network.inputs.len(), 1);
        assert_eq!(model.network.inputs[0].name, "multi_in");
        assert_eq!(model.network.inputs[0].shape, vec![12]);

        // Per variable: Slice + Reshape; then the Add node.
        assert_eq!(count(model, &LayerType::Slice), 2);
        assert_eq!(count(model, &LayerType::Reshape), 2);
        assert_eq!(count(model, &LayerType::Add), 1);

        // Slices carve out [0, 6) and [6, 12) in node order along axis 0 —
        // the stacked model is always classified unbatched (single rank-1
        // input), so ny-build converts the axis VERBATIM: axis 0 is the only
        // axis of the rank-1 `multi_in` tensor. (The legacy `axis=1`
        // pretend-batched encoding is rejected fail-closed by ny-propagate:
        // "Slice: axis 1 out of range for 1D tensor".)
        let slices: Vec<&LayerSpec> = model
            .network
            .layers
            .iter()
            .filter(|l| l.layer_type == LayerType::Slice)
            .collect();
        assert_eq!(
            slices[0].attributes.get("axis"),
            Some(&AttributeValue::Int(0))
        );
        assert_eq!(
            slices[0].attributes.get("start"),
            Some(&AttributeValue::Int(0))
        );
        assert_eq!(
            slices[0].attributes.get("end"),
            Some(&AttributeValue::Int(6))
        );
        assert_eq!(
            slices[1].attributes.get("start"),
            Some(&AttributeValue::Int(6))
        );
        assert_eq!(
            slices[1].attributes.get("end"),
            Some(&AttributeValue::Int(12))
        );

        let net = model
            .build_graph_network(ny_build::GraphNetworkOptions::default())
            .expect("multi-input GraphModel builds a graph network");

        // Regression (#seams Slice-axis): the historic axis=1 emission built
        // fine but failed at PROPAGATION time inside ny-propagate's SliceLayer
        // axis resolution, so lock the fix by actually propagating IBP through
        // the stacked Slice/Reshape split.
        let lower = ArrayD::from_elem(IxDyn(&[12]), -1.0f32);
        let upper = ArrayD::from_elem(IxDyn(&[12]), 1.0f32);
        let input = ny_tensor::BoundedTensor::new(lower, upper).expect("valid input box");
        let out = net
            .propagate_ibp(&input)
            .expect("multi-input stacked Slice/Reshape propagates IBP");
        // a + b with a,b ∈ [-1,1]^{2x3} → output ∈ [-2,2]^{2x3}.
        assert_eq!(out.lower().len(), 6);
        for (&lo, &hi) in out.lower().iter().zip(out.upper().iter()) {
            assert_eq!(lo, -2.0);
            assert_eq!(hi, 2.0);
        }
    }

    #[test]
    fn multi_input_shape_product_overflow_is_an_error_not_a_panic() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "a", TraceOp::Input, &[], &[usize::MAX, 2]),
            node(1, "b", TraceOp::Input, &[], &[1]),
            node(2, "sum", TraceOp::Add, &[0, 1], &[1]),
        ]);
        let err = translate_multi_input(&graph).expect_err("shape overflow must be refused");
        assert!(
            matches!(err, NyError::InternalError(ref m)
                if m.contains("shape product overflows")),
            "got {err:?}"
        );
    }

    /// With one variable input, the multi-input entry degrades to exactly the
    /// single-input translation (no stacked tensor, no Slice/Reshape).
    #[test]
    fn multi_input_with_one_variable_input_degrades_to_single() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[4]),
            node(1, "act", TraceOp::Relu, &[0], &[4]),
        ]);
        let translation = translate_multi_input(&graph).expect("single-variable translates");
        let model = &translation.model;
        assert_eq!(count(model, &LayerType::Slice), 0);
        assert_eq!(count(model, &LayerType::Reshape), 0);
        assert_eq!(model.network.inputs.len(), 1);
        assert_ne!(model.network.inputs[0].name, "multi_in");
        assert_eq!(
            layer_types(model),
            layer_types(&translate(&graph).expect("single-input translates"))
        );
    }

    /// An F32 -> F32 cast is the one lowerable ToDtype: the producer's
    /// recorded `output_dtype` (the resolved source) and the target are both
    /// F32, so the cast lowers with the same `Add(x, 0)` identity idiom as
    /// Dropout and bounds pass through exactly.
    #[test]
    fn f32_identity_cast_lowers_as_identity_with_exact_bounds() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[4]),
            node(
                1,
                "cast",
                TraceOp::ToDtype {
                    target_dtype: DType::F32,
                },
                &[0],
                &[4],
            ),
        ]);
        let translation = translate_with_metadata(&graph).expect("F32 identity cast translates");
        let model = &translation.model;
        assert_eq!(translation.dtype_cast_count, 0);
        // Input identity Add + cast identity Add; nothing else.
        assert_eq!(count(model, &LayerType::Add), 2);
        assert_eq!(layer_types(model).len(), 2);

        let network = model
            .build_graph_network(ny_build::GraphNetworkOptions::default())
            .expect("identity-cast GraphModel builds a graph network");
        let lower = vec![-2.0_f32, -0.5, 0.25, 3.0];
        let upper = vec![-1.0_f32, 0.5, 1.25, 4.0];
        let input = ny_tensor::BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[4]), lower.clone()).expect("lower shape"),
            ArrayD::from_shape_vec(IxDyn(&[4]), upper.clone()).expect("upper shape"),
        )
        .expect("valid input box");
        let out = network
            .propagate_ibp(&input)
            .expect("identity cast propagates IBP");
        assert_eq!(
            out.lower().as_slice().expect("contiguous"),
            lower.as_slice(),
            "identity cast must not move the lower bound"
        );
        assert_eq!(
            out.upper().as_slice().expect("contiguous"),
            upper.as_slice(),
            "identity cast must not move the upper bound"
        );
    }

    /// Build `Input -> ToDtype(target) -> Relu`, so the cast is a well-formed
    /// INTERIOR node (its own declared output dtype IS the target) rather than
    /// the effective graph output. The trailing consumer keeps leg (d) of
    /// [`is_guarded_integer_cast`] out of the way, so these tests exercise the
    /// dtype rule itself.
    fn cast_with_consumer(target_dtype: DType) -> ComputationGraph {
        let mut cast = node(1, "cast", TraceOp::ToDtype { target_dtype }, &[0], &[4]);
        cast.output_dtype = target_dtype;
        ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[4]),
            cast,
            node(2, "use", TraceOp::Relu, &[1], &[4]),
        ])
    }

    /// A well-formed non-F32 cast declares the target as its own output
    /// dtype, so the graph-wide F32 gate refuses it before the ToDtype arm
    /// ever runs — for every target WITHOUT a modeled lowering. The two that
    /// do have one (I32/I64, the guarded Trunc) are covered separately.
    #[test]
    fn non_f32_dtype_casts_are_refused_by_the_graph_gate() {
        for target_dtype in [
            DType::F64,
            DType::F16,
            DType::Bf16,
            DType::U32,
            DType::U8,
            DType::Bool,
        ] {
            let err = translate_with_metadata(&cast_with_consumer(target_dtype))
                .expect_err("non-F32 cast must fail closed at the graph gate");
            assert!(
                matches!(err, NyError::UnsupportedOp(ref m)
                    if m.contains("output dtype")
                        && m.contains("only soundly supports F32")),
                "{target_dtype:?}: got {err:?}"
            );
        }
    }

    /// `F32 -> I32`/`I64` passes the gate and lowers to the GUARDED `Trunc`
    /// (`to = 6`/`7`). The integer label is admissible here because the f32
    /// mirror of `trunc(x)` is bit-exact — see [`is_guarded_integer_cast`].
    #[test]
    fn guarded_integer_casts_pass_the_dtype_gate_and_lower_to_trunc() {
        for (target_dtype, expected_to) in [(DType::I32, 6_i64), (DType::I64, 7_i64)] {
            let translation = translate_with_metadata(&cast_with_consumer(target_dtype))
                .unwrap_or_else(|e| panic!("{target_dtype:?} cast must pass the gate: {e:?}"));
            let model = &translation.model;
            assert_eq!(count(model, &LayerType::Trunc), 1, "{target_dtype:?}");
            let trunc = model
                .network
                .layers
                .iter()
                .find(|l| l.layer_type == LayerType::Trunc)
                .expect("Trunc present");
            assert_eq!(
                trunc.attributes.get("to"),
                Some(&AttributeValue::Int(expected_to)),
                "{target_dtype:?}: the guarded Cast target must survive to the LayerSpec"
            );
            assert!(
                translation.skipped_nodes.is_empty(),
                "a lowered cast is not a skip"
            );
        }
    }

    /// Leg (d): the carve-out does not admit an integer cast that IS the
    /// graph's effective output. ny's verifier API declares FLOAT32 outputs
    /// and ny-onnx refuses the same shape on an authored ONNX output; the
    /// bridge fails closed rather than diverge from it.
    #[test]
    fn guarded_integer_cast_as_the_graph_output_stays_refused() {
        for target_dtype in [DType::I32, DType::I64] {
            let mut cast = node(1, "cast", TraceOp::ToDtype { target_dtype }, &[0], &[4]);
            cast.output_dtype = target_dtype;
            let graph =
                ComputationGraph::from_nodes(vec![node(0, "x", TraceOp::Input, &[], &[4]), cast]);
            let err = translate_with_metadata(&graph)
                .expect_err("an integer cast on the graph output must fail closed");
            assert!(
                matches!(err, NyError::UnsupportedOp(ref m)
                    if m.contains("output dtype")
                        && m.contains("only soundly supports F32")),
                "{target_dtype:?}: got {err:?}"
            );
        }
    }

    /// Leg (b): a cast node that targets I64 but declares some OTHER integer
    /// dtype as its own output is an internally inconsistent trace. The gate
    /// trusts neither half of it.
    #[test]
    fn guarded_integer_cast_with_mismatched_declared_dtype_is_refused() {
        let mut cast = node(
            1,
            "cast",
            TraceOp::ToDtype {
                target_dtype: DType::I64,
            },
            &[0],
            &[4],
        );
        cast.output_dtype = DType::U32;
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[4]),
            cast,
            node(2, "use", TraceOp::Relu, &[1], &[4]),
        ]);
        let err = translate_with_metadata(&graph)
            .expect_err("a cast whose declared dtype contradicts its target must fail closed");
        assert!(
            matches!(err, NyError::UnsupportedOp(ref m)
                if m.contains("output dtype") && m.contains("only soundly supports F32")),
            "got {err:?}"
        );
    }

    /// Leg (c): the carve-out is keyed on the RESOLVED source too. A cast
    /// whose producer is not F32 is refused — here the producer itself is the
    /// first node the gate rejects, which is exactly the fail-closed order
    /// wanted: neither node is admitted on the other's behalf.
    #[test]
    fn guarded_integer_cast_from_a_non_f32_producer_is_refused() {
        let mut producer = node(0, "x", TraceOp::Input, &[], &[4]);
        producer.output_dtype = DType::F16;
        let mut cast = node(
            1,
            "cast",
            TraceOp::ToDtype {
                target_dtype: DType::I64,
            },
            &[0],
            &[4],
        );
        cast.output_dtype = DType::I64;
        let graph = ComputationGraph::from_nodes(vec![
            producer,
            cast,
            node(2, "use", TraceOp::Relu, &[1], &[4]),
        ]);
        let err = translate_with_metadata(&graph)
            .expect_err("a non-F32-sourced integer cast must fail closed");
        assert!(
            matches!(err, NyError::UnsupportedOp(ref m)
                if m.contains("output dtype") && m.contains("only soundly supports F32")),
            "got {err:?}"
        );
    }

    /// A malformed trace whose cast node claims an F32 output (passing the
    /// gate) but targets a non-F32 dtype is refused naming the RESOLVED
    /// source dtype — never lowered, and never blamed on a missing source.
    #[test]
    fn resolved_non_identity_cast_is_refused_naming_both_dtypes() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[4]),
            node(
                1,
                "cast",
                TraceOp::ToDtype {
                    target_dtype: DType::F16,
                },
                &[0],
                &[4],
            ),
        ]);
        let err = translate(&graph).expect_err("F32 -> F16 cast must be refused");
        assert!(
            matches!(err, NyError::UnsupportedOp(ref m)
                if m.contains("ToDtype F32 -> F16 is refused")
                    && m.contains("F16 target")),
            "got {err:?}"
        );
    }

    /// Real nn traces record Argmax/Argmin/ArgSort with an integer (U32)
    /// output dtype. The gate admits those because the emitted index layers
    /// model indices as EXACT f32 integers (widened outward above 2²⁴), so
    /// the dtype label adds no semantics the F32 graph mismodels.
    ///
    /// The carve-out is narrow on both axes: it does not admit float carriers
    /// (F16/BF16/F64 keep the standing refusal — their rounding story is the
    /// one the gate exists for), and it does not admit other ops.
    #[test]
    fn integer_index_outputs_of_the_arg_family_pass_the_dtype_gate() {
        let index_ops = [
            TraceOp::Argmax { dim: 1 },
            TraceOp::Argmin { dim: 1 },
            TraceOp::ArgSort {
                dim: 1,
                descending: false,
            },
        ];
        for op in index_ops {
            for dtype in [DType::U32, DType::I64, DType::I32, DType::U8] {
                let out_shape: &[usize] = if matches!(op, TraceOp::ArgSort { .. }) {
                    &[2, 4]
                } else {
                    &[2]
                };
                let mut sel = node(1, "sel", op.clone(), &[0], out_shape);
                sel.output_dtype = dtype;
                let graph = ComputationGraph::from_nodes(vec![
                    node(0, "x", TraceOp::Input, &[], &[2, 4]),
                    sel,
                ]);
                translate(&graph).unwrap_or_else(|e| {
                    panic!("{op:?} with {dtype:?} indices must pass the gate: {e:?}")
                });
            }
        }
    }

    /// The float half of the carve-out, asserted as a refusal (kept separate
    /// so the `expect_err` shape is honest).
    #[test]
    fn float_carriers_of_the_arg_family_stay_refused() {
        for dtype in [DType::F16, DType::Bf16, DType::F64] {
            let mut sel = node(1, "sel", TraceOp::Argmax { dim: 1 }, &[0], &[2]);
            sel.output_dtype = dtype;
            let graph =
                ComputationGraph::from_nodes(vec![node(0, "x", TraceOp::Input, &[], &[2, 4]), sel]);
            let err = translate(&graph).expect_err("float index carrier must fail closed");
            assert!(
                matches!(err, NyError::UnsupportedOp(ref m)
                    if m.contains("output dtype") && m.contains("only soundly supports F32")),
                "{dtype:?}: got {err:?}"
            );
        }
    }

    /// `RepeatInterleave`'s counts arrive as a translate-time-constant leaf in
    /// input position 1. Integer counts pass the value-independent-leaf gate
    /// (the translator resolves them from the weight payload and never lowers
    /// the tensor into the runtime graph) — the same justification as the
    /// IndexSelect/Gather index slot.
    #[test]
    fn integer_repeat_interleave_counts_leaf_passes_the_dtype_gate() {
        for dtype in [DType::U32, DType::I64, DType::I32, DType::U8] {
            let mut counts = node(
                1,
                "counts",
                TraceOp::ConstantWeight {
                    weight: WeightPayload::f32(vec![2.0, 2.0, 2.0], vec![3]),
                },
                &[],
                &[3],
            );
            counts.output_dtype = dtype;
            let graph = ComputationGraph::from_nodes(vec![
                node(0, "x", TraceOp::Input, &[], &[3]),
                counts,
                node(2, "r", TraceOp::RepeatInterleave { dim: 0 }, &[0, 1], &[6]),
            ]);
            translate(&graph)
                .unwrap_or_else(|e| panic!("{dtype:?} counts leaf must pass the gate: {e:?}"));
        }
    }

    /// The counts carve-out is POSITION-scoped: an integer constant feeding
    /// RepeatInterleave's DATA slot (position 0) is still refused — that
    /// value would be consumed numerically by the emitted graph.
    #[test]
    fn integer_leaf_in_the_repeat_interleave_data_slot_stays_refused() {
        let mut data = node(
            0,
            "data",
            TraceOp::ConstantWeight {
                weight: WeightPayload::f32(vec![1.0, 2.0, 3.0], vec![3]),
            },
            &[],
            &[3],
        );
        data.output_dtype = DType::U32;
        let graph = ComputationGraph::from_nodes(vec![
            data,
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
        let err = translate(&graph).expect_err("integer DATA leaf must fail closed");
        assert!(
            matches!(err, NyError::UnsupportedOp(ref m)
                if m.contains("output dtype") && m.contains("only soundly supports F32")),
            "got {err:?}"
        );
    }

    #[test]
    fn reachable_non_f32_node_is_refused_before_lowering() {
        let mut input = node(0, "x", TraceOp::Input, &[], &[1]);
        input.output_dtype = DType::F64;
        let graph =
            ComputationGraph::from_nodes(vec![input, node(1, "relu", TraceOp::Relu, &[0], &[1])]);
        let err = translate(&graph).expect_err("non-F32 trace must fail closed");
        assert!(
            matches!(err, NyError::UnsupportedOp(ref message)
                if message.contains("output dtype F64")
                    && message.contains("only soundly supports F32")),
            "got {err:?}"
        );
    }

    #[test]
    fn lossy_wide_weight_values_are_refused() {
        for data in [
            WeightData::F64(vec![16_777_217.0]),
            WeightData::I32(vec![16_777_217]),
            WeightData::I64(vec![16_777_217]),
        ] {
            let graph = ComputationGraph::from_nodes(vec![node(
                0,
                "weight",
                TraceOp::ConstantWeight {
                    weight: WeightPayload {
                        shape: vec![1],
                        data,
                    },
                },
                &[],
                &[1],
            )]);
            let err = translate(&graph).expect_err("lossy f32 weight cast must be refused");
            assert!(
                matches!(err, NyError::ModelLoad(ref message)
                    if message.contains("not exactly representable as f32")),
                "got {err:?}"
            );
        }
    }

    #[test]
    fn exactly_representable_wide_weight_values_are_accepted() {
        let graph = ComputationGraph::from_nodes(vec![node(
            0,
            "weight",
            TraceOp::ConstantWeight {
                weight: WeightPayload {
                    shape: vec![3],
                    data: WeightData::F64(vec![0.5, -2.0, 16_777_216.0]),
                },
            },
            &[],
            &[3],
        )]);
        let model = translate(&graph).expect("exact F64-to-F32 weights translate");
        assert_eq!(
            model
                .weights
                .get("layer0_trace_0_out")
                .expect("weight stored")
                .as_slice()
                .expect("contiguous"),
            &[0.5, -2.0, 16_777_216.0]
        );
    }

    #[test]
    fn constant_shape_overflow_and_resource_exhaustion_are_refused() {
        for shape in [vec![usize::MAX, 2], vec![10_000_001]] {
            let graph = ComputationGraph::from_nodes(vec![node(
                0,
                "constant",
                TraceOp::Constant { value: 1.0 },
                &[],
                &shape,
            )]);
            let err = translate(&graph).expect_err("unsafe Constant allocation must be refused");
            assert!(
                matches!(err, NyError::ModelLoad(ref message)
                    if message.contains("Constant")
                        && (message.contains("overflows")
                            || message.contains("materialization limit"))),
                "{shape:?}: got {err:?}"
            );
        }
    }

    #[test]
    fn constant_matmul_output_obeys_materialization_cap() {
        let side = 4_000;
        let graph = ComputationGraph::from_nodes(vec![
            node(
                0,
                "lhs",
                TraceOp::ConstantWeight {
                    weight: WeightPayload::f32(vec![1.0; side], vec![side, 1]),
                },
                &[],
                &[side, 1],
            ),
            node(
                1,
                "rhs",
                TraceOp::ConstantWeight {
                    weight: WeightPayload::f32(vec![1.0; side], vec![1, side]),
                },
                &[],
                &[1, side],
            ),
            node(2, "matmul", TraceOp::MatMul, &[0, 1], &[side, side]),
        ]);

        let err =
            translate(&graph).expect_err("oversized constant MatMul must fail before allocation");
        assert!(
            matches!(err, NyError::ModelLoad(ref message)
                if message.contains("constant fold of MatMul output")
                    && message.contains("materialization limit")),
            "got {err:?}"
        );
    }

    /// Constant-only chains fold away (no emitted layer, output is a weight).
    #[test]
    fn constant_unary_folds() {
        let graph = ComputationGraph::from_nodes(vec![
            node(
                0,
                "c",
                TraceOp::ConstantWeight {
                    weight: WeightPayload::f32(vec![-1.0, 2.0, -3.0], vec![3]),
                },
                &[],
                &[3],
            ),
            node(1, "r", TraceOp::Relu, &[0], &[3]),
        ]);
        let model = translate(&graph).expect("const relu folds");
        // No ReLU layer emitted — the constant was folded into a weight.
        assert_eq!(count(&model, &LayerType::ReLU), 0);
        let folded = model
            .weights
            .get("layer0_trace_1_out")
            .expect("folded constant is stored as a weight");
        assert_eq!(folded.as_slice().unwrap(), &[0.0, 2.0, 0.0]);
    }

    #[test]
    fn constant_softplus_uses_stable_large_positive_formula() {
        let graph = ComputationGraph::from_nodes(vec![
            node(
                0,
                "c",
                TraceOp::ConstantWeight {
                    weight: WeightPayload::f32(vec![100.0], vec![1]),
                },
                &[],
                &[1],
            ),
            node(1, "softplus", TraceOp::Softplus, &[0], &[1]),
        ]);
        let model = translate(&graph).expect("stable Softplus constant fold");
        let value = model
            .weights
            .get("layer0_trace_1_out")
            .expect("folded weight")
            .as_slice()
            .expect("contiguous")[0];
        assert!(value.is_finite());
        assert!(value > 99.0, "Softplus(100) was truncated: {value}");
    }

    #[test]
    fn non_finite_constant_fold_result_is_refused() {
        let graph = ComputationGraph::from_nodes(vec![
            node(
                0,
                "c",
                TraceOp::ConstantWeight {
                    weight: WeightPayload::f32(vec![100.0], vec![1]),
                },
                &[],
                &[1],
            ),
            node(1, "exp", TraceOp::Exp, &[0], &[1]),
        ]);
        let err = translate(&graph).expect_err("non-finite constant fold must fail closed");
        assert!(
            matches!(err, NyError::NumericalInstability(ref message)
                if message.contains("constant fold of Exp")
                    && message.contains("non-finite")),
            "got {err:?}"
        );
    }
}
