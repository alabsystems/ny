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
//! Every [`TraceOp::ToDtype`] is refused. The wire format records only the
//! target dtype, so even F32/F64 may be a precision-losing cast; the bridge
//! cannot prove that any target is an identity. Translation also rejects every
//! reachable node whose declared output dtype is not F32, because the emitted
//! graph contract is F32.
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
//! `mod.rs` and are visible to every child module via `super::`.

mod ops_attention;
mod ops_conv_linear;
mod ops_core;
mod ops_kokoro;
mod ops_lstm;
mod ops_misc;
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
    let analysis = analyze_graph(graph)?;
    validate_single_input_mode(&analysis)?;
    translate_impl(graph, false, &analysis)
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
    let analysis = analyze_graph(graph)?;
    translate_impl(graph, true, &analysis)
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
        if node.output_dtype != DType::F32 {
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

/// Shared implementation for single-input and multi-input translation.
fn translate_impl(
    graph: &ComputationGraph,
    enable_multi_input: bool,
    analysis: &GraphAnalysis,
) -> Result<Translation> {
    let reachable = &analysis.reachable;
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
            tensor_producer.insert(output_tensor, slice_out);

            node_names.insert(node.id.get(), name);
            continue;
        }

        let output = translate_node(node, &name, &node_names, &mut ctx)?;

        // Register the input tensor spec for single-input mode.
        if matches!(node.op, TraceOp::Input) {
            let shape = shape_to_i64(&node.output_shape, "Input")?;
            input_specs.push((format!("{name}_in"), shape));
        }

        // Record the output tensor shape from the traced node.
        let out_shape = shape_to_i64(&node.output_shape, "output shape")?;
        ctx.tensor_shapes.insert(format!("{name}_out"), out_shape);

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
    })
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
    /// Whether the trace uses the batched convention (batch=1 at dim 0).
    is_batched: bool,
}

impl Ctx {
    fn new(is_batched: bool) -> Self {
        Self {
            weights: WeightStore::new(),
            constant_tensors: HashSet::new(),
            tensor_shapes: HashMap::new(),
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

/// Validate that eps is positive and finite, returning the f32 cast.
fn validate_eps(eps: f64, context: &str) -> Result<f32> {
    if !eps.is_finite() || eps <= 0.0 {
        return Err(NyError::NumericalInstability(format!(
            "{context}: eps must be positive and finite, got {eps}"
        )));
    }
    let eps_f32 = eps as f32;
    if !eps_f32.is_finite() || eps_f32 <= 0.0 {
        return Err(NyError::NumericalInstability(format!(
            "{context}: eps overflows/underflows f32 ({eps} -> {eps_f32})"
        )));
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

        // -- Dtype cast (source dtype absent, so every target fails closed) --
        TraceOp::ToDtype { target_dtype } => {
            ops_core::translate_to_dtype(*target_dtype, &input_tensors)
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
        TraceOp::Sdpa { .. } | TraceOp::SdpaCausal { .. } | TraceOp::RotaryEmbedding { .. } => {
            ops_attention::translate_attention(
                node,
                name,
                &input_tensors,
                &output_tensor,
                node_names,
                ctx,
            )
        }

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

        // -- Misc elementwise / indexing / masking family --
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
        // semantics gets no vacuous fallback — refusal forces a real
        // lowering instead of silently destroying precision.
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

        // -- Genuinely refused forever: data-dependent output shape/routing
        //    (Topk / Argmax / Argmin / ArgSort / Sort / Scatter), pipeline
        //    markers (SegmentBoundary), and the fused MultiHeadAttention
        //    escape. Sound-by-construction refusal; explicit arm
        //    (no wildcard) so adding a TraceOp variant forces a routing
        //    decision here at compile time. --
        TraceOp::Topk { .. }
        | TraceOp::Argmax { .. }
        | TraceOp::Argmin { .. }
        | TraceOp::ArgSort { .. }
        | TraceOp::Sort { .. }
        | TraceOp::Scatter { .. }
        | TraceOp::MultiHeadAttention { .. }
        | TraceOp::SegmentBoundary { .. } => Err(NyError::UnsupportedOp(format!(
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

    /// An unhandled op (Topk) returns UnsupportedOp naming the op.
    #[test]
    fn unhandled_op_returns_unsupported() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[8]),
            node(1, "topk", TraceOp::Topk { k: 3, dim: 0 }, &[0], &[3]),
        ]);
        let err = translate(&graph).expect_err("Topk is unsupported");
        match err {
            NyError::UnsupportedOp(msg) => {
                assert!(msg.contains("Topk"), "error names the op: {msg}");
            }
            other => panic!("expected UnsupportedOp, got {other:?}"),
        }
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

    #[test]
    fn exp_above_f32_threshold_fails_closed_without_clipping() {
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
        let err = network
            .propagate_ibp(&input)
            .expect_err("Exp(100) must fail closed instead of becoming Exp(88)");
        assert!(
            matches!(err, NyError::NumericalInstability(ref message)
                if message.contains("Exp IBP") && message.contains("overflow threshold")),
            "got {err:?}"
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
            node(1, "topk", TraceOp::Topk { k: 3, dim: 0 }, &[0], &[3]),
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

    #[test]
    fn every_dtype_cast_is_refused_when_source_dtype_is_unknown() {
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
            let graph = ComputationGraph::from_nodes(vec![
                node(0, "x", TraceOp::Input, &[], &[4]),
                node(1, "cast", TraceOp::ToDtype { target_dtype }, &[0], &[4]),
            ]);
            let err = translate_with_metadata(&graph)
                .expect_err("cast with unknown source dtype must fail closed");
            assert!(
                matches!(err, NyError::UnsupportedOp(ref m)
                    if m.contains("ToDtype") && m.contains("source dtype")),
                "{target_dtype:?}: got {err:?}"
            );
        }
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
