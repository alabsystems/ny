// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::attributes::parse_node_attributes;
use super::const_fold::common::read_tensor_i64s;
use super::const_fold::{
    int64_cast_has_raw_int64_provenance, int64_cast_has_static_reshape_shape_use,
    is_exact_materialized_int64_cast,
};
use super::fusion::{
    fold_batch_norm_into_conv_linear_with_context, try_discriminate_instance_norm,
    try_fuse_causal_softmax, try_fuse_gelu, try_fuse_gelu_tanh, try_fuse_layer_norm,
    try_fuse_logsumexp, try_fuse_merge_linear,
};
use super::{BatchNormFoldingPolicy, CustomOpRegistry};
use crate::onnx_proto;
use crate::{AttributeValue, LayerSpec, WeightStore};
use ny_build::EXPAND_LIVE_SHAPE_REFERENCE_ATTR;
use ny_core::{NyError, Result};
use ny_propagate::layers::NORMALIZATION_MIN_EPS;
use tracing::debug;
mod op_map;
mod selection;
use op_map::op_type_to_layer_type;
use selection::canonicalize_standard_topk;

// Composing authored f32 affine coefficients through f32 dot products changes
// their exact-real dyadic semantics whenever any exact coefficient is not
// representable by the rounded result.  Keep the opt-in rewrite dark until it
// has an exact accumulator and entrywise representability certificate.
const MERGE_LINEAR_EXACT_COMPOSITION_AUTHENTICATED: bool = false;

// The legacy matcher recognizes only the presence of a Trilu-shaped branch;
// it does not prove that the complete additive mask is exactly 0 on allowed
// entries and -infinity on forbidden entries.  In particular, finite masks
// such as -10000 cannot be replaced by exact zero probabilities.
const CAUSAL_SOFTMAX_MASK_AUTHENTICATED: bool = false;

// These patterns are close to the canonical GELU formulas but not identical
// under the authored f32 arithmetic.  For example, Div(x, f32::SQRT_2) has an
// exact-real coefficient different from the fused layer's f32 multiply, and
// the tanh pattern's f32 coefficients differ from the fused f64 reference.
// A tight threshold can distinguish either function.
const DECOMPOSED_ERF_GELU_SOURCE_AUTHENTICATED: bool = false;
const DECOMPOSED_TANH_GELU_SOURCE_AUTHENTICATED: bool = false;

// A last-axis LayerNorm pattern cannot become InstanceNorm merely because a
// one-dimensional scale length matches raw axis 1.  Valid channel-wise
// broadcasting and parameter layout must be authenticated explicitly before
// this semantic remap can publish.
const DECOMPOSED_INSTANCE_NORM_SOURCE_AUTHENTICATED: bool = false;

// The current perturbation radius is not an outward-rounded enclosure of the
// full QuantizeLinear -> DequantizeLinear f32 pipeline at saturation ties.
// Keep it dark until division, ties-to-even, clamp, dequant multiplication,
// and bound addition are all accounted for with directed rounding.
const QDQ_PERTURBATION_SOURCE_AUTHENTICATED: bool = false;

// Conv/Gemm -> BatchNormalization folding: governed by `BatchNormFoldingPolicy`,
// NOT by a const in this block.
//
// #bn-fold-restore (2026-08-05). A previous hard gate here
// (`BATCH_NORM_AFFINE_COMPOSITION_AUTHENTICATED: bool = false`) disabled the
// fold unconditionally, overriding the policy that is threaded all the way to
// `fold_batch_norm_into_conv_linear_with_context` and checked INSIDE it
// (`batch_norm_fold.rs`: `if policy == PreserveRaw { .. }`). The measured cost
// of that double gate, at the official cifar100 budgets on the GB10:
//
//   * resnet_large went 41 -> 61 nodes and resnet_medium 40 -> 59, so BOTH
//     crossed `CROWN_IBP_PER_NODE_THRESHOLD = 50` and lost the per-node
//     CROWN-IBP collector lane entirely (32/32 runs with the lane before,
//     0/40 after);
//   * 15 `unsat` rows on resnet_large alone went to `timeout` — every one of
//     them independently confirmed `unsat` by the official 2025 field, i.e.
//     the quarantine deleted sound proofs, not unsound ones (~11.6 normalized
//     points);
//   * the disable was SILENT: nothing is logged on the skip path at any
//     verbosity, so it shipped unnoticed.
//
// The soundness concern the gate encoded is real but bounded: the fold
// composes the BN affine into conv/gemm weights and drops the BatchNorm
// layer's `scale_err`/`bias_err` enclosure, leaving the storage rounding of
// the composed f32 weights uncertified. Composition now runs in f64 with a
// single final rounding (see `build_channel_affine`), so the uncertified
// residual is at most 0.5 ulp per weight — the same preprocessing the
// published alpha-beta-CROWN winner applies unconditionally
// (complete_verifier/onnx_opt.py, fuse_conv_and_bn), and the configuration ny
// ran for months at zero wrong verdicts corpus-wide. Callers that want the
// conservative behavior set `BatchNormFoldingPolicy::PreserveRaw` — the
// control built for exactly that. Full certificate transfer (a
// `kernel_err`/`bias_err` channel on Conv2dLayer seeded by the fold's
// per-channel interval report) remains the authenticated end-state; the
// interval machinery for it already exists behind `interval_report_enabled`.
const ONNX_BATCH_NORM_INPUT_RANK_ATTR: &str = "__onnx_batch_norm_input_rank";

/// The `to` dtype of a Cast node, or `None` when the required attribute is
/// absent (`parse::prepare` already rejects that at the protobuf boundary).
fn cast_target(node: &onnx_proto::NodeProto) -> Option<i64> {
    node.attribute
        .iter()
        .find(|attr| attr.name == "to")
        .map(|attr| attr.i_value())
}

/// Whether a Cast node's target dtype (`to` attribute) is an integer type ny
/// lowers to `LayerType::Trunc`.
///
/// ONNX TensorProto.DataType: INT32=6, INT64=7 — and ONLY those. Float->int
/// Cast is round-toward-zero for IN-RANGE values; out-of-range is explicitly
/// undefined in the ONNX Cast spec. The lowering preserves the target dtype so
/// every verdict-bearing propagation must prove its runtime input finite and
/// inside the signed destination range. UINT8=2, INT8=3, UINT16=4, INT16=5,
/// UINT32=12 and UINT64=13 have no modeled guarded lowering and stay refused;
/// a wrong enclosure there is a wrong `unsat`. BOOL(9) is not integer here
/// either — cast-to-bool is `x != 0`, not truncation.
fn cast_target_is_integer(node: &onnx_proto::NodeProto) -> bool {
    cast_target(node).is_some_and(|to| matches!(to, 6 | 7))
}

/// ONNX ops whose BOOL output is exactly `{0, 1}` on every input, so an f32
/// graph carrying their result already holds 0.0 or 1.0.
///
/// Casting such a value to BOOL is `(x != 0)` applied to a value that is
/// already 0 or 1 — i.e. the identity on every reachable value, which makes the
/// identity drop EXACT rather than merely convenient. Any sound enclosure of
/// the operand is therefore a sound enclosure of the cast, whatever relaxation
/// the comparison layer itself uses.
fn produces_boolean_valued_output(node: &onnx_proto::NodeProto) -> bool {
    // The guarantee is ONNX's: these ops are SPECIFIED to return BOOL. A
    // custom-domain op is free to reuse the name with any semantics at all, so
    // only the standard domain counts.
    if !matches!(node.domain.as_str(), "" | "ai.onnx") {
        return false;
    }
    match node.op_type.as_str() {
        "Equal" | "Greater" | "GreaterOrEqual" | "Less" | "LessOrEqual" | "And" | "Or" | "Xor"
        | "Not" | "IsNaN" | "IsInf" => true,
        // A BOOL cast of a BOOL cast is idempotent.
        "Cast" => cast_target(node) == Some(9),
        _ => false,
    }
}

/// Whether `Cast(to = BOOL)` on this node is provably an identity, given the
/// node that produces its operand (`None` when the operand is a graph input, an
/// initializer, or otherwise not produced by a node in this graph).
///
/// Fails closed: an operand ny cannot prove is already `{0,1}`-valued yields
/// `false`, and the caller then emits `LayerType::Cast`, which ny-build refuses.
fn cast_to_bool_is_identity(producer: Option<&onnx_proto::NodeProto>) -> bool {
    producer.is_some_and(produces_boolean_valued_output)
}

fn compare_op_attribute(op_type: &str) -> Option<&'static str> {
    match op_type {
        "Greater" => Some("Gt"),
        "GreaterOrEqual" => Some("Ge"),
        "Less" => Some("Lt"),
        "LessOrEqual" => Some("Le"),
        "Equal" => Some("Eq"),
        _ => None,
    }
}

pub(super) fn convert_graph_to_layers(
    nodes: &mut [onnx_proto::NodeProto],
    weights: &mut WeightStore,
    registry: &CustomOpRegistry,
    opset_imports: &std::collections::HashMap<String, i64>,
    tensor_shapes: &std::collections::HashMap<String, Vec<i64>>,
    graph_output_names: &std::collections::HashSet<String>,
    raw_int64_shape_values: &std::collections::HashSet<String>,
    merge_linear_enabled: bool,
    batch_norm_folding: BatchNormFoldingPolicy,
) -> Result<Vec<LayerSpec>> {
    use std::collections::{HashMap, HashSet};

    // Validate every standard Cast before any fusion or custom handler can
    // consume it.  Only the targets whose value semantics ny models are
    // admitted: FLOAT32 identity, guarded INT32/INT64 truncation, and BOOL
    // (which conversion admits only after proving its operand is boolean).
    for node in nodes
        .iter()
        .filter(|node| node.op_type == "Cast" && is_standard_domain(&node.domain))
    {
        let mut targets = node
            .attribute
            .iter()
            .filter(|attribute| attribute.name == "to");
        let target = targets.next().ok_or_else(|| {
            NyError::UnsupportedOp(format!(
                "ONNX Cast node '{}' is missing its required 'to' dtype",
                node.name
            ))
        })?;
        let duplicate_target = targets.next().is_some();
        let modeled_target = target.r#type == onnx_proto::attribute_type::INT
            && matches!(target.i_value(), 1 | 6 | 7 | 9);
        if node.input.len() != 1
            || node.input[0].is_empty()
            || node.output.len() != 1
            || node.output[0].is_empty()
            || target.r#type != onnx_proto::attribute_type::INT
            || duplicate_target
            || !modeled_target
        {
            return Err(NyError::UnsupportedOp(format!(
                "ONNX Cast node '{}' targets dtype {}; it survived preparation with an unsupported target/signature",
                node.name,
                target.i_value()
            )));
        }
    }

    // Fusion matchers are defined over standard ONNX semantics.  If a custom
    // domain is present, disable graph-wide fusion so a lookalike custom op
    // cannot be consumed as part of a standard pattern before its registry
    // handler (or missing-registration error) sees it.
    let standard_only_graph = nodes.iter().all(|node| is_standard_domain(&node.domain));
    let mut consumed: HashSet<usize> = if standard_only_graph {
        fold_batch_norm_into_conv_linear_with_context(
            nodes,
            weights,
            tensor_shapes,
            graph_output_names,
            batch_norm_folding,
        )
    } else {
        HashSet::new()
    };

    let mut producer_by_output: HashMap<&str, usize> = HashMap::new();
    let mut consumers_by_input: HashMap<&str, Vec<usize>> = HashMap::new();
    for (idx, node) in nodes.iter().enumerate() {
        for out in &node.output {
            producer_by_output.insert(out.as_str(), idx);
        }
        for inp in &node.input {
            consumers_by_input
                .entry(inp.as_str())
                .or_default()
                .push(idx);
        }
    }
    let mut fused_starts: HashMap<usize, LayerSpec> = HashMap::new();
    for (idx, node) in nodes.iter().enumerate() {
        if !standard_only_graph {
            break;
        }
        if consumed.contains(&idx) {
            continue;
        }
        if node_is_fully_materialized_constant(node, weights, graph_output_names) {
            continue;
        }
        if node.op_type == "QuantizeLinear" && QDQ_PERTURBATION_SOURCE_AUTHENTICATED {
            if let Some((spec, taken)) =
                try_fuse_qdq_relaxation(nodes, idx, &consumers_by_input, graph_output_names)
            {
                fused_starts.insert(idx, spec);
                consumed.extend(taken);
                continue;
            }
        }
        if merge_linear_enabled
            && MERGE_LINEAR_EXACT_COMPOSITION_AUTHENTICATED
            && node.op_type == "MatMul"
        {
            if let Some((spec, taken)) =
                try_fuse_merge_linear(nodes, idx, &consumers_by_input, weights, graph_output_names)
            {
                fused_starts.insert(idx, spec);
                consumed.extend(taken);
                continue;
            }
        }
        if node.op_type == "Erf" && DECOMPOSED_ERF_GELU_SOURCE_AUTHENTICATED {
            if let Some((start_idx, spec, taken)) = try_fuse_gelu(
                nodes,
                idx,
                &producer_by_output,
                &consumers_by_input,
                weights,
                graph_output_names,
            ) {
                fused_starts.insert(start_idx, spec);
                consumed.extend(taken);
            }
        } else if node.op_type == "Tanh" && DECOMPOSED_TANH_GELU_SOURCE_AUTHENTICATED {
            if let Some((start_idx, spec, taken)) = try_fuse_gelu_tanh(
                nodes,
                idx,
                &producer_by_output,
                &consumers_by_input,
                weights,
                graph_output_names,
            ) {
                fused_starts.insert(start_idx, spec);
                consumed.extend(taken);
            }
        } else if node.op_type == "ReduceMean" {
            if let Some((start_idx, mut spec, taken)) = try_fuse_layer_norm(
                nodes,
                idx,
                &producer_by_output,
                &consumers_by_input,
                weights,
                graph_output_names,
            ) {
                // The fused LayerNorm implementation normalizes only the last
                // axis and embeds one-dimensional affine parameters.  A
                // channel-shaped affine such as [1, C, 1] may be valid ONNX
                // broadcasting after a last-axis normalization, but it is not
                // representable by that layer.  Keep the primitive graph when
                // the exact last-axis shape cannot be authenticated.
                if !decomposed_layer_norm_affine_is_authenticated(&spec, tensor_shapes, weights) {
                    continue;
                }
                if DECOMPOSED_INSTANCE_NORM_SOURCE_AUTHENTICATED {
                    try_discriminate_instance_norm(&mut spec, tensor_shapes, weights);
                }
                fused_starts.insert(start_idx, spec);
                consumed.extend(taken);
            }
        } else if node.op_type == "Softmax" && CAUSAL_SOFTMAX_MASK_AUTHENTICATED {
            // Check if this Softmax is preceded by Trilu -> Add (causal mask pattern)
            if let Some((start_idx, spec, taken)) =
                try_fuse_causal_softmax(nodes, idx, &producer_by_output, &consumers_by_input)
            {
                debug!(
                    "Fused causal softmax pattern starting at node {}",
                    start_idx
                );
                fused_starts.insert(start_idx, spec);
                consumed.extend(taken);
            }
        } else if node.op_type == "Log" {
            if let Some((start_idx, spec, taken)) = try_fuse_logsumexp(
                nodes,
                idx,
                &producer_by_output,
                &consumers_by_input,
                graph_output_names,
            ) {
                debug!("Fused logsumexp pattern starting at node {}", start_idx);
                fused_starts.insert(start_idx, spec);
                consumed.extend(taken);
            }
        }
    }
    // A proto-level fusion is allowed to swallow Cast nodes (see
    // `fusion::causal_softmax`, which walks Trilu -> Cast -> Mul, and
    // `fusion::layer_norm`, which treats Cast as a pass-through producer). A
    // FLOAT32 Cast may be swallowed — it is an exact identity. Anything else
    // must NOT disappear this way: an integer Cast owes a `Trunc` layer and a
    // BOOL Cast owes the `{0,1}` producer proof, and a fusion that erases the
    // node erases both. Fail closed rather than let a fusion launder a
    // non-identity Cast out of the graph.
    for &idx in &consumed {
        // A fusion START still emits a layer (`fused_starts`), so it has not
        // disappeared; only the nodes a fusion swallowed are at issue.
        if fused_starts.contains_key(&idx) {
            continue;
        }
        let Some(node) = nodes.get(idx) else { continue };
        if node.op_type != "Cast" {
            continue;
        }
        if cast_target(node) != Some(1) {
            let target = cast_target(node).unwrap_or_default();
            return Err(NyError::UnsupportedOp(format!(
                "ONNX Cast node '{}' targets dtype {target} and was consumed by a proto-level \
                 fusion; only an exact FLOAT32 Cast may be fused away",
                node.name
            )));
        }
    }
    let mut layers = Vec::new();
    for (idx, node) in nodes.iter().enumerate() {
        if let Some(spec) = fused_starts.get(&idx) {
            layers.push(spec.clone());
            continue;
        }
        if consumed.contains(&idx) {
            continue;
        }
        // A non-FLOAT Cast is never an identity on a runtime data path.  The
        // sole exception is an INT64 shape cast already proven and materialized
        // exactly by constant folding.  Do not apply even that exception to an
        // authored graph output: ny's verifier API exposes FLOAT32 outputs.
        if is_exact_materialized_int64_cast(node, weights)
            && int64_cast_has_static_reshape_shape_use(
                nodes,
                idx,
                weights,
                graph_output_names,
                raw_int64_shape_values,
            )
        {
            debug!(
                "Skipping exactly materialized INT64 shape Cast '{}'",
                node.name
            );
            continue;
        }
        // Everything below lowers an integer Cast to `LayerType::Trunc`, which
        // is exact on a RUNTIME operand. Two cases must not take that route,
        // and both are the ones the pre-Trunc gate was right about:
        //
        //  * an authored graph OUTPUT — ny's verifier API exposes FLOAT32
        //    outputs, so an INT64 output would be reported through a lossy
        //    mirror;
        //  * an output the constant folder has ALREADY materialized without
        //    proving the i64 payload and its f32 mirror agree bit-for-bit
        //    (`is_exact_materialized_int64_cast`). The rounded constant is
        //    baked in before this point, so `Trunc` cannot undo it. Failing
        //    closed here is exactly what 0184b7c9 does today; a dynamic
        //    operand (cctsdb's patch-position gates) has no materialized
        //    output and is unaffected.
        if let Some(target) = cast_target(node).filter(|_| cast_target_is_integer(node)) {
            for output in node.output.iter().filter(|output| !output.is_empty()) {
                if graph_output_names.contains(output) {
                    return Err(NyError::UnsupportedOp(format!(
                        "ONNX Cast node '{}' targets dtype {target} on authored graph output \
                         '{output}'; ny's verifier API exposes FLOAT32 outputs",
                        node.name
                    )));
                }
                if weights.contains_key(output)
                    && !int64_cast_has_raw_int64_provenance(node, weights, raw_int64_shape_values)
                {
                    return Err(NyError::UnsupportedOp(format!(
                        "ONNX Cast node '{}' targets dtype {target} and was materialized by \
                         constant folding without proven raw INT64 provenance",
                        node.name
                    )));
                }
            }
        }
        // Keep this generic skip after the Cast-specific provenance gate.
        // Otherwise a forged or rounded materialized INT64 mirror can bypass
        // both the static-shape proof and the guarded runtime Trunc lowering
        // merely because the Cast's input and output are present in WeightStore.
        if node_is_fully_materialized_constant(node, weights, graph_output_names) {
            debug!(
                "Skipping exactly materialized constant node '{}' ({})",
                node.name, node.op_type
            );
            continue;
        }
        // Producer of the node's first operand, for the Cast-to-BOOL identity
        // proof. Absent for graph inputs and initializers, which fail closed.
        let operand_producer = node
            .input
            .first()
            .filter(|operand| !operand.is_empty())
            .and_then(|operand| producer_by_output.get(operand.as_str()))
            .and_then(|&producer_idx| nodes.get(producer_idx));
        if let Some(topk_layers) = canonicalize_standard_topk(node, idx, weights, opset_imports)? {
            layers.extend(topk_layers);
            continue;
        }
        if let Some(mut layer) =
            convert_node_to_layer(node, registry, opset_imports, operand_producer)?
        {
            authenticate_standard_softmax_semantics(
                node,
                &mut layer,
                opset_imports,
                tensor_shapes,
            )?;
            authenticate_direct_normalization_parameters(node, weights, tensor_shapes)?;
            if layer.layer_type == ny_core::LayerType::Expand {
                if let Some(reference) = authenticate_live_shape_expand(
                    node,
                    nodes,
                    &producer_by_output,
                    weights,
                    tensor_shapes,
                )? {
                    // Normalize input 1 from the integer Shape tensor to the
                    // live tensor whose complete shape it denotes.  ny-build
                    // admits its narrow binary lowering only with this sealed
                    // source-semantics attribute and exact normalized input.
                    layer.inputs[1] = reference.clone();
                    layer.attributes.insert(
                        EXPAND_LIVE_SHAPE_REFERENCE_ATTR.to_string(),
                        AttributeValue::String(reference),
                    );
                }
            }
            if layer.layer_type == ny_core::LayerType::BatchNorm {
                let authored_rank = infer_authored_tensor_rank(
                    &node.input[0],
                    nodes,
                    &producer_by_output,
                    tensor_shapes,
                    weights,
                    &mut HashSet::new(),
                )
                .filter(|rank| *rank >= 2)
                .ok_or_else(|| {
                    NyError::UnsupportedOp(format!(
                        "ONNX BatchNormalization node '{}' requires an authenticated authored input rank",
                        node.name
                    ))
                })?;
                layer.attributes.insert(
                    ONNX_BATCH_NORM_INPUT_RANK_ATTR.to_string(),
                    AttributeValue::Int(i64::try_from(authored_rank).map_err(|_| {
                        NyError::UnsupportedOp(format!(
                            "ONNX BatchNormalization node '{}' input rank overflows i64",
                            node.name
                        ))
                    })?),
                );
            }
            normalize_conv_rank_layer(&mut layer, weights);
            normalize_tile_layer(&mut layer, weights, tensor_shapes)?;
            layers.push(layer);
        }
    }
    Ok(layers)
}

/// Authenticate the only dynamic Expand form represented by ny's runtime
/// graph: `Expand(source[..., 1], Shape(reference[..., T]))` with identical,
/// concrete prefixes.  Merely tracing the shape tensor through the first input
/// of arbitrary shape arithmetic is not a proof—the arithmetic may select or
/// construct a different target vector—so this accepts only the complete,
/// unmodified output of a standard-domain Shape node.
fn authenticate_live_shape_expand(
    expand: &onnx_proto::NodeProto,
    nodes: &[onnx_proto::NodeProto],
    producer_by_output: &std::collections::HashMap<&str, usize>,
    weights: &WeightStore,
    tensor_shapes: &std::collections::HashMap<String, Vec<i64>>,
) -> Result<Option<String>> {
    if !is_standard_domain(&expand.domain) || expand.op_type != "Expand" {
        return Ok(None);
    }
    if expand.input.len() != 2
        || expand.input.iter().any(String::is_empty)
        || expand.output.len() != 1
        || expand.output[0].is_empty()
        || !expand.attribute.is_empty()
    {
        return Err(NyError::UnsupportedOp(format!(
            "standard ONNX Expand node '{}' requires exactly two non-empty inputs, one non-empty output, and no attributes",
            expand.name
        )));
    }

    let shape_tensor = expand.input[1].as_str();
    let Some(&shape_index) = producer_by_output.get(shape_tensor) else {
        return Ok(None);
    };
    let shape = &nodes[shape_index];
    if !is_standard_domain(&shape.domain)
        || shape.op_type != "Shape"
        || shape.input.len() != 1
        || shape.input[0].is_empty()
        || shape.output.len() != 1
        || shape.output[0] != shape_tensor
        || !shape.attribute.is_empty()
    {
        return Ok(None);
    }
    let reference = shape.input[0].as_str();
    if weights.contains_key(reference) {
        return Ok(None);
    }

    let Some(source_shape) = tensor_shapes.get(&expand.input[0]) else {
        return Ok(None);
    };
    let Some(reference_shape) = tensor_shapes.get(reference) else {
        return Ok(None);
    };
    if source_shape.is_empty() || source_shape.len() != reference_shape.len() {
        return Ok(None);
    }
    let last = source_shape.len() - 1;
    if source_shape[last] != 1 || reference_shape[last] == 0 {
        return Ok(None);
    }
    if source_shape[..last]
        .iter()
        .zip(&reference_shape[..last])
        .any(|(&source, &reference_dim)| source <= 0 || source != reference_dim)
    {
        return Ok(None);
    }

    Ok(Some(reference.to_string()))
}

/// Canonicalize the versioned ONNX Softmax/LogSoftmax axis only when ny's
/// single-axis layer represents the authored operator exactly.
///
/// Through opset 12 these operators coerce the input to a two-dimensional
/// `[product(dims[..axis]), product(dims[axis..])]` view.  That is equivalent
/// to a modern single-axis softmax only when `axis` denotes the final authored
/// dimension.  Opset 13 changed the operator to act on one axis directly and
/// changed the default from 1 to -1.
fn authenticate_standard_softmax_semantics(
    node: &onnx_proto::NodeProto,
    layer: &mut LayerSpec,
    opset_imports: &std::collections::HashMap<String, i64>,
    tensor_shapes: &std::collections::HashMap<String, Vec<i64>>,
) -> Result<()> {
    if !is_standard_domain(&node.domain)
        || !matches!(node.op_type.as_str(), "Softmax" | "LogSoftmax")
    {
        return Ok(());
    }
    if node.input.len() != 1
        || node.input[0].is_empty()
        || node.output.len() != 1
        || node.output[0].is_empty()
    {
        return Err(NyError::UnsupportedOp(format!(
            "ONNX {} node '{}' requires exactly one non-empty input and output",
            node.op_type, node.name
        )));
    }
    let version = lookup_opset_version(opset_imports, &node.domain).ok_or_else(|| {
        NyError::UnsupportedOp(format!(
            "ONNX {} node '{}' has no standard-domain opset authority",
            node.op_type, node.name
        ))
    })?;
    if version < 1 {
        return Err(NyError::UnsupportedOp(format!(
            "ONNX {} node '{}' requires opset 1 or newer, got {version}",
            node.op_type, node.name
        )));
    }

    let mut axis = None;
    for attribute in &node.attribute {
        if attribute.name != "axis"
            || attribute.r#type != onnx_proto::attribute_type::INT
            || axis.replace(attribute.i_value()).is_some()
        {
            return Err(NyError::UnsupportedOp(format!(
                "ONNX {} node '{}' has an unsupported, malformed, or duplicate '{}' attribute",
                node.op_type, node.name, attribute.name
            )));
        }
    }
    let axis = axis.unwrap_or(if version < 13 { 1 } else { -1 });
    if version < 13 && axis != -1 {
        let rank = tensor_shapes
            .get(&node.input[0])
            .map(Vec::len)
            .filter(|rank| *rank > 0)
            .ok_or_else(|| {
                NyError::UnsupportedOp(format!(
                    "legacy ONNX {} node '{}' needs an authenticated input rank to prove its flattened suffix is one dimension",
                    node.op_type, node.name
                ))
            })?;
        let rank_i64 = i64::try_from(rank).map_err(|_| {
            NyError::UnsupportedOp(format!(
                "ONNX {} node '{}' input rank does not fit i64",
                node.op_type, node.name
            ))
        })?;
        let resolved = if axis < 0 {
            axis.checked_add(rank_i64)
        } else {
            Some(axis)
        };
        if resolved != Some(rank_i64 - 1) {
            return Err(NyError::UnsupportedOp(format!(
                "legacy ONNX {} node '{}' axis {axis} on rank {rank} flattens multiple suffix dimensions, which ny's single-axis layer does not represent",
                node.op_type, node.name
            )));
        }
    }
    layer
        .attributes
        .insert("axis".to_string(), AttributeValue::Int(axis));
    Ok(())
}

/// Recover a raw ONNX tensor rank without inventing dimensions.  Lightweight
/// builds may lack value_info for Conv-produced intermediates, but these
/// operators provably preserve rank, so walking to a declared graph input is
/// sufficient to authenticate BatchNorm's optional batch stripping.
fn infer_authored_tensor_rank(
    tensor_name: &str,
    nodes: &[onnx_proto::NodeProto],
    producer_by_output: &std::collections::HashMap<&str, usize>,
    tensor_shapes: &std::collections::HashMap<String, Vec<i64>>,
    weights: &WeightStore,
    visiting: &mut std::collections::HashSet<usize>,
) -> Option<usize> {
    if let Some(shape) = tensor_shapes.get(tensor_name) {
        return Some(shape.len());
    }
    if let Some(tensor) = weights.get_integers(tensor_name) {
        return Some(tensor.ndim());
    }
    if let Some(tensor) = weights.get(tensor_name) {
        return Some(tensor.ndim());
    }
    let producer_index = *producer_by_output.get(tensor_name)?;
    if !visiting.insert(producer_index) {
        return None;
    }
    let producer = nodes.get(producer_index)?;
    let rank = match producer.op_type.as_str() {
        "Gemm" => Some(2),
        // Reshape's output rank is exactly the number of entries in its
        // authenticated INT64 shape vector. The dimensions themselves may be
        // dynamic (-1/0), but rank does not depend on resolving them.
        "Reshape" if producer.input.len() == 2 && producer.attribute.len() <= 1 => producer
            .input
            .get(1)
            .and_then(|shape_name| weights.get_integers(shape_name))
            .filter(|shape| shape.ndim() == 1)
            .map(|shape| shape.len()),
        "Flatten" if producer.input.len() == 1 => Some(2),
        "Conv" | "ConvTranspose" | "AveragePool" | "MaxPool" | "BatchNormalization" | "Relu"
        | "LeakyRelu" | "Sigmoid" | "Tanh" | "Erf" | "Exp" | "Log" | "Sqrt" | "Abs" | "Neg"
        | "Identity" | "Transpose" => producer.input.first().and_then(|input| {
            infer_authored_tensor_rank(
                input,
                nodes,
                producer_by_output,
                tensor_shapes,
                weights,
                visiting,
            )
        }),
        _ => None,
    };
    visiting.remove(&producer_index);
    rank
}

fn normalize_conv_rank_layer(layer: &mut LayerSpec, weights: &WeightStore) {
    let is_conv = layer.layer_type == ny_core::LayerType::Conv2d;
    let is_conv_transpose = layer.layer_type == ny_core::LayerType::ConvTranspose2d;
    if !is_conv && !is_conv_transpose {
        return;
    }
    let Some(kernel_name) = layer.inputs.get(1) else {
        return;
    };
    let Some(kernel) = weights.get(kernel_name) else {
        return;
    };
    if is_conv && kernel.ndim() == 3 {
        layer.layer_type = ny_core::LayerType::Conv1d;
    } else if is_conv_transpose && kernel.ndim() == 3 {
        layer.layer_type = ny_core::LayerType::ConvTranspose1d;
    }
}

fn node_is_fully_materialized_constant(
    node: &onnx_proto::NodeProto,
    weights: &WeightStore,
    graph_output_names: &std::collections::HashSet<String>,
) -> bool {
    !node.output.is_empty()
        && node.output.iter().all(|output| {
            !output.is_empty()
                && weights.contains_key(output)
                && !graph_output_names.contains(output)
        })
        && node
            .input
            .iter()
            .filter(|input| !input.is_empty())
            .all(|input| weights.contains_key(input))
}

fn normalize_tile_layer(
    layer: &mut LayerSpec,
    weights: &WeightStore,
    tensor_shapes: &std::collections::HashMap<String, Vec<i64>>,
) -> Result<()> {
    if layer.layer_type != ny_core::LayerType::Tile
        || (layer.attributes.contains_key("axis") && layer.attributes.contains_key("reps"))
    {
        return Ok(());
    }

    let repeats_name = layer.inputs.get(1).ok_or_else(|| {
        NyError::ModelLoad(format!("Tile '{}' requires ONNX repeats input", layer.name))
    })?;
    let repeats = read_tensor_i64s(weights, repeats_name).ok_or_else(|| {
        NyError::UnsupportedOp(format!(
            "Tile '{}' requires constant repeats input '{}'",
            layer.name, repeats_name
        ))
    })?;
    let repeated_axes: Vec<(usize, i64)> = repeats
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, reps)| *reps != 1)
        .collect();
    let (onnx_axis, reps) = match repeated_axes.as_slice() {
        [] => (0_usize, 1_i64),
        [(axis, reps)] if *reps > 0 => (*axis, *reps),
        [(axis, reps)] => {
            return Err(NyError::ModelLoad(format!(
                "Tile '{}': repeats on axis {} must be positive (got {})",
                layer.name, axis, reps
            )))
        }
        _ => {
            return Err(NyError::UnsupportedOp(format!(
                "Tile '{}' only supports a single non-unit repeat axis, got {:?}",
                layer.name, repeats
            )))
        }
    };

    let data_name = layer
        .inputs
        .first()
        .ok_or_else(|| NyError::ModelLoad(format!("Tile '{}' requires data input", layer.name)))?;
    let data_had_batch_axis = tensor_shapes
        .get(data_name)
        .map(|shape| shape.len() > 1)
        .unwrap_or(true);
    let internal_axis = if onnx_axis == 0 {
        if data_had_batch_axis && reps != 1 {
            return Err(NyError::UnsupportedOp(format!(
                "Tile '{}' repeats ONNX batch axis 0, which is stripped in unbatched propagation",
                layer.name
            )));
        }
        0_i64
    } else {
        i64::try_from(onnx_axis - 1).map_err(|_| {
            NyError::ModelLoad(format!(
                "Tile '{}': axis {} out of i64 range",
                layer.name, onnx_axis
            ))
        })?
    };

    layer
        .attributes
        .insert("axis".to_string(), AttributeValue::Int(internal_axis));
    layer
        .attributes
        .insert("reps".to_string(), AttributeValue::Int(reps));
    Ok(())
}

fn try_fuse_qdq_relaxation(
    nodes: &[onnx_proto::NodeProto],
    quant_idx: usize,
    consumers_by_input: &std::collections::HashMap<&str, Vec<usize>>,
    graph_output_names: &std::collections::HashSet<String>,
) -> Option<(LayerSpec, Vec<usize>)> {
    let quant = &nodes[quant_idx];
    if quant.op_type != "QuantizeLinear"
        || !matches!(quant.input.len(), 2 | 3)
        || quant.input.iter().any(String::is_empty)
        || quant.output.len() != 1
        || quant.output[0].is_empty()
        || !quant.attribute.is_empty()
    {
        return None;
    }
    let quant_output = quant.output[0].as_str();
    if graph_output_names.contains(quant_output) {
        return None;
    }
    let consumers = consumers_by_input.get(quant_output)?;
    if consumers.len() != 1 {
        return None;
    }
    let dequant_idx = consumers[0];
    let dequant = &nodes[dequant_idx];
    if dequant.op_type != "DequantizeLinear"
        || !matches!(dequant.input.len(), 2 | 3)
        || dequant.input.iter().any(String::is_empty)
        || dequant.output.len() != 1
        || dequant.output[0].is_empty()
        || !dequant.attribute.is_empty()
    {
        return None;
    }
    if dequant.input.first().map(String::as_str) != Some(quant_output) {
        return None;
    }
    if quant.input.get(1) != dequant.input.get(1) {
        return None;
    }
    let quant_zero_point = quant.input.get(2).filter(|name| !name.is_empty());
    let dequant_zero_point = dequant.input.get(2).filter(|name| !name.is_empty());
    if quant_zero_point != dequant_zero_point {
        return None;
    }

    let name = if dequant.name.is_empty() {
        dequant.output.first().cloned().unwrap_or_default()
    } else {
        dequant.name.clone()
    };
    let mut inputs = vec![quant.input[0].clone(), quant.input[1].clone()];
    if let Some(zero_point) = quant_zero_point {
        inputs.push(zero_point.clone());
    }
    let mut attributes = std::collections::HashMap::new();
    attributes.insert("qdq_relaxation".to_string(), AttributeValue::Int(1));

    Some((
        LayerSpec {
            name,
            layer_type: ny_core::LayerType::QuantizeLinear,
            inputs,
            outputs: dequant.output.clone(),
            weights: None,
            attributes,
        },
        vec![dequant_idx],
    ))
}

fn is_standard_domain(domain: &str) -> bool {
    // Core ONNX operators live only in the empty / ai.onnx domain. The
    // ai.onnx.ml operator set is a distinct schema namespace; accepting a
    // same-named lookalike there as a core neural-network op is a semantic
    // substitution, not compatibility.
    matches!(domain, "" | "ai.onnx")
}

fn lookup_opset_version(
    opset_imports: &std::collections::HashMap<String, i64>,
    domain: &str,
) -> Option<i64> {
    if let Some(version) = opset_imports.get(domain) {
        return Some(*version);
    }
    if domain.is_empty() {
        if let Some(version) = opset_imports.get("ai.onnx") {
            return Some(*version);
        }
    } else if domain == "ai.onnx" {
        if let Some(version) = opset_imports.get("") {
            return Some(*version);
        }
    }
    None
}

/// Canonical input/output names for a direct normalization operator after
/// optional empty ONNX placeholders have been removed.
struct DirectNormalizationIo {
    inputs: Vec<String>,
    outputs: Vec<String>,
}

/// Validate the inference-only modern BatchNormalization subset represented
/// by [`ny_propagate::layers::BatchNormLayer`].  Opset 9 removed the legacy
/// `is_test`/`spatial` ambiguity; opset 14 introduced `training_mode` and
/// reduced the optional output list from five to three.  In either schema,
/// inference publishes only Y, though ONNX permits empty trailing optional
/// placeholders in the serialized node.
fn validate_batch_normalization_schema(
    node: &onnx_proto::NodeProto,
    opset_version: Option<i64>,
) -> Result<Option<DirectNormalizationIo>> {
    if node.op_type != "BatchNormalization" {
        return Ok(None);
    }
    if !matches!(node.domain.as_str(), "" | "ai.onnx") {
        return Err(NyError::UnsupportedOp(format!(
            "ONNX BatchNormalization node '{}' uses unsupported domain '{}'",
            node.name, node.domain
        )));
    }
    let version = opset_version.ok_or_else(|| {
        NyError::UnsupportedOp(format!(
            "ONNX BatchNormalization node '{}' has no main-domain opset import",
            node.name
        ))
    })?;
    if version < 9 {
        return Err(NyError::UnsupportedOp(format!(
            "ONNX BatchNormalization node '{}' requires opset 9 or newer for unambiguous inference semantics (model imports {})",
            node.name, version
        )));
    }
    if node.input.len() != 5 || node.input.iter().any(String::is_empty) {
        return Err(NyError::UnsupportedOp(format!(
            "ONNX BatchNormalization node '{}' requires exactly five non-empty inputs, got {:?}",
            node.name, node.input
        )));
    }
    let maximum_outputs = if version >= 14 { 3 } else { 5 };
    if node.output.is_empty()
        || node.output.len() > maximum_outputs
        || node.output[0].is_empty()
        || node.output.iter().skip(1).any(|output| !output.is_empty())
    {
        return Err(NyError::UnsupportedOp(format!(
            "ONNX BatchNormalization node '{}' has unsupported inference outputs {:?} for opset {}; expected Y followed only by up to {} empty optional placeholders",
            node.name,
            node.output,
            version,
            maximum_outputs - 1
        )));
    }

    let mut seen = std::collections::HashSet::new();
    for attribute in &node.attribute {
        if !seen.insert(attribute.name.as_str()) {
            return Err(NyError::UnsupportedOp(format!(
                "ONNX BatchNormalization node '{}' has duplicate '{}' attributes",
                node.name, attribute.name
            )));
        }
        match attribute.name.as_str() {
            "epsilon"
                if attribute.r#type == onnx_proto::attribute_type::FLOAT
                    && attribute.f_value().is_finite()
                    && attribute.f_value() >= 0.0 => {}
            "momentum"
                if attribute.r#type == onnx_proto::attribute_type::FLOAT
                    && attribute.f_value().is_finite() => {}
            "training_mode"
                if version >= 14
                    && attribute.r#type == onnx_proto::attribute_type::INT
                    && attribute.i_value() == 0 => {}
            "epsilon" | "momentum" | "training_mode" => {
                return Err(NyError::UnsupportedOp(format!(
                    "ONNX BatchNormalization node '{}' has unsupported '{}' value, type, or opset for inference",
                    node.name, attribute.name
                )));
            }
            _ => {
                return Err(NyError::UnsupportedOp(format!(
                    "ONNX BatchNormalization node '{}' has unsupported attribute '{}' in opset {}",
                    node.name, attribute.name, version
                )));
            }
        }
    }

    Ok(Some(DirectNormalizationIo {
        inputs: node.input.clone(),
        outputs: vec![node.output[0].clone()],
    }))
}

/// Validate the exact direct-normalization subset represented by ny's fused
/// layers.  ONNX LayerNormalization and RMSNormalization can normalize an
/// arbitrary suffix beginning at `axis`; ny currently represents only the
/// final axis.  Optional statistic outputs and non-f32 accumulation are also
/// not represented, so those forms must fail closed rather than be silently
/// mapped to the primary output.
fn validate_direct_normalization_schema(
    node: &onnx_proto::NodeProto,
    opset_version: Option<i64>,
) -> Result<Option<DirectNormalizationIo>> {
    if let Some(io) = validate_batch_normalization_schema(node, opset_version)? {
        return Ok(Some(io));
    }
    let (minimum_opset, minimum_inputs, maximum_inputs, maximum_outputs) =
        match node.op_type.as_str() {
            "LayerNormalization" => (17_i64, 2_usize, 3_usize, 3_usize),
            // Legacy experimental ONNX operator.  It has a second optional
            // inv_std_var output, which ny deliberately does not expose.
            "SimplifiedLayerNormalization" => (1_i64, 2_usize, 2_usize, 2_usize),
            "RMSNormalization" => (23_i64, 2_usize, 2_usize, 1_usize),
            "InstanceNormalization" => (1_i64, 3_usize, 3_usize, 1_usize),
            "GroupNormalization" => (21_i64, 3_usize, 3_usize, 1_usize),
            _ => return Ok(None),
        };

    if !matches!(node.domain.as_str(), "" | "ai.onnx") {
        return Err(NyError::UnsupportedOp(format!(
            "ONNX {} node '{}' uses unsupported domain '{}'",
            node.op_type, node.name, node.domain
        )));
    }

    let version = opset_version.ok_or_else(|| {
        NyError::UnsupportedOp(format!(
            "ONNX {} node '{}' has no main-domain opset import",
            node.op_type, node.name
        ))
    })?;
    if version < minimum_opset {
        return Err(NyError::UnsupportedOp(format!(
            "ONNX {} node '{}' requires opset {} or newer (model imports {})",
            node.op_type, node.name, minimum_opset, version
        )));
    }

    if !(minimum_inputs..=maximum_inputs).contains(&node.input.len())
        || node.input[..minimum_inputs].iter().any(String::is_empty)
    {
        return Err(NyError::UnsupportedOp(format!(
            "ONNX {} node '{}' has unsupported input signature {:?}",
            node.op_type, node.name, node.input
        )));
    }

    if node.output.is_empty() || node.output.len() > maximum_outputs || node.output[0].is_empty() {
        return Err(NyError::UnsupportedOp(format!(
            "ONNX {} node '{}' has unsupported output signature {:?}",
            node.op_type, node.name, node.output
        )));
    }
    if node.output.iter().skip(1).any(|output| !output.is_empty()) {
        return Err(NyError::UnsupportedOp(format!(
            "ONNX {} node '{}' requests optional statistic outputs that ny does not represent",
            node.op_type, node.name
        )));
    }

    let mut seen = std::collections::HashSet::new();
    for attribute in &node.attribute {
        if !seen.insert(attribute.name.as_str()) {
            return Err(NyError::UnsupportedOp(format!(
                "ONNX {} node '{}' has duplicate '{}' attributes",
                node.op_type, node.name, attribute.name
            )));
        }
        match (node.op_type.as_str(), attribute.name.as_str()) {
            (
                "LayerNormalization"
                | "SimplifiedLayerNormalization"
                | "RMSNormalization",
                "axis",
            )
                if attribute.r#type == onnx_proto::attribute_type::INT
                    && attribute.i_value() == -1 => {}
            (
                "LayerNormalization"
                | "SimplifiedLayerNormalization"
                | "RMSNormalization",
                "axis",
            ) => {
                return Err(NyError::UnsupportedOp(format!(
                    "ONNX {} node '{}' uses unsupported axis attribute; only INT -1 is represented",
                    node.op_type, node.name
                )))
            }
            (
                "LayerNormalization"
                | "SimplifiedLayerNormalization"
                | "RMSNormalization"
                | "InstanceNormalization"
                | "GroupNormalization",
                "epsilon",
            )
                if attribute.r#type == onnx_proto::attribute_type::FLOAT
                    && attribute.f_value().is_finite()
                    && attribute.f_value() >= NORMALIZATION_MIN_EPS => {}
            (
                "LayerNormalization"
                | "SimplifiedLayerNormalization"
                | "RMSNormalization"
                | "InstanceNormalization"
                | "GroupNormalization",
                "epsilon",
            ) => {
                return Err(NyError::UnsupportedOp(format!(
                    "ONNX {} node '{}' has unsupported epsilon; expected a finite FLOAT of at least {}",
                    node.op_type, node.name, NORMALIZATION_MIN_EPS
                )))
            }
            (
                "LayerNormalization"
                | "SimplifiedLayerNormalization"
                | "RMSNormalization"
                | "GroupNormalization",
                "stash_type",
            )
                if attribute.r#type == onnx_proto::attribute_type::INT && attribute.i_value() == 1 => {}
            (
                "LayerNormalization"
                | "SimplifiedLayerNormalization"
                | "RMSNormalization"
                | "GroupNormalization",
                "stash_type",
            ) => {
                return Err(NyError::UnsupportedOp(format!(
                    "ONNX {} node '{}' uses unsupported stash_type; only FLOAT accumulation (INT 1) is represented",
                    node.op_type, node.name
                )))
            }
            ("GroupNormalization", "num_groups")
                if attribute.r#type == onnx_proto::attribute_type::INT && attribute.i_value() > 0 => {}
            ("GroupNormalization", "num_groups") => {
                return Err(NyError::UnsupportedOp(format!(
                    "ONNX GroupNormalization node '{}' requires a positive INT num_groups",
                    node.name
                )))
            }
            _ => {
                return Err(NyError::UnsupportedOp(format!(
                    "ONNX {} node '{}' has unsupported attribute '{}'",
                    node.op_type, node.name, attribute.name
                )))
            }
        }
    }

    if node.op_type == "GroupNormalization" && !seen.contains("num_groups") {
        return Err(NyError::UnsupportedOp(format!(
            "ONNX GroupNormalization node '{}' is missing required num_groups",
            node.name
        )));
    }

    let mut inputs = node.input[..minimum_inputs].to_vec();
    if node.op_type == "LayerNormalization"
        && node.input.get(2).is_some_and(|input| !input.is_empty())
    {
        inputs.push(node.input[2].clone());
    }
    Ok(Some(DirectNormalizationIo {
        inputs,
        outputs: vec![node.output[0].clone()],
    }))
}

/// Direct fused normalization requires static, constant, one-dimensional
/// affine parameters matching the final input dimension.  The ny-build
/// converter historically substituted ones/zeros when a named parameter was
/// not found, which is suitable for internally generated specs but would
/// silently change an authored ONNX operator with a runtime Scale/B input.
fn authenticate_direct_normalization_parameters(
    node: &onnx_proto::NodeProto,
    weights: &WeightStore,
    tensor_shapes: &std::collections::HashMap<String, Vec<i64>>,
) -> Result<()> {
    // A registered custom-domain handler owns that operator's schema and
    // semantics even when its op_type happens to collide with a core ONNX
    // normalization name.
    if !is_standard_domain(&node.domain) {
        return Ok(());
    }
    if !matches!(
        node.op_type.as_str(),
        "BatchNormalization"
            | "LayerNormalization"
            | "SimplifiedLayerNormalization"
            | "RMSNormalization"
            | "InstanceNormalization"
            | "GroupNormalization"
    ) {
        return Ok(());
    }

    let input_name = node
        .input
        .first()
        .filter(|name| !name.is_empty())
        .ok_or_else(|| {
            NyError::UnsupportedOp(format!(
                "ONNX {} node '{}' requires a non-empty data input",
                node.op_type, node.name
            ))
        })?;
    let input_shape = tensor_shapes.get(input_name);
    let affine_dim = if node.op_type == "BatchNormalization" {
        // Shape inference is optional in lightweight builds and may omit a
        // Conv-produced intermediate.  The raw FLOAT32 scale is itself the
        // schema's authoritative `[C]` declaration; authenticate every other
        // statistic against it and additionally cross-check raw axis 1 whenever
        // the activation shape is available.
        let scale_name = node
            .input
            .get(1)
            .filter(|name| !name.is_empty())
            .ok_or_else(|| {
                NyError::UnsupportedOp(format!(
                    "ONNX BatchNormalization node '{}' requires a non-empty Scale input",
                    node.name
                ))
            })?;
        let scale = weights.get(scale_name).ok_or_else(|| {
            NyError::UnsupportedOp(format!(
                "ONNX BatchNormalization node '{}' requires constant Scale input '{}'",
                node.name, scale_name
            ))
        })?;
        let [channels] = scale.shape() else {
            return Err(NyError::UnsupportedOp(format!(
                "ONNX BatchNormalization node '{}' requires one-dimensional Scale shape [C], got {:?}",
                node.name,
                scale.shape()
            )));
        };
        if *channels == 0 {
            return Err(NyError::UnsupportedOp(format!(
                "ONNX BatchNormalization node '{}' has zero channels",
                node.name
            )));
        }
        if let Some(input_shape) = input_shape {
            let input_channels = input_shape
                .get(1)
                .and_then(|dimension| usize::try_from(*dimension).ok())
                .filter(|dimension| *dimension > 0)
                .ok_or_else(|| {
                    NyError::UnsupportedOp(format!(
                        "ONNX BatchNormalization node '{}' requires a positive channel dimension at raw axis 1, got {:?}",
                        node.name, input_shape
                    ))
                })?;
            if input_channels != *channels {
                return Err(NyError::UnsupportedOp(format!(
                    "ONNX BatchNormalization node '{}' Scale has {} channels but input '{}' has {} at raw axis 1",
                    node.name, channels, input_name, input_channels
                )));
            }
        }
        *channels
    } else {
        let input_shape = input_shape.ok_or_else(|| {
            NyError::UnsupportedOp(format!(
                "ONNX {} node '{}' requires a known input shape for '{}'",
                node.op_type, node.name, input_name
            ))
        })?;
        match node.op_type.as_str() {
            "LayerNormalization" | "SimplifiedLayerNormalization" | "RMSNormalization" => {
                input_shape.last().copied()
            }
            "InstanceNormalization" | "GroupNormalization" => input_shape.get(1).copied(),
            _ => None,
        }
        .and_then(|dimension| usize::try_from(dimension).ok())
        .filter(|dimension| *dimension > 0)
        .ok_or_else(|| {
            NyError::UnsupportedOp(format!(
                "ONNX {} node '{}' requires a known positive affine dimension for input '{}' with shape {:?}",
                node.op_type, node.name, input_name, input_shape
            ))
        })?
    };

    let authenticate_parameter = |index: usize, label: &str| -> Result<()> {
        let parameter_name = node
            .input
            .get(index)
            .filter(|name| !name.is_empty())
            .ok_or_else(|| {
                NyError::UnsupportedOp(format!(
                    "ONNX {} node '{}' requires a non-empty {} input at index {}",
                    node.op_type, node.name, label, index
                ))
            })?;
        let parameter = weights.get(parameter_name).ok_or_else(|| {
            NyError::UnsupportedOp(format!(
                "ONNX {} node '{}' requires constant {} input '{}'",
                node.op_type, node.name, label, parameter_name
            ))
        })?;
        if parameter.shape() != [affine_dim] {
            return Err(NyError::UnsupportedOp(format!(
                "ONNX {} node '{}' requires one-dimensional {} shape [{}], got {:?}",
                node.op_type,
                node.name,
                label,
                affine_dim,
                parameter.shape()
            )));
        }
        Ok(())
    };

    match node.op_type.as_str() {
        "BatchNormalization" => {
            for (index, label) in [(1, "Scale"), (2, "B"), (3, "mean"), (4, "var")] {
                authenticate_parameter(index, label)?;
            }
        }
        "LayerNormalization" => {
            authenticate_parameter(1, "Scale")?;
            if node.input.get(2).is_some_and(|input| !input.is_empty()) {
                authenticate_parameter(2, "B")?;
            }
        }
        "InstanceNormalization" | "GroupNormalization" => {
            authenticate_parameter(1, "Scale")?;
            authenticate_parameter(2, "bias")?;
        }
        "SimplifiedLayerNormalization" | "RMSNormalization" => {
            authenticate_parameter(1, "Scale")?;
        }
        _ => unreachable!("normalization operator filtered above"),
    }

    if node.op_type == "GroupNormalization" {
        let input_shape = input_shape.expect("non-BatchNorm shape authenticated above");
        if input_shape.len() != 3 {
            return Err(NyError::UnsupportedOp(format!(
                "ONNX GroupNormalization node '{}' supports authored rank 3 [N,C,T], got {:?}",
                node.name, input_shape
            )));
        }
        let num_groups = node
            .attribute
            .iter()
            .find(|attribute| attribute.name == "num_groups")
            .and_then(|attribute| usize::try_from(attribute.i_value()).ok())
            .ok_or_else(|| {
                NyError::UnsupportedOp(format!(
                    "ONNX GroupNormalization node '{}' has invalid num_groups",
                    node.name
                ))
            })?;
        if !affine_dim.is_multiple_of(num_groups) {
            return Err(NyError::UnsupportedOp(format!(
                "ONNX GroupNormalization node '{}' has {} channels not divisible by {} groups",
                node.name, affine_dim, num_groups
            )));
        }
    }
    Ok(())
}

fn decomposed_layer_norm_affine_is_authenticated(
    spec: &LayerSpec,
    tensor_shapes: &std::collections::HashMap<String, Vec<i64>>,
    weights: &WeightStore,
) -> bool {
    if spec.layer_type != ny_core::LayerType::LayerNorm
        || spec.inputs.len() != 3
        || spec.outputs.len() != 1
    {
        return false;
    }
    let Some(last_dim) = tensor_shapes
        .get(&spec.inputs[0])
        .and_then(|shape| shape.last())
        .and_then(|dimension| usize::try_from(*dimension).ok())
        .filter(|dimension| *dimension > 0)
    else {
        return false;
    };
    spec.inputs[1..].iter().all(|name| {
        weights
            .get(name)
            .is_some_and(|weight| weight.shape() == [last_dim])
    })
}

/// `operand_producer` is the node producing `node.input[0]`, when one exists in
/// this graph. Only the Cast-to-BOOL admission consults it; every other op
/// converts from the node alone, so callers without graph context may pass
/// `None` and get the fail-closed reading of a BOOL cast.
fn convert_node_to_layer(
    node: &onnx_proto::NodeProto,
    registry: &CustomOpRegistry,
    opset_imports: &std::collections::HashMap<String, i64>,
    operand_producer: Option<&onnx_proto::NodeProto>,
) -> Result<Option<LayerSpec>> {
    let op_type = &node.op_type;
    let domain = node.domain.as_str();
    let opset_version = lookup_opset_version(opset_imports, domain);
    let name = if node.name.is_empty() {
        node.output.first().cloned().unwrap_or_default()
    } else {
        node.name.clone()
    };

    for handler in registry.handlers() {
        if let Some(layer) = handler.try_convert_with_context(node, opset_version) {
            return Ok(Some(layer));
        }
        if handler.supports_with_context(op_type, domain, opset_version) {
            return Err(NyError::UnsupportedConfiguration(format!(
                "Custom op handler {} claimed support for domain=\"{}\", op_type=\"{}\" but returned None",
                handler.name(),
                domain,
                op_type
            )));
        }
    }

    if !domain.is_empty() && !is_standard_domain(domain) {
        let version = opset_version
            .map(|version| version.to_string())
            .unwrap_or_else(|| "unknown".to_string());
        if opset_version.is_none() {
            return Err(NyError::UnsupportedConfiguration(format!(
                "Custom op missing opset import: domain=\"{}\", op_type=\"{}\", opset_version={}. \
Hint: add an opset import for the domain or register a CustomOpHandler via OnnxLoadConfig::new",
                domain, op_type, version
            )));
        }
        return Err(NyError::UnsupportedConfiguration(format!(
            "Custom op missing registration: domain=\"{}\", op_type=\"{}\", opset_version={}. \
Hint: register a CustomOpHandler via OnnxLoadConfig::new",
            domain, op_type, version
        )));
    }

    if op_type == "Cast" {
        let mut target_attributes = node
            .attribute
            .iter()
            .filter(|attribute| attribute.name == "to");
        let target_attribute = target_attributes.next().ok_or_else(|| {
            NyError::UnsupportedOp(format!(
                "ONNX Cast node '{name}' is missing its required 'to' dtype"
            ))
        })?;
        if target_attributes.next().is_some() {
            return Err(NyError::UnsupportedOp(format!(
                "ONNX Cast node '{name}' has duplicate 'to' dtype attributes"
            )));
        }
        if target_attribute.r#type != onnx_proto::attribute_type::INT {
            return Err(NyError::UnsupportedOp(format!(
                "ONNX Cast node '{name}' has a non-INT 'to' dtype attribute"
            )));
        }
        let target = target_attribute.i_value();
        // Admit EXACTLY the three target families whose value semantics ny
        // reproduces, and fail closed on the rest (#cctsdb B1).
        //
        // * INT32/INT64 target: the cast TRUNCATES toward zero, so the
        //   historical identity drop was unsound for fractional values
        //   (trunc(0.5) = 0 is not in [0.5, 62] — cctsdb_yolo_2023 gates patch
        //   positions through exactly such casts). Lower to a target-carrying
        //   guarded Trunc. Truncation is monotone and exact only after the
        //   runtime pre-activation is proved finite and within the destination
        //   range; ny-build retains `to=INT32/INT64` and all verdict-bearing
        //   propagation paths enforce that obligation. The cell-enumeration
        //   driver (ny-cli beta_crown::cell_enum) recognizes these nodes as the
        //   piecewise-constant gates that make the model decidable cell by cell.
        //   Constant operands never reach here — load-time const-fold and the
        //   builder's constant pre-evaluation fold them, applying the same trunc.
        // * FLOAT32 target: exact identity, all bound math is already f32.
        // * BOOL target: `x != 0`, an identity only on a `{0,1}`-valued operand.
        //   Admitted just for that provable case; everything else falls through
        //   to the fail-closed branch below.
        //
        // The NARROW integer set is deliberate. `trunc` is the ONNX float->int
        // semantics only for IN-RANGE values; out-of-range is explicitly
        // undefined in the Cast spec and ONNX Runtime's answer (wraparound) is
        // NOT trunc. INT32/INT64 therefore carry explicit finite/range guards;
        // no model-specific reachability assumption substitutes for that proof.
        // The narrow and unsigned types have no corresponding guarded lowering,
        // and a wrong enclosure there is a wrong `unsat`, so they stay refused.
        //
        // Reduced-precision floats (f16 2^-11, bf16 2^-8 relative error) and
        // DOUBLE round or widen in ways ny does not model, and every exotic
        // dtype (string/complex/float8/4-bit) has no f32 reading at all. Those
        // are already refused at the protobuf boundary by
        // `parse::prepare::cast_target_semantics_are_modeled`, ahead of constant
        // folding so a fold cannot launder them into a plain FLOAT constant.
        // Keep the refusal here too as defence in depth: emitting
        // `LayerType::Cast` makes ny-build's `convert_layer` return
        // `UnsupportedOp`, so a strict build surfaces the error and a permissive
        // graph build degrades the node to a sound OpaqueSkip [-inf, +inf].
        if cast_target_is_integer(node) {
            debug!("Cast op '{}' has integer target; lowering to Trunc", name);
            return Ok(Some(LayerSpec {
                name,
                layer_type: ny_core::LayerType::Trunc,
                inputs: node.input.clone(),
                outputs: node.output.clone(),
                weights: None,
                attributes: parse_node_attributes(node),
            }));
        }
        let bool_target_is_identity = target == 9 && cast_to_bool_is_identity(operand_producer);
        if target != 1 && !bool_target_is_identity {
            debug!(
                "Cast op '{}' targets dtype {} with unmodeled semantics; \
                 refusing identity drop (fail closed)",
                name, target
            );
            return Ok(Some(LayerSpec {
                name,
                layer_type: ny_core::LayerType::Cast,
                inputs: node.input.clone(),
                outputs: node.output.clone(),
                weights: None,
                attributes: parse_node_attributes(node),
            }));
        }
    }

    let direct_normalization_io = validate_direct_normalization_schema(node, opset_version)?;
    let (layer_type, supported) = op_type_to_layer_type(op_type, &name)?;

    if !supported {
        return Ok(None);
    }

    let mut attributes = parse_node_attributes(node);
    if let Some(compare_op) = compare_op_attribute(op_type) {
        attributes.insert(
            "compare_op".to_string(),
            AttributeValue::String(compare_op.to_string()),
        );
    }

    let (inputs, outputs) = direct_normalization_io
        .map(|io| (io.inputs, io.outputs))
        .unwrap_or_else(|| (node.input.clone(), node.output.clone()));

    Ok(Some(LayerSpec {
        name,
        layer_type,
        inputs,
        outputs,
        weights: None,
        attributes,
    }))
}

#[cfg(test)]
mod tests;
