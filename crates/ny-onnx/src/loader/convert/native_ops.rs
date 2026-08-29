// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! ONNX operators whose ny layer already exists and is reached by the trace
//! bridge, canonicalized into that same layer from the ONNX front end.
//!
//! Both canonicalizers here close an ASYMMETRY, not a gap in ny's runtime: the
//! layers (`LayerType::Triu` / `Tril`, `LayerType::ScatterAdd`) are built,
//! propagated and tested; only the ONNX intake refused to reach them, so a
//! model expressing the identical function in ONNX bytes hard-errored while
//! the same function recorded through a trace loaded.
//!
//! Each canonicalizer is TOTAL for its op type in the standard domain: it
//! returns `Ok(Some(spec))` when the authored schema is one ny represents
//! exactly, and `Err` naming the specific configuration otherwise. It never
//! returns `Ok(None)` for its own op, so nothing falls through to
//! `op_map`'s generic "not supported" arm and no unrepresented configuration
//! can be silently admitted.

use super::{lookup_opset_version, parse_node_attributes};
use crate::loader::const_fold::common::read_tensor_i64s_and_shape;
use crate::onnx_proto::{self, NodeProto};
use crate::{AttributeValue, LayerSpec, WeightStore};
use ny_core::{LayerType, NyError, Result};
use std::collections::HashMap;

/// Core ONNX operators live only in the empty / `ai.onnx` domain; a
/// same-named lookalike elsewhere belongs to its registered handler.
fn is_standard_domain(node: &NodeProto) -> bool {
    matches!(node.domain.as_str(), "" | "ai.onnx")
}

fn node_label(node: &NodeProto, node_index: usize, prefix: &str) -> String {
    if node.name.is_empty() {
        format!("{prefix}_{node_index}")
    } else {
        node.name.clone()
    }
}

fn opset_for(node: &NodeProto, opset_imports: &HashMap<String, i64>, minimum: i64) -> Result<i64> {
    let opset = lookup_opset_version(opset_imports, &node.domain).ok_or_else(|| {
        NyError::ModelLoad(format!(
            "standard ONNX {} node '{}' has no standard-domain opset authority",
            node.op_type, node.name
        ))
    })?;
    if opset < minimum {
        return Err(NyError::UnsupportedOp(format!(
            "standard ONNX {} node '{}' uses opset {opset}; the represented schema requires \
             opset {minimum} or newer",
            node.op_type, node.name
        )));
    }
    Ok(opset)
}

// ---------------------------------------------------------------------------
// Trilu -> LayerType::Triu / Tril
// ---------------------------------------------------------------------------

/// Canonicalize a live-data standard ONNX `Trilu` into ny's triangular-mask
/// layer.
///
/// # Why this is EXACT, not an approximation
///
/// ONNX `Trilu-14` retains, over the trailing two axes of `input`:
///
/// ```text
/// upper = 1 (default):  keep entry (i, j)  iff  j - i >= k
/// upper = 0:            keep entry (i, j)  iff  j - i <= k
/// ```
///
/// and zeroes everything else. ny-build's `convert_triangular`
/// (`ny-build/src/convert/elementwise.rs`) materializes a binary mask with
///
/// ```text
/// LayerType::Triu:  keep iff col >= row + diagonal
/// LayerType::Tril:  keep iff col <= row + diagonal
/// ```
///
/// over the same trailing two axes, and emits a `MulConstantLayer` carrying
/// it. `col >= row + diagonal` is `col - row >= diagonal`, so with
/// `diagonal := k` the two predicates are the SAME predicate, entry for
/// entry — this is a renaming, not a numeric reinterpretation. The mask is
/// exactly `{0.0, 1.0}`, so the layer multiplies each element by 1 or by 0;
/// IBP through `MulConstant` is exact for both endpoints (a non-negative
/// coefficient does not swap them) and no rounding occurs (`x * 1.0f32 == x`,
/// `x * 0.0f32 == 0.0` for every finite `x`).
///
/// This is the same lowering the trace bridge reaches through
/// `TraceOp::Triu` / `TraceOp::Tril` (`ny-trace-bridge`
/// `translate/ops_misc.rs::translate_triangular`), which likewise emits only
/// the `diagonal` attribute and lets `convert_triangular` build the mask.
///
/// # What stays refused, and why
///
/// * A `k` operand that is not an exact INT64 constant. `k` selects WHICH
///   entries survive; a runtime `k` makes the mask data-dependent, and ny's
///   `MulConstant` carries ONE fixed mask. Enclosing every possible `k` would
///   need a union over masks (i.e. `[min(0, x), max(0, x)]` everywhere the
///   diagonal can move), which is not what this layer computes — so refuse
///   rather than silently bound a different function.
/// * A numerically integral FLOAT `k`. `Trilu`'s `k` is INT64 on the wire; a
///   FLOAT constant that happens to be integral is not dtype-equivalent and
///   must not gain discrete authority through `WeightStore`'s f32 view. This
///   mirrors the identical refusal in the constant-folding path
///   (`const_fold/ops/trilu.rs`).
/// * `upper` outside `{0, 1}`, or any other attribute.
///
/// A FULLY CONSTANT `Trilu` never reaches here: load-time constant folding
/// (`const_fold/ops/trilu.rs`) evaluates it and
/// `node_is_fully_materialized_constant` drops the node before conversion.
/// This function therefore only ever sees a live data path.
pub(super) fn canonicalize_standard_trilu(
    node: &NodeProto,
    node_index: usize,
    weights: &WeightStore,
    opset_imports: &HashMap<String, i64>,
) -> Result<Option<LayerSpec>> {
    if node.op_type != "Trilu" || !is_standard_domain(node) {
        return Ok(None);
    }
    // Trilu entered the main ONNX domain at opset 14.
    opset_for(node, opset_imports, 14)?;

    if !matches!(node.input.len(), 1 | 2)
        || node.input[0].is_empty()
        || node.output.len() != 1
        || node.output[0].is_empty()
    {
        return Err(NyError::ModelLoad(format!(
            "standard ONNX Trilu node '{}' has an unsupported signature: inputs {:?}, outputs {:?}",
            node.name, node.input, node.output
        )));
    }

    let mut upper = None;
    for attribute in &node.attribute {
        if attribute.name != "upper" {
            return Err(NyError::UnsupportedOp(format!(
                "standard ONNX Trilu node '{}' has unsupported attribute '{}'",
                node.name, attribute.name
            )));
        }
        if attribute.r#type != onnx_proto::attribute_type::INT
            || !matches!(attribute.i_value(), 0 | 1)
        {
            return Err(NyError::UnsupportedOp(format!(
                "standard ONNX Trilu node '{}' requires 'upper' to be INT 0 or 1",
                node.name
            )));
        }
        upper = Some(attribute.i_value() == 1);
    }
    let upper = upper.unwrap_or(true);

    let diagonal = match node.input.get(1).filter(|name| !name.is_empty()) {
        None => 0,
        Some(name) => {
            // INT64 provenance first: a numerically integral FLOAT constant is
            // not the authored dtype and must not act as the discrete `k`.
            if weights.get_integers(name).is_none() {
                return Err(NyError::UnsupportedOp(format!(
                    "standard ONNX Trilu node '{}' requires an exact constant INT64 'k' operand; \
                     '{}' is not an authenticated INT64 constant, and a data-dependent diagonal \
                     would make the retained-entry set data-dependent, which ny's fixed \
                     triangular mask cannot enclose",
                    node.name, name
                )));
            }
            let (values, shape) = read_tensor_i64s_and_shape(weights, name).ok_or_else(|| {
                NyError::UnsupportedOp(format!(
                    "standard ONNX Trilu node '{}' could not materialize its INT64 'k' operand '{}'",
                    node.name, name
                ))
            })?;
            // ONNX declares `k` a scalar; accept the equivalent `[1]` spelling
            // exporters emit, and nothing wider.
            if values.len() != 1 || shape.len() > 1 {
                return Err(NyError::ModelLoad(format!(
                    "standard ONNX Trilu node '{}' requires a scalar 'k', got shape {:?}",
                    node.name, shape
                )));
            }
            values[0]
        }
    };

    let mut attributes = HashMap::new();
    attributes.insert("diagonal".to_string(), AttributeValue::Int(diagonal));
    Ok(Some(LayerSpec {
        name: node_label(node, node_index, "trilu"),
        layer_type: if upper {
            LayerType::Triu
        } else {
            LayerType::Tril
        },
        // The INT64 `k` operand is consumed here; keeping it as an input would
        // present an integer control tensor as a second activation edge.
        inputs: vec![node.input[0].clone()],
        outputs: vec![node.output[0].clone()],
        weights: None,
        attributes,
    }))
}

// ---------------------------------------------------------------------------
// ScatterElements(reduction="add") -> LayerType::ScatterAdd
// ---------------------------------------------------------------------------

/// Canonicalize a standard ONNX `ScatterElements` with `reduction="add"` into
/// ny's `LayerType::ScatterAdd`, and refuse every other reduction BY NAME.
///
/// # Why `reduction="add"` is exactly `ScatterAdd`
///
/// ONNX `ScatterElements` takes `(data, indices, updates)` with
/// `rank(indices) == rank(updates) == rank(data)` and `indices.shape ==
/// updates.shape`, and for `reduction="add"` computes (illustrated at
/// `axis == 1`):
///
/// ```text
/// output = data
/// output[i][indices[i][j][k]][k] += updates[i][j][k]
/// ```
///
/// with index values admitted in `[-s, s-1]` for the axis length `s`,
/// negatives counting from the end.
///
/// `LayerType::ScatterAdd` is documented (`ny-core/src/layer_type.rs`) with
/// that exact update rule, the same `(data, index, src)` operand triple, and
/// the same trailing-relative `axis`. `ScatterAddLayer::scatter_add_tensor`
/// (`ny-propagate/src/layers/transform/accumulate.rs`) checks
/// `indices.shape() == src.shape()` and equal ranks, then walks `src` element
/// by element writing to `dst_coord[axis] = normalize_index(...)`, where
/// `normalize_index` maps a negative raw index to `axis_len + raw` and refuses
/// anything still out of bounds. Operand order, index shape rule, axis role
/// and negative-index convention all coincide, so this is a renaming.
///
/// SOUNDNESS of the additive family (why no duplicate-index union is owed):
/// `+` over the reals is associative and commutative, so a destination named
/// by several index entries receives the SUM of the contributions — one
/// value, not an order-dependent set. That is the property `ScatterND`'s
/// overwrite semantics lacks, which is why `ScatterND` needs a hull over write
/// orders and this does not. The exactness argument stops at the reals: f32
/// `+` is round-to-nearest, so `ScatterAddLayer` accumulates the lower output
/// with `add_down_f32` and the upper with `add_up_f32`, rounding each endpoint
/// OUTWARD. The interval therefore encloses the exact real sum and hence every
/// f32 visit order of it.
///
/// A NON-constant `indices` tensor is deliberately NOT refused: the layer's
/// `ResolvedIndices::Dynamic` path never reads the index tensor and instead
/// widens `data` by the total possible additive contribution of `src` (the sum
/// of the negative parts below and of the positive parts above). Every index
/// assignment routes each `src` element to exactly one destination, so every
/// realizable output lies inside that hull — sound for ANY index, just loose.
///
/// # What stays refused, and why
///
/// * `reduction="none"` (the default, and all of the deprecated `Scatter`
///   op): an OVERWRITE along one axis. Overwriting is not additive, so the
///   `ScatterAdd` layer would compute a different function; and duplicate
///   destinations make it order-dependent, which needs a union ny has no
///   per-axis layer for. `ScatterND` is NOT a substitute — its `indices`
///   carry full coordinates in a trailing axis, a different indexing scheme.
/// * `reduction="mul"`: multiplicative accumulation is not monotone in the
///   accumulated value once negatives are possible, so neither the endpoint
///   pairing nor the directed rounding of `ScatterAddLayer` transfers.
/// * `reduction="max"` / `"min"` (opset 18+): idempotent but not additive;
///   again a different function.
///
/// Each of those is refused by NAME here rather than falling into `op_map`'s
/// generic arm, so the message says which reduction was authored and what ny
/// represents.
pub(super) fn canonicalize_standard_scatter_elements(
    node: &NodeProto,
    node_index: usize,
    opset_imports: &HashMap<String, i64>,
) -> Result<Option<LayerSpec>> {
    if !is_standard_domain(node) {
        return Ok(None);
    }
    if node.op_type == "Scatter" {
        // Deprecated in opset 11 in favour of ScatterElements; its semantics
        // are exactly `ScatterElements(reduction="none")`.
        return Err(NyError::UnsupportedOp(format!(
            "standard ONNX Scatter node '{}' is the deprecated overwrite scatter \
             (equivalent to ScatterElements reduction='none'); ny represents only the additive \
             ScatterElements reduction='add' (LayerType::ScatterAdd), because an overwrite along \
             one axis is order-dependent at duplicate destinations and has no ny layer",
            node.name
        )));
    }
    if node.op_type != "ScatterElements" {
        return Ok(None);
    }
    let opset = opset_for(node, opset_imports, 11)?;

    if node.input.len() != 3
        || node.input.iter().any(String::is_empty)
        || node.output.len() != 1
        || node.output[0].is_empty()
    {
        return Err(NyError::ModelLoad(format!(
            "standard ONNX ScatterElements node '{}' requires exactly three non-empty inputs \
             (data, indices, updates) and one non-empty output; got inputs {:?}, outputs {:?}",
            node.name, node.input, node.output
        )));
    }

    let mut axis = 0_i64;
    let mut reduction: Option<String> = None;
    for attribute in &node.attribute {
        match attribute.name.as_str() {
            "axis" if attribute.r#type == onnx_proto::attribute_type::INT => {
                axis = attribute.i_value();
            }
            // `reduction` only exists from opset 16; an opset-11 graph carrying
            // one is malformed, not an additive scatter.
            "reduction"
                if opset >= 16 && attribute.r#type == onnx_proto::attribute_type::STRING =>
            {
                reduction = Some(String::from_utf8_lossy(attribute.s_value()).to_string());
            }
            _ => {
                return Err(NyError::ModelLoad(format!(
                    "standard ONNX ScatterElements node '{}' has an unsupported or malformed \
                     '{}' attribute at opset {opset}",
                    node.name, attribute.name
                )));
            }
        }
    }

    let reduction = reduction.unwrap_or_else(|| "none".to_string());
    if reduction != "add" {
        return Err(NyError::UnsupportedOp(format!(
            "standard ONNX ScatterElements node '{}' uses reduction='{reduction}'; ny represents \
             only reduction='add' (LayerType::ScatterAdd). 'none' overwrites along the axis and \
             is order-dependent at duplicate destinations, and 'mul'/'max'/'min' are not additive \
             — none of them is the function ScatterAdd computes",
            node.name
        )));
    }

    // Emit the ONNX axis VERBATIM, exactly as the loader does for Gather:
    // `ny-build`'s `remap_axis_trailing` owns the batched/unbatched decision
    // and refuses the ambiguous ONNX axis 0 of a rank>1 tensor rather than
    // guessing. Encoding it here would duplicate that policy in a place with
    // less shape information.
    let mut attributes = parse_node_attributes(node);
    attributes.remove("reduction");
    attributes.insert("axis".to_string(), AttributeValue::Int(axis));

    Ok(Some(LayerSpec {
        name: node_label(node, node_index, "scatter_elements"),
        layer_type: LayerType::ScatterAdd,
        inputs: node.input.clone(),
        outputs: node.output.clone(),
        weights: None,
        attributes,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr0, arr1, ArrayD, IxDyn};

    fn opsets(version: i64) -> HashMap<String, i64> {
        HashMap::from([(String::new(), version)])
    }

    fn int_attr(name: &str, value: i64) -> onnx_proto::AttributeProto {
        onnx_proto::AttributeProto {
            name: name.to_string(),
            i: Some(value),
            r#type: onnx_proto::attribute_type::INT,
            ..Default::default()
        }
    }

    fn string_attr(name: &str, value: &str) -> onnx_proto::AttributeProto {
        onnx_proto::AttributeProto {
            name: name.to_string(),
            s: Some(value.as_bytes().to_vec()),
            r#type: onnx_proto::attribute_type::STRING,
            ..Default::default()
        }
    }

    fn trilu_node(inputs: &[&str], upper: Option<i64>) -> NodeProto {
        NodeProto {
            name: "mask".to_string(),
            op_type: "Trilu".to_string(),
            input: inputs.iter().map(|value| (*value).to_string()).collect(),
            output: vec!["out".to_string()],
            attribute: upper.into_iter().map(|v| int_attr("upper", v)).collect(),
            ..Default::default()
        }
    }

    #[test]
    fn trilu_upper_and_lower_map_to_the_triangular_layers_with_k_as_diagonal() {
        let mut weights = WeightStore::new();
        weights.insert_integers("k".to_string(), arr0(2_i64).into_dyn());
        weights.insert("k".to_string(), arr0(2.0_f32).into_dyn());

        let upper = canonicalize_standard_trilu(
            &trilu_node(&["x", "k"], Some(1)),
            0,
            &weights,
            &opsets(14),
        )
        .unwrap()
        .unwrap();
        assert_eq!(upper.layer_type, LayerType::Triu);
        assert_eq!(upper.inputs, ["x"], "the INT64 k operand is consumed");
        assert_eq!(upper.attributes["diagonal"], AttributeValue::Int(2));

        let lower = canonicalize_standard_trilu(
            &trilu_node(&["x", "k"], Some(0)),
            0,
            &weights,
            &opsets(14),
        )
        .unwrap()
        .unwrap();
        assert_eq!(lower.layer_type, LayerType::Tril);
        assert_eq!(lower.attributes["diagonal"], AttributeValue::Int(2));
    }

    #[test]
    fn trilu_defaults_to_upper_with_diagonal_zero() {
        let spec = canonicalize_standard_trilu(
            &trilu_node(&["x"], None),
            0,
            &WeightStore::new(),
            &opsets(14),
        )
        .unwrap()
        .unwrap();
        assert_eq!(spec.layer_type, LayerType::Triu);
        assert_eq!(spec.attributes["diagonal"], AttributeValue::Int(0));
    }

    #[test]
    fn trilu_refuses_a_runtime_or_float_diagonal() {
        // No constant at all: the retained-entry set would be data-dependent.
        assert!(matches!(
            canonicalize_standard_trilu(
                &trilu_node(&["x", "k"], None),
                0,
                &WeightStore::new(),
                &opsets(14)
            ),
            Err(NyError::UnsupportedOp(_))
        ));

        // A numerically integral FLOAT is not the authored INT64 dtype.
        let mut weights = WeightStore::new();
        weights.insert("k".to_string(), arr0(1.0_f32).into_dyn());
        assert!(matches!(
            canonicalize_standard_trilu(&trilu_node(&["x", "k"], None), 0, &weights, &opsets(14)),
            Err(NyError::UnsupportedOp(_))
        ));
    }

    #[test]
    fn trilu_refuses_non_boolean_upper_unknown_attributes_and_old_opsets() {
        assert!(canonicalize_standard_trilu(
            &trilu_node(&["x"], Some(2)),
            0,
            &WeightStore::new(),
            &opsets(14)
        )
        .is_err());

        let mut extra = trilu_node(&["x"], Some(1));
        extra.attribute.push(int_attr("lower", 1));
        assert!(canonicalize_standard_trilu(&extra, 0, &WeightStore::new(), &opsets(14)).is_err());

        assert!(canonicalize_standard_trilu(
            &trilu_node(&["x"], Some(1)),
            0,
            &WeightStore::new(),
            &opsets(13)
        )
        .is_err());
    }

    #[test]
    fn trilu_declines_a_custom_domain_node_for_its_registered_handler() {
        let mut node = trilu_node(&["x"], Some(1));
        node.domain = "com.vendor".to_string();
        assert!(
            canonicalize_standard_trilu(&node, 0, &WeightStore::new(), &opsets(14))
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn trilu_requires_a_scalar_k() {
        let mut weights = WeightStore::new();
        weights.insert_integers("k".to_string(), arr1(&[1_i64, 2]).into_dyn());
        weights.insert(
            "k".to_string(),
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0_f32, 2.0]).unwrap(),
        );
        assert!(canonicalize_standard_trilu(
            &trilu_node(&["x", "k"], None),
            0,
            &weights,
            &opsets(14)
        )
        .is_err());
    }

    fn scatter_node(reduction: Option<&str>, axis: Option<i64>) -> NodeProto {
        let mut attribute = Vec::new();
        if let Some(axis) = axis {
            attribute.push(int_attr("axis", axis));
        }
        if let Some(reduction) = reduction {
            attribute.push(string_attr("reduction", reduction));
        }
        NodeProto {
            name: "scatter".to_string(),
            op_type: "ScatterElements".to_string(),
            input: vec!["data".to_string(), "idx".to_string(), "upd".to_string()],
            output: vec!["out".to_string()],
            attribute,
            ..Default::default()
        }
    }

    #[test]
    fn scatter_elements_add_maps_to_scatter_add_with_the_authored_axis() {
        let spec = canonicalize_standard_scatter_elements(
            &scatter_node(Some("add"), Some(1)),
            0,
            &opsets(18),
        )
        .unwrap()
        .unwrap();
        assert_eq!(spec.layer_type, LayerType::ScatterAdd);
        assert_eq!(spec.inputs, ["data", "idx", "upd"]);
        assert_eq!(spec.attributes["axis"], AttributeValue::Int(1));
        assert!(
            !spec.attributes.contains_key("reduction"),
            "the reduction is consumed by the layer-type choice"
        );
    }

    #[test]
    fn scatter_elements_negative_axis_survives_verbatim() {
        let spec = canonicalize_standard_scatter_elements(
            &scatter_node(Some("add"), Some(-1)),
            0,
            &opsets(18),
        )
        .unwrap()
        .unwrap();
        assert_eq!(spec.attributes["axis"], AttributeValue::Int(-1));
    }

    #[test]
    fn scatter_elements_refuses_every_non_additive_reduction_by_name() {
        for reduction in [None, Some("none"), Some("mul"), Some("max"), Some("min")] {
            let error = canonicalize_standard_scatter_elements(
                &scatter_node(reduction, Some(1)),
                0,
                &opsets(18),
            )
            .expect_err("non-additive reduction must be refused");
            let message = error.to_string();
            assert!(
                message.contains(reduction.unwrap_or("none")),
                "refusal must name the authored reduction, got: {message}"
            );
        }
    }

    #[test]
    fn scatter_elements_refuses_a_reduction_attribute_before_opset_16() {
        assert!(canonicalize_standard_scatter_elements(
            &scatter_node(Some("add"), Some(1)),
            0,
            &opsets(13)
        )
        .is_err());
    }

    #[test]
    fn deprecated_scatter_is_refused_by_name() {
        let mut node = scatter_node(None, Some(1));
        node.op_type = "Scatter".to_string();
        let error = canonicalize_standard_scatter_elements(&node, 0, &opsets(10))
            .expect_err("the deprecated overwrite Scatter must be refused");
        assert!(error.to_string().contains("Scatter"));
    }

    #[test]
    fn scatter_elements_declines_a_custom_domain_node() {
        let mut node = scatter_node(Some("add"), Some(1));
        node.domain = "com.vendor".to_string();
        assert!(
            canonicalize_standard_scatter_elements(&node, 0, &opsets(18))
                .unwrap()
                .is_none()
        );
    }
}
