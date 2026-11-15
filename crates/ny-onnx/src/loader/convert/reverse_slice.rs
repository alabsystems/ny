// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Reversing ONNX `Slice` (`steps = [-1]`) lowered to unit `Slice` + `Concat`.
//!
//! ny's `SliceLayer` extracts a contiguous FORWARD range `[start, end)` along
//! one axis; it has no stride and no direction. A reversing slice is therefore
//! not expressible as one layer — but it IS expressible as the composition ny
//! already runs on the trace path: `ny-trace-bridge`'s `TraceOp::Flip` lowers
//! to N unit `Slice`s emitted in reverse order followed by one `Concat`
//! (`translate/ops_misc.rs::translate_flip`). This module reaches the same
//! decomposition from ONNX bytes.
//!
//! ## Scope: only a PROVEN negative step is rewritten
//!
//! The canonicalizer returns `Ok(None)` — leaving the node to the ordinary
//! Slice path, byte for byte as before — unless it can positively resolve the
//! `steps` operand at load time AND see a negative entry there. It never
//! refuses a node the existing path accepts.
//!
//! KNOWN RESIDUAL (not fixed here, deliberately): when `steps` is not
//! resolvable at LOAD time, `ny-build`'s `convert_slice` reads
//! `slice_constant_i64_values(...) -> Ok(None)` and then SKIPS its
//! `step != 1` check entirely, i.e. an unresolved `steps` operand is treated
//! as `step = 1`. Closing that from here would mean refusing nodes whose
//! `steps` `ny-build` can still resolve from its own `evaluated_constants`
//! (a superset of the loader's `WeightStore`), which would break loads that
//! work today. The fix belongs in `convert_slice`, in the crate that owns the
//! resolution.

use crate::loader::const_fold::common::read_tensor_i64s;
use crate::onnx_proto::NodeProto;
use crate::{AttributeValue, LayerSpec, WeightStore};
use ny_core::{LayerType, NyError, Result};
use std::collections::HashMap;

/// Cap on the number of unit slices one reversing `Slice` may expand into.
///
/// Matches `ny-trace-bridge`'s `MAX_DECOMPOSE_DIM`, which bounds the identical
/// `Flip` decomposition. A larger reversal is refused rather than silently
/// materialized as thousands of graph nodes.
const MAX_REVERSED_SLICE_ELEMENTS: usize = 2048;

/// One resolved `(start, end, axis, step)` of an opset-10+ `Slice`.
struct SliceControl {
    starts: Vec<i64>,
    ends: Vec<i64>,
    axes: Option<Vec<i64>>,
    steps: Vec<i64>,
}

fn is_standard_domain(node: &NodeProto) -> bool {
    matches!(node.domain.as_str(), "" | "ai.onnx")
}

/// Read the opset-10+ control operands, or `None` when any REQUIRED one is not
/// resolvable from the loader's constant view.
fn resolve_control(node: &NodeProto, weights: &WeightStore) -> Option<SliceControl> {
    let starts = read_tensor_i64s(weights, node.input.get(1)?)?;
    let ends = read_tensor_i64s(weights, node.input.get(2)?)?;
    let axes = match node.input.get(3).filter(|name| !name.is_empty()) {
        Some(name) => Some(read_tensor_i64s(weights, name)?),
        None => None,
    };
    // An ABSENT steps operand means the ONNX default of all ones; an operand
    // that is present but unresolvable makes the direction unknown, so this
    // module declines (see the module-level RESIDUAL note).
    let steps = match node.input.get(4).filter(|name| !name.is_empty()) {
        Some(name) => read_tensor_i64s(weights, name)?,
        None => vec![1; starts.len()],
    };
    Some(SliceControl {
        starts,
        ends,
        axes,
        steps,
    })
}

/// Resolve the ONNX index list a reversing slice selects, in output order.
///
/// Replicates the ONNX `Slice` normalization for a NEGATIVE step
/// operation-for-operation:
///
/// ```text
/// if start < 0 { start += dim }   start = clamp(start,  0, dim - 1)
/// if end   < 0 { end   += dim }   end   = clamp(end,   -1, dim - 1)
/// indices = start, start - 1, ... while index > end
/// ```
///
/// The arithmetic runs in `i128` so the canonical `ends = i64::MIN`
/// "to the very beginning" sentinel cannot overflow before it is clamped.
fn reversed_indices(start_raw: i64, end_raw: i64, dim: usize) -> Vec<usize> {
    let dim_i = dim as i128;
    let mut start = i128::from(start_raw);
    if start < 0 {
        start += dim_i;
    }
    let start = start.clamp(0, dim_i - 1);

    let mut end = i128::from(end_raw);
    if end < 0 {
        end += dim_i;
    }
    let end = end.clamp(-1, dim_i - 1);

    if start <= end {
        return Vec::new();
    }
    ((end + 1)..=start)
        .rev()
        .map(|index| index as usize)
        .collect()
}

fn slice_spec(name: &str, input: &str, output: &str, axis: i64, index: usize) -> LayerSpec {
    let mut attributes = HashMap::new();
    attributes.insert("axis".to_string(), AttributeValue::Int(axis));
    attributes.insert("start".to_string(), AttributeValue::Int(index as i64));
    attributes.insert("end".to_string(), AttributeValue::Int(index as i64 + 1));
    LayerSpec {
        name: name.to_string(),
        layer_type: LayerType::Slice,
        inputs: vec![input.to_string()],
        outputs: vec![output.to_string()],
        weights: None,
        attributes,
    }
}

/// Canonicalize a standard ONNX `Slice` whose resolved `steps` is `[-1]` into
/// the `Slice` + `Concat` reversal ny already runs for `TraceOp::Flip`.
///
/// # Why the decomposition is EXACT
///
/// A reversing slice is a pure REINDEXING: output position `p` carries input
/// position `indices[p]`, with every input element used at most once and no
/// arithmetic performed. The decomposition realizes exactly that map — unit
/// `Slice` number `p` extracts `[indices[p], indices[p] + 1)` along the axis,
/// and `Concat` lays those length-1 blocks out along the same axis in output
/// order. `SliceLayer` copies each endpoint tensor elementwise (IBP is an
/// index copy, CROWN backward is the transposed gather), and `ConcatLayer`
/// likewise. No value is combined with another, so no rounding is introduced
/// and both endpoints are carried exactly: the emitted bounds are the input
/// bounds permuted, which encloses the reversed concrete evaluation with zero
/// slack. This is the same argument that licenses `TraceOp::Flip`.
///
/// The axis is emitted TRAILING-RELATIVE (negative, `axis - rank`), the
/// encoding that names the same semantic dimension whether or not ny's
/// unbatched runtime tensor kept the leading batch axis — the convention the
/// trace bridge uses for every axis it emits, and the one
/// `ConvertContext::remap_axis_trailing` passes through unchanged.
///
/// # What stays refused, and why
///
/// * A negative step other than `-1`. The index list is equally computable,
///   but a strided reversal has never been exercised end to end here; refuse
///   by name rather than ship an unexercised index map.
/// * More than one sliced axis when any step is negative. `ny-build`'s
///   `convert_slice` is single-axis, and composing several reversals would
///   need an emission order this function does not fix.
/// * An empty result (`start <= end` after clamping). ny's tensors have no
///   empty-extent representation, and `SliceLayer` refuses a range that clamps
///   to nothing.
/// * A reversal longer than [`MAX_REVERSED_SLICE_ELEMENTS`].
/// * An axis whose length is not a known, positive, static dimension: the
///   index list is not decidable without it.
/// * ONNX axis 0 of a rank > 1 tensor — the (possibly stripped) batch axis,
///   which `remap_axis_trailing` refuses for the same reason: no single
///   encoding is correct under both of ny's internal layouts.
pub(super) fn canonicalize_reversed_slice(
    node: &NodeProto,
    node_index: usize,
    weights: &WeightStore,
    tensor_shapes: &HashMap<String, Vec<i64>>,
) -> Result<Option<Vec<LayerSpec>>> {
    if node.op_type != "Slice" || !is_standard_domain(node) {
        return Ok(None);
    }
    // Opset 1-9 stores starts/ends/axes as attributes and has no steps at all,
    // so it can never be a reversing slice.
    if node.input.len() < 3 {
        return Ok(None);
    }
    let Some(control) = resolve_control(node, weights) else {
        return Ok(None);
    };
    if control.steps.iter().all(|&step| step >= 1) {
        return Ok(None);
    }

    let name = if node.name.is_empty() {
        format!("slice_{node_index}")
    } else {
        node.name.clone()
    };

    if control.starts.len() != 1
        || control.ends.len() != 1
        || control.steps.len() != 1
        || control.axes.as_ref().is_some_and(|axes| axes.len() != 1)
    {
        return Err(NyError::UnsupportedOp(format!(
            "standard ONNX Slice node '{name}' reverses along one of {} axes; ny represents a \
             reversing slice only in the single-axis form (starts/ends/axes/steps of length 1)",
            control.starts.len()
        )));
    }
    let step = control.steps[0];
    if step != -1 {
        return Err(NyError::UnsupportedOp(format!(
            "standard ONNX Slice node '{name}' uses step={step}; ny represents step=1 (a native \
             SliceLayer) and step=-1 (the Slice+Concat reversal), and refuses other strides \
             rather than emitting an unexercised index map"
        )));
    }

    let data = &node.input[0];
    let shape = tensor_shapes.get(data).ok_or_else(|| {
        NyError::UnsupportedOp(format!(
            "standard ONNX Slice node '{name}' reverses along an axis of '{data}', whose shape is \
             not known at load time; the reversal index list is not decidable without it"
        ))
    })?;
    let rank = shape.len();
    let raw_axis = control.axes.as_ref().map(|axes| axes[0]).unwrap_or(0);
    let axis = if raw_axis < 0 {
        raw_axis + rank as i64
    } else {
        raw_axis
    };
    if axis < 0 || axis >= rank as i64 {
        return Err(NyError::ModelLoad(format!(
            "standard ONNX Slice node '{name}': axis {raw_axis} is out of range for '{data}' of \
             rank {rank}"
        )));
    }
    if axis == 0 && rank > 1 {
        return Err(NyError::UnsupportedOp(format!(
            "standard ONNX Slice node '{name}' reverses ONNX axis 0 of rank-{rank} input \
             '{data}', the (possibly stripped) batch dimension; no single internal axis encoding \
             is correct under both of ny's runtime layouts, so this is refused rather than guessed"
        )));
    }
    let dim = usize::try_from(shape[axis as usize])
        .ok()
        .filter(|&d| d > 0)
        .ok_or_else(|| {
            NyError::UnsupportedOp(format!(
            "standard ONNX Slice node '{name}' reverses axis {axis} of '{data}', whose recorded \
             extent {} is not a positive static dimension",
            shape[axis as usize]
        ))
        })?;

    let indices = reversed_indices(control.starts[0], control.ends[0], dim);
    if indices.is_empty() {
        return Err(NyError::UnsupportedOp(format!(
            "standard ONNX Slice node '{name}' selects no elements (start={}, end={}, step=-1 on \
             an axis of length {dim}); ny has no empty-extent tensor representation",
            control.starts[0], control.ends[0]
        )));
    }
    if indices.len() > MAX_REVERSED_SLICE_ELEMENTS {
        return Err(NyError::UnsupportedOp(format!(
            "standard ONNX Slice node '{name}' reverses {} positions, above the {} decomposition \
             limit for the Slice+Concat reversal",
            indices.len(),
            MAX_REVERSED_SLICE_ELEMENTS
        )));
    }

    // Trailing-relative encoding: names the same dimension whether or not the
    // runtime tensor kept its leading batch axis.
    let trailing_axis = axis - rank as i64;
    let output = &node.output[0];

    if indices.len() == 1 {
        // A single retained position is an ordinary forward unit slice; a
        // Concat would need two inputs and there is nothing to reverse.
        return Ok(Some(vec![slice_spec(
            &name,
            data,
            output,
            trailing_axis,
            indices[0],
        )]));
    }

    let mut specs = Vec::with_capacity(indices.len() + 1);
    let mut concat_inputs = Vec::with_capacity(indices.len());
    for (position, &index) in indices.iter().enumerate() {
        let spec_name = format!("{name}__ny_rev{node_index}_{position}");
        let tensor = format!("{spec_name}_out");
        specs.push(slice_spec(&spec_name, data, &tensor, trailing_axis, index));
        concat_inputs.push(tensor);
    }
    let mut attributes = HashMap::new();
    attributes.insert("axis".to_string(), AttributeValue::Int(trailing_axis));
    specs.push(LayerSpec {
        name,
        layer_type: LayerType::Concat,
        inputs: concat_inputs,
        outputs: vec![output.clone()],
        weights: None,
        attributes,
    });
    Ok(Some(specs))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, ArrayD, IxDyn};

    fn slice_node(inputs: &[&str]) -> NodeProto {
        NodeProto {
            name: "rev".to_string(),
            op_type: "Slice".to_string(),
            input: inputs.iter().map(|value| (*value).to_string()).collect(),
            output: vec!["out".to_string()],
            ..Default::default()
        }
    }

    fn int_weight(weights: &mut WeightStore, name: &str, values: &[i64]) {
        weights.insert_integers(name.to_string(), arr1(values).into_dyn());
        weights.insert(
            name.to_string(),
            ArrayD::from_shape_vec(
                IxDyn(&[values.len()]),
                values.iter().map(|&v| v as f32).collect(),
            )
            .unwrap(),
        );
    }

    fn reverse_weights(start: i64, end: i64, axis: i64, step: i64) -> WeightStore {
        let mut weights = WeightStore::new();
        int_weight(&mut weights, "starts", &[start]);
        int_weight(&mut weights, "ends", &[end]);
        int_weight(&mut weights, "axes", &[axis]);
        int_weight(&mut weights, "steps", &[step]);
        weights
    }

    fn shapes(shape: &[i64]) -> HashMap<String, Vec<i64>> {
        HashMap::from([("x".to_string(), shape.to_vec())])
    }

    #[test]
    fn full_reversal_emits_unit_slices_in_reverse_order_plus_a_concat() {
        let specs = canonicalize_reversed_slice(
            &slice_node(&["x", "starts", "ends", "axes", "steps"]),
            7,
            &reverse_weights(-1, i64::MIN, 1, -1),
            &shapes(&[2, 4]),
        )
        .unwrap()
        .unwrap();

        assert_eq!(specs.len(), 5, "4 unit slices + 1 concat");
        let starts: Vec<i64> = specs[..4]
            .iter()
            .map(|spec| match spec.attributes["start"] {
                AttributeValue::Int(value) => value,
                _ => panic!("start must be an Int"),
            })
            .collect();
        assert_eq!(starts, vec![3, 2, 1, 0], "output order is reversed");
        for spec in &specs[..4] {
            assert_eq!(spec.layer_type, LayerType::Slice);
            assert_eq!(spec.inputs, ["x"]);
            assert_eq!(
                spec.attributes["axis"],
                AttributeValue::Int(-1),
                "axis is trailing-relative"
            );
        }
        let concat = specs.last().unwrap();
        assert_eq!(concat.layer_type, LayerType::Concat);
        assert_eq!(concat.outputs, ["out"]);
        assert_eq!(concat.attributes["axis"], AttributeValue::Int(-1));
        assert_eq!(
            concat.inputs,
            specs[..4]
                .iter()
                .map(|spec| spec.outputs[0].clone())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn partial_reversal_honours_onnx_start_end_clamping() {
        // x[2:0:-1] over an axis of length 4 selects positions 2 then 1.
        let specs = canonicalize_reversed_slice(
            &slice_node(&["x", "starts", "ends", "axes", "steps"]),
            0,
            &reverse_weights(2, 0, 1, -1),
            &shapes(&[2, 4]),
        )
        .unwrap()
        .unwrap();
        let starts: Vec<i64> = specs[..specs.len() - 1]
            .iter()
            .map(|spec| match spec.attributes["start"] {
                AttributeValue::Int(value) => value,
                _ => panic!("start must be an Int"),
            })
            .collect();
        assert_eq!(starts, vec![2, 1]);
    }

    #[test]
    fn a_single_retained_position_needs_no_concat() {
        // x[-1:-2:-1] selects only position 3.
        let specs = canonicalize_reversed_slice(
            &slice_node(&["x", "starts", "ends", "axes", "steps"]),
            0,
            &reverse_weights(-1, -2, 1, -1),
            &shapes(&[2, 4]),
        )
        .unwrap()
        .unwrap();
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].layer_type, LayerType::Slice);
        assert_eq!(specs[0].outputs, ["out"]);
        assert_eq!(specs[0].attributes["start"], AttributeValue::Int(3));
        assert_eq!(specs[0].attributes["end"], AttributeValue::Int(4));
    }

    #[test]
    fn a_forward_slice_is_left_entirely_to_the_ordinary_path() {
        for step in [None, Some(1)] {
            let mut weights = WeightStore::new();
            int_weight(&mut weights, "starts", &[0]);
            int_weight(&mut weights, "ends", &[2]);
            int_weight(&mut weights, "axes", &[1]);
            let inputs: Vec<&str> = if step.is_some() {
                int_weight(&mut weights, "steps", &[1]);
                vec!["x", "starts", "ends", "axes", "steps"]
            } else {
                vec!["x", "starts", "ends", "axes"]
            };
            assert!(canonicalize_reversed_slice(
                &slice_node(&inputs),
                0,
                &weights,
                &shapes(&[2, 4])
            )
            .unwrap()
            .is_none());
        }
    }

    #[test]
    fn an_unresolvable_steps_operand_changes_nothing() {
        let mut weights = WeightStore::new();
        int_weight(&mut weights, "starts", &[0]);
        int_weight(&mut weights, "ends", &[4]);
        int_weight(&mut weights, "axes", &[1]);
        assert!(canonicalize_reversed_slice(
            &slice_node(&["x", "starts", "ends", "axes", "runtime_steps"]),
            0,
            &weights,
            &shapes(&[2, 4])
        )
        .unwrap()
        .is_none());
    }

    #[test]
    fn strides_other_than_minus_one_are_refused_by_name() {
        let error = canonicalize_reversed_slice(
            &slice_node(&["x", "starts", "ends", "axes", "steps"]),
            0,
            &reverse_weights(-1, i64::MIN, 1, -2),
            &shapes(&[2, 4]),
        )
        .expect_err("step=-2 must be refused");
        assert!(error.to_string().contains("step=-2"), "{error}");
    }

    #[test]
    fn reversing_the_batch_axis_is_refused() {
        assert!(matches!(
            canonicalize_reversed_slice(
                &slice_node(&["x", "starts", "ends", "axes", "steps"]),
                0,
                &reverse_weights(-1, i64::MIN, 0, -1),
                &shapes(&[2, 4])
            ),
            Err(NyError::UnsupportedOp(_))
        ));
    }

    #[test]
    fn an_empty_reversal_and_an_unknown_extent_are_refused() {
        // start == end after clamping selects nothing.
        assert!(canonicalize_reversed_slice(
            &slice_node(&["x", "starts", "ends", "axes", "steps"]),
            0,
            &reverse_weights(0, 0, 1, -1),
            &shapes(&[2, 4])
        )
        .is_err());

        // A dynamic extent leaves the index list undecidable.
        assert!(canonicalize_reversed_slice(
            &slice_node(&["x", "starts", "ends", "axes", "steps"]),
            0,
            &reverse_weights(-1, i64::MIN, 1, -1),
            &shapes(&[2, -1])
        )
        .is_err());
    }

    #[test]
    fn oversized_reversals_are_refused() {
        let dim = (MAX_REVERSED_SLICE_ELEMENTS + 1) as i64;
        assert!(canonicalize_reversed_slice(
            &slice_node(&["x", "starts", "ends", "axes", "steps"]),
            0,
            &reverse_weights(-1, i64::MIN, 1, -1),
            &shapes(&[2, dim])
        )
        .is_err());
    }

    #[test]
    fn a_custom_domain_slice_is_declined() {
        let mut node = slice_node(&["x", "starts", "ends", "axes", "steps"]);
        node.domain = "com.vendor".to_string();
        assert!(canonicalize_reversed_slice(
            &node,
            0,
            &reverse_weights(-1, i64::MIN, 1, -1),
            &shapes(&[2, 4])
        )
        .unwrap()
        .is_none());
    }

    #[test]
    fn onnx_clamping_rules_match_the_spec_for_negative_steps() {
        // Sentinel end: whole axis, reversed.
        assert_eq!(reversed_indices(-1, i64::MIN, 4), vec![3, 2, 1, 0]);
        // Negative start counts from the end; negative end likewise.
        assert_eq!(reversed_indices(-1, -3, 4), vec![3, 2]);
        // An oversized start clamps to dim - 1 (NOT dim, as a forward slice would).
        assert_eq!(reversed_indices(99, -5, 4), vec![3, 2, 1, 0]);
        // start <= end selects nothing.
        assert_eq!((reversed_indices(1, 2, 4)).len(), 0);
        assert_eq!((reversed_indices(0, 0, 4)).len(), 0);
    }
}
