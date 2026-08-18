// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Raw, versioned schemas for ONNX layout and shape operators represented by
//! ny.
//!
//! These checks run before constant folding.  That ordering is important:
//! folding can erase an INT64 control edge, an obsolete attribute form, or a
//! non-FLOAT data dtype and leave a superficially compatible `LayerSpec`.

use crate::loader::const_fold::is_standard_onnx_domain;
use crate::onnx_proto::{self, AttributeProto, GraphProto, NodeProto};
use ny_core::{NyError, Result};
use std::collections::{HashMap, HashSet};

use super::literal_cone::{is_exact_integer_literal_node, LiteralCone, LiteralExemptions};
use super::quantization_preflight::RawDtypeResolver;

const FLOAT: i32 = 1;
const INT32: i32 = 6;
const INT64: i32 = 7;

const AUDITED_OPERATORS: &[&str] = &[
    "Shape",
    "Expand",
    "Reshape",
    "Flatten",
    "Transpose",
    "Squeeze",
    "Unsqueeze",
    "Slice",
    "Concat",
    "Split",
    "Tile",
    "Range",
];

pub(super) fn validate_structural_schemas(
    graph: &GraphProto,
    opset_imports: &HashMap<String, i64>,
    literals: &LiteralCone,
    exemptions: &mut LiteralExemptions,
) -> Result<()> {
    let standard_opset = opset_imports
        .get("")
        .copied()
        .or_else(|| opset_imports.get("ai.onnx").copied());
    let mut dtypes = RawDtypeResolver::new(graph);
    let mut shapes = RawShapeResolver::new(graph);

    for node in &graph.node {
        if !is_standard_onnx_domain(&node.domain)
            || !AUDITED_OPERATORS.contains(&node.op_type.as_str())
        {
            continue;
        }
        let opset = standard_opset
            .ok_or_else(|| node_error(node, "has no standard-domain opset import".to_string()))?;
        match node.op_type.as_str() {
            "Shape" => validate_shape(node, opset, &mut dtypes)?,
            "Expand" => validate_expand(node, opset, &mut dtypes, &mut shapes)?,
            "Reshape" => validate_reshape(node, opset, &mut dtypes, &mut shapes)?,
            "Flatten" => validate_flatten(node, opset, &mut dtypes, &mut shapes)?,
            "Transpose" => validate_transpose(node, opset, &mut dtypes, &mut shapes)?,
            "Squeeze" => validate_squeeze(node, opset, &mut dtypes, &mut shapes)?,
            "Unsqueeze" => validate_unsqueeze(node, opset, &mut dtypes, &mut shapes)?,
            "Slice" => validate_slice(node, opset, &mut dtypes, &mut shapes)?,
            "Concat" => validate_concat(node, opset, &mut dtypes, &mut shapes)?,
            "Split" => validate_split(node, opset, &mut dtypes, &mut shapes)?,
            "Tile" => validate_tile(node, opset, &mut dtypes, &mut shapes)?,
            "Range" => validate_range(node, opset, &mut dtypes, &mut shapes)?,
            _ => unreachable!("operator filtered above"),
        }
    }
    validate_int64_structural_control_paths(graph, &mut dtypes, literals, exemptions)?;
    Ok(())
}

/// Test-only: validate and immediately discharge any literal-cone deferral.
///
/// A unit test never runs constant folding, so a deferred refusal must re-raise
/// with its original message here.
#[cfg(test)]
pub(super) fn validate_structural_schemas_unfolded(
    graph: &GraphProto,
    opset_imports: &HashMap<String, i64>,
) -> Result<()> {
    let literals = LiteralCone::new(graph);
    let mut exemptions = LiteralExemptions::default();
    validate_structural_schemas(graph, opset_imports, &literals, &mut exemptions)?;
    exemptions.require_all_folded(&crate::WeightStore::new())
}

fn validate_shape(node: &NodeProto, opset: i64, dtypes: &mut RawDtypeResolver<'_>) -> Result<()> {
    require_minimum_opset(node, opset, 1)?;
    require_exact_io(node, 1, 1)?;
    for attribute in &node.attribute {
        match attribute.name.as_str() {
            "start" | "end" if opset >= 15 => {
                require_attribute_type(node, attribute, onnx_proto::attribute_type::INT)?;
            }
            "start" | "end" => {
                return Err(node_error(
                    node,
                    format!(
                        "uses '{}' before the Shape-15 sliced-shape schema",
                        attribute.name
                    ),
                ));
            }
            _ => return Err(unknown_attribute(node, attribute, opset)),
        }
    }

    // Shape reads metadata rather than tensor elements.  FLOAT activation
    // tensors and exact INT64 shape-control tensors are both represented; the
    // result itself is always INT64.
    require_dtype_in(node, dtypes, &node.input[0], "data", &[FLOAT, INT64])?;
    require_dtype(node, dtypes, &node.output[0], "shape output", INT64)
}

fn validate_range(
    node: &NodeProto,
    opset: i64,
    dtypes: &mut RawDtypeResolver<'_>,
    shapes: &mut RawShapeResolver<'_>,
) -> Result<()> {
    require_minimum_opset(node, opset, 11)?;
    require_exact_io(node, 3, 1)?;
    for attribute in &node.attribute {
        match attribute.name.as_str() {
            // Range-27 added stash_type for FLOAT16/BFLOAT16 accumulation.
            // NY admits only FLOAT32 or exact static INT64 Range programs, so
            // the sole represented value is the schema default FLOAT.
            "stash_type" if opset >= 27 => {
                require_attribute_type(node, attribute, onnx_proto::attribute_type::INT)?;
                if attribute.i_value() != i64::from(FLOAT) {
                    return Err(NyError::UnsupportedOp(format!(
                        "standard ONNX Range node '{}' sets stash_type={}; ny represents only FLOAT ({FLOAT}) accumulation",
                        node.name,
                        attribute.i_value()
                    )));
                }
            }
            _ => return Err(unknown_attribute(node, attribute, opset)),
        }
    }

    // The folder models native FLOAT32 arithmetic and exact INT64 shape
    // programs. Other official Range dtypes deliberately fail closed before
    // their authored rounding/provenance can be erased by constant folding.
    require_layout_data_type(node, dtypes, &[0, 1, 2], &[FLOAT, INT64])?;
    for (index, role) in [(0, "start"), (1, "limit"), (2, "delta")] {
        require_rank(node, shapes, &node.input[index], role, 0)?;
    }
    require_rank(node, shapes, &node.output[0], "output", 1)
}

fn validate_expand(
    node: &NodeProto,
    opset: i64,
    dtypes: &mut RawDtypeResolver<'_>,
    shapes: &mut RawShapeResolver<'_>,
) -> Result<()> {
    require_minimum_opset(node, opset, 8)?;
    require_exact_io(node, 2, 1)?;
    require_no_attributes(node, opset)?;
    require_layout_data_type(node, dtypes, &[0], &[FLOAT, INT64])?;
    require_dtype(node, dtypes, &node.input[1], "shape input", INT64)?;
    require_rank(node, shapes, &node.input[1], "shape input", 1)
}

fn validate_reshape(
    node: &NodeProto,
    opset: i64,
    dtypes: &mut RawDtypeResolver<'_>,
    shapes: &mut RawShapeResolver<'_>,
) -> Result<()> {
    // Reshape-1 used a `shape` attribute and one input.  NY's standard ONNX
    // lowering implements the tensor-input schema introduced at opset 5.
    require_minimum_opset(node, opset, 5)?;
    require_exact_io(node, 2, 1)?;
    for attribute in &node.attribute {
        match attribute.name.as_str() {
            "allowzero" if opset >= 14 => {
                let value = require_boolean_int_attribute(node, attribute)?;
                if value != 0 {
                    return Err(NyError::UnsupportedOp(format!(
                        "standard ONNX Reshape node '{}' sets allowzero=1; ny's propagated Reshape layer interprets zero as the ONNX copy-dimension sentinel",
                        node.name
                    )));
                }
            }
            "allowzero" => {
                return Err(node_error(
                    node,
                    format!("uses allowzero before its opset-14 introduction (opset {opset})"),
                ));
            }
            _ => return Err(unknown_attribute(node, attribute, opset)),
        }
    }
    require_layout_data_type(node, dtypes, &[0], &[FLOAT, INT64])?;
    require_dtype(node, dtypes, &node.input[1], "shape input", INT64)?;
    require_rank(node, shapes, &node.input[1], "shape input", 1)
}

fn validate_flatten(
    node: &NodeProto,
    opset: i64,
    dtypes: &mut RawDtypeResolver<'_>,
    shapes: &mut RawShapeResolver<'_>,
) -> Result<()> {
    require_minimum_opset(node, opset, 1)?;
    require_exact_io(node, 1, 1)?;
    let mut axis = 1;
    for attribute in &node.attribute {
        match attribute.name.as_str() {
            "axis" => axis = require_int_attribute(node, attribute)?,
            _ => return Err(unknown_attribute(node, attribute, opset)),
        }
    }
    require_layout_data_type(node, dtypes, &[0], &[FLOAT])?;
    if let Some(shape) = shapes.resolve(&node.input[0])? {
        let rank = i64::try_from(shape.rank())
            .map_err(|_| node_error(node, "input rank does not fit i64".to_string()))?;
        if axis < -rank || axis > rank {
            return Err(node_error(
                node,
                format!("axis {axis} is outside Flatten's [-{rank}, {rank}] range"),
            ));
        }
    }
    Ok(())
}

fn validate_transpose(
    node: &NodeProto,
    opset: i64,
    dtypes: &mut RawDtypeResolver<'_>,
    shapes: &mut RawShapeResolver<'_>,
) -> Result<()> {
    require_minimum_opset(node, opset, 1)?;
    require_exact_io(node, 1, 1)?;
    let mut permutation = None;
    for attribute in &node.attribute {
        match attribute.name.as_str() {
            "perm" => {
                permutation = Some(require_ints_attribute(node, attribute)?);
            }
            _ => return Err(unknown_attribute(node, attribute, opset)),
        }
    }
    require_layout_data_type(node, dtypes, &[0], &[FLOAT, INT64])?;
    if let Some(permutation) = permutation {
        let mut normalized = Vec::with_capacity(permutation.len());
        for &axis in permutation {
            normalized
                .push(usize::try_from(axis).map_err(|_| {
                    node_error(node, format!("perm contains negative axis {axis}"))
                })?);
        }
        normalized.sort_unstable();
        if normalized != (0..permutation.len()).collect::<Vec<_>>() {
            return Err(node_error(
                node,
                format!(
                    "perm {:?} is not a permutation of 0..{}",
                    permutation,
                    permutation.len()
                ),
            ));
        }
        if let Some(input_shape) = shapes.resolve(&node.input[0])? {
            if permutation.len() != input_shape.rank() {
                return Err(node_error(
                    node,
                    format!(
                        "perm {:?} has length {}, but the authenticated input rank is {}",
                        permutation,
                        permutation.len(),
                        input_shape.rank()
                    ),
                ));
            }
        }
    }
    Ok(())
}

fn validate_squeeze(
    node: &NodeProto,
    opset: i64,
    dtypes: &mut RawDtypeResolver<'_>,
    shapes: &mut RawShapeResolver<'_>,
) -> Result<()> {
    require_minimum_opset(node, opset, 1)?;
    if opset < 13 {
        require_exact_io(node, 1, 1)?;
        for attribute in &node.attribute {
            match attribute.name.as_str() {
                "axes" => {
                    let axes = require_ints_attribute(node, attribute)?;
                    require_unique_axes(node, axes, "axes")?;
                    validate_axes_if_rank_known(node, axes, shapes, &node.input[0], false)?;
                }
                _ => return Err(unknown_attribute(node, attribute, opset)),
            }
        }
    } else {
        require_optional_second_input(node)?;
        if node.output.len() != 1 {
            return Err(node_error(
                node,
                format!("requires exactly one output, got {:?}", node.output),
            ));
        }
        require_no_attributes(node, opset)?;
        if let Some(axes) = node.input.get(1).filter(|name| !name.is_empty()) {
            require_dtype(node, dtypes, axes, "axes input", INT64)?;
            require_rank(node, shapes, axes, "axes input", 1)?;
        }
    }
    require_layout_data_type(node, dtypes, &[0], &[FLOAT, INT64])
}

fn validate_unsqueeze(
    node: &NodeProto,
    opset: i64,
    dtypes: &mut RawDtypeResolver<'_>,
    shapes: &mut RawShapeResolver<'_>,
) -> Result<()> {
    require_minimum_opset(node, opset, 1)?;
    if opset < 13 {
        require_exact_io(node, 1, 1)?;
        if node.attribute.len() != 1 || node.attribute[0].name != "axes" {
            return Err(node_error(
                node,
                format!(
                    "requires exactly one INTS 'axes' attribute before opset 13, got {:?}",
                    node.attribute
                        .iter()
                        .map(|attribute| attribute.name.as_str())
                        .collect::<Vec<_>>()
                ),
            ));
        }
        let axes = require_ints_attribute(node, &node.attribute[0])?;
        require_unique_axes(node, axes, "axes")?;
        validate_axes_if_rank_known(node, axes, shapes, &node.input[0], true)?;
    } else {
        require_exact_io(node, 2, 1)?;
        require_no_attributes(node, opset)?;
        require_dtype(node, dtypes, &node.input[1], "axes input", INT64)?;
        require_rank(node, shapes, &node.input[1], "axes input", 1)?;
    }
    require_layout_data_type(node, dtypes, &[0], &[FLOAT, INT64])
}

fn validate_slice(
    node: &NodeProto,
    opset: i64,
    dtypes: &mut RawDtypeResolver<'_>,
    shapes: &mut RawShapeResolver<'_>,
) -> Result<()> {
    require_minimum_opset(node, opset, 1)?;
    if opset < 10 {
        require_exact_io(node, 1, 1)?;
        let mut starts = None;
        let mut ends = None;
        let mut axes = None;
        for attribute in &node.attribute {
            match attribute.name.as_str() {
                "starts" => starts = Some(require_ints_attribute(node, attribute)?),
                "ends" => ends = Some(require_ints_attribute(node, attribute)?),
                "axes" => axes = Some(require_ints_attribute(node, attribute)?),
                _ => return Err(unknown_attribute(node, attribute, opset)),
            }
        }
        let starts =
            starts.ok_or_else(|| node_error(node, "is missing required starts".to_string()))?;
        let ends = ends.ok_or_else(|| node_error(node, "is missing required ends".to_string()))?;
        if starts.len() != ends.len() || axes.is_some_and(|axes| axes.len() != starts.len()) {
            return Err(node_error(
                node,
                "requires starts, ends, and optional axes to have equal lengths".to_string(),
            ));
        }
        if let Some(axes) = axes {
            require_unique_axes(node, axes, "axes")?;
            validate_axes_if_rank_known(node, axes, shapes, &node.input[0], false)?;
        }
    } else {
        if !(3..=5).contains(&node.input.len())
            || node.input[..3].iter().any(String::is_empty)
            || node.output.len() != 1
            || node.output[0].is_empty()
        {
            return Err(node_error(
                node,
                format!(
                    "has an invalid opset-{opset} Slice signature: inputs {:?}, outputs {:?}",
                    node.input, node.output
                ),
            ));
        }
        require_no_attributes(node, opset)?;
        let index_dtype = require_dtype_in(
            node,
            dtypes,
            &node.input[1],
            "starts input",
            &[INT32, INT64],
        )?;
        for (index, role) in [(1, "starts"), (2, "ends"), (3, "axes"), (4, "steps")] {
            let Some(value) = node.input.get(index).filter(|name| !name.is_empty()) else {
                continue;
            };
            require_dtype(node, dtypes, value, &format!("{role} input"), index_dtype)?;
            require_rank(node, shapes, value, &format!("{role} input"), 1)?;
        }
    }
    require_layout_data_type(node, dtypes, &[0], &[FLOAT, INT64])
}

fn validate_concat(
    node: &NodeProto,
    opset: i64,
    dtypes: &mut RawDtypeResolver<'_>,
    shapes: &mut RawShapeResolver<'_>,
) -> Result<()> {
    require_minimum_opset(node, opset, 1)?;
    if node.input.len() < 2
        || node.input.iter().any(String::is_empty)
        || node.output.len() != 1
        || node.output[0].is_empty()
    {
        return Err(node_error(
            node,
            format!(
                "requires at least two non-empty inputs and one non-empty output for ny's represented subset; got inputs {:?}, outputs {:?}",
                node.input, node.output
            ),
        ));
    }
    // Concat-1 made axis optional with default 1, while ny's legacy LayerSpec
    // default is 0.  Its explicit-axis subset is nevertheless represented;
    // Concat-4 made the attribute required for every valid model.
    if node.attribute.len() != 1 || node.attribute[0].name != "axis" {
        return Err(node_error(
            node,
            if opset == 1 {
                "requires an explicit INT axis attribute because ny cannot reinterpret Concat-1's omitted-axis default of 1 as its internal default of 0".to_string()
            } else {
                "requires exactly one INT axis attribute".to_string()
            },
        ));
    }
    let axis = require_int_attribute(node, &node.attribute[0])?;
    let input_indices = (0..node.input.len()).collect::<Vec<_>>();
    let allowed = if opset == 1 {
        &[FLOAT][..]
    } else {
        &[FLOAT, INT64][..]
    };
    require_layout_data_type(node, dtypes, &input_indices, allowed)?;

    let mut rank = None;
    for input in &node.input {
        if let Some(shape) = shapes.resolve(input)? {
            if rank.is_some_and(|rank| rank != shape.rank()) {
                return Err(node_error(
                    node,
                    "has inputs with conflicting authenticated ranks".to_string(),
                ));
            }
            rank = Some(shape.rank());
        }
    }
    if let Some(rank) = rank {
        validate_axis(node, axis, rank, false)?;
    }
    Ok(())
}

fn validate_split(
    node: &NodeProto,
    opset: i64,
    dtypes: &mut RawDtypeResolver<'_>,
    shapes: &mut RawShapeResolver<'_>,
) -> Result<()> {
    // Split-1's optional second input shared the data tensor type and has
    // different semantics from the modern attribute/input forms.
    require_minimum_opset(node, opset, 2)?;
    if node.output.is_empty() || node.output.iter().any(String::is_empty) {
        return Err(node_error(
            node,
            "requires one or more non-empty outputs".to_string(),
        ));
    }

    if opset < 13 {
        if node.input.len() != 1 || node.input[0].is_empty() {
            return Err(node_error(
                node,
                format!(
                    "requires one data input before opset 13, got {:?}",
                    node.input
                ),
            ));
        }
        for attribute in &node.attribute {
            match attribute.name.as_str() {
                "axis" => {
                    require_int_attribute(node, attribute)?;
                }
                "split" => {
                    let splits = require_ints_attribute(node, attribute)?;
                    if splits.len() != node.output.len() || splits.iter().any(|size| *size < 0) {
                        return Err(node_error(
                            node,
                            "split must contain one non-negative extent per output".to_string(),
                        ));
                    }
                }
                _ => return Err(unknown_attribute(node, attribute, opset)),
            }
        }
    } else {
        require_optional_second_input(node)?;
        let mut axis = 0;
        let mut num_outputs = None;
        for attribute in &node.attribute {
            match attribute.name.as_str() {
                "axis" => {
                    axis = require_int_attribute(node, attribute)?;
                }
                "num_outputs" if opset >= 18 => {
                    let value = require_int_attribute(node, attribute)?;
                    if value <= 0 || usize::try_from(value).ok() != Some(node.output.len()) {
                        return Err(node_error(
                            node,
                            format!(
                                "num_outputs {value} must be positive and equal the {} authored outputs",
                                node.output.len()
                            ),
                        ));
                    }
                    num_outputs = Some(value as usize);
                }
                _ => return Err(unknown_attribute(node, attribute, opset)),
            }
        }
        if let Some(split) = node.input.get(1).filter(|name| !name.is_empty()) {
            if num_outputs.is_some() {
                return Err(node_error(
                    node,
                    "cannot supply both split input and num_outputs".to_string(),
                ));
            }
            require_dtype(node, dtypes, split, "split input", INT64)?;
            require_rank(node, shapes, split, "split input", 1)?;
        }
        if let Some(num_outputs) = num_outputs {
            let data_shape = shapes.resolve(&node.input[0])?.ok_or_else(|| {
                node_error(
                    node,
                    "cannot authenticate Split-18 num_outputs against the data shape".to_string(),
                )
            })?;
            let axis_index = normalize_axis(node, axis, data_shape.rank())?;
            let extent = data_shape.0[axis_index].ok_or_else(|| {
                node_error(
                    node,
                    format!(
                        "cannot authenticate Split-18 num_outputs against symbolic axis {axis}"
                    ),
                )
            })?;
            let num_outputs = i64::try_from(num_outputs)
                .map_err(|_| node_error(node, "num_outputs does not fit i64".to_string()))?;
            if extent % num_outputs != 0 {
                return Err(NyError::UnsupportedOp(format!(
                    "standard ONNX Split node '{}' has axis extent {extent} not divisible by num_outputs={num_outputs}; ny's equal-split lowering does not represent Split-18's smaller final chunk",
                    node.name
                )));
            }
        }
    }
    require_layout_data_type(node, dtypes, &[0], &[FLOAT])
}

fn validate_tile(
    node: &NodeProto,
    opset: i64,
    dtypes: &mut RawDtypeResolver<'_>,
    shapes: &mut RawShapeResolver<'_>,
) -> Result<()> {
    // Tile-1 used scalar `tiles` and `axis` inputs.  The repeats-vector schema
    // implemented by ny was introduced at opset 6.
    require_minimum_opset(node, opset, 6)?;
    require_exact_io(node, 2, 1)?;
    require_no_attributes(node, opset)?;
    require_layout_data_type(node, dtypes, &[0], &[FLOAT])?;
    require_dtype(node, dtypes, &node.input[1], "repeats input", INT64)?;
    require_rank(node, shapes, &node.input[1], "repeats input", 1)?;

    if let (Some(data_shape), Some(repeats_shape)) = (
        shapes.resolve(&node.input[0])?,
        shapes.resolve(&node.input[1])?,
    ) {
        if let Some(length) = repeats_shape.vector_length() {
            if length != data_shape.rank() {
                return Err(node_error(
                    node,
                    format!(
                        "repeats length {length} does not match authenticated data rank {}",
                        data_shape.rank()
                    ),
                ));
            }
        }
    }
    Ok(())
}

fn require_layout_data_type(
    node: &NodeProto,
    dtypes: &mut RawDtypeResolver<'_>,
    data_input_indices: &[usize],
    allowed: &[i32],
) -> Result<()> {
    let mut dtype = None;
    for index in data_input_indices {
        let value = &node.input[*index];
        let candidate = require_dtype_in(node, dtypes, value, "data input", allowed)?;
        if dtype.is_some_and(|known| known != candidate) {
            return Err(node_error(
                node,
                format!(
                    "requires all data inputs to share one dtype, got {known_dtypes:?}",
                    known_dtypes = [dtype, Some(candidate)]
                ),
            ));
        }
        dtype = Some(candidate);
    }
    let dtype = dtype.ok_or_else(|| node_error(node, "has no data input".to_string()))?;
    for output in &node.output {
        require_dtype(node, dtypes, output, "data output", dtype)?;
    }
    Ok(())
}

fn require_dtype(
    node: &NodeProto,
    dtypes: &mut RawDtypeResolver<'_>,
    value: &str,
    role: &str,
    expected: i32,
) -> Result<()> {
    let actual = require_resolved_dtype(node, dtypes, value, role)?;
    if actual != expected {
        return Err(node_error(
            node,
            format!("requires {role} '{value}' to have dtype {expected}, got {actual}"),
        ));
    }
    Ok(())
}

fn require_dtype_in(
    node: &NodeProto,
    dtypes: &mut RawDtypeResolver<'_>,
    value: &str,
    role: &str,
    allowed: &[i32],
) -> Result<i32> {
    let actual = require_resolved_dtype(node, dtypes, value, role)?;
    if !allowed.contains(&actual) {
        return Err(NyError::UnsupportedOp(format!(
            "standard ONNX {} node '{}' uses {role} '{}' with dtype {actual}; ny represents only dtypes {allowed:?} on this structural path",
            node.op_type, node.name, value
        )));
    }
    Ok(actual)
}

fn require_resolved_dtype(
    node: &NodeProto,
    dtypes: &mut RawDtypeResolver<'_>,
    value: &str,
    role: &str,
) -> Result<i32> {
    dtypes.resolve(value)?.ok_or_else(|| {
        node_error(
            node,
            format!("cannot authenticate dtype of {role} '{value}' before folding"),
        )
    })
}

fn require_rank(
    node: &NodeProto,
    shapes: &mut RawShapeResolver<'_>,
    value: &str,
    role: &str,
    expected: usize,
) -> Result<()> {
    let shape = shapes.resolve(value)?.ok_or_else(|| {
        node_error(
            node,
            format!("cannot authenticate rank of {role} '{value}' before folding"),
        )
    })?;
    if shape.rank() != expected {
        return Err(node_error(
            node,
            format!(
                "requires {role} '{value}' to have rank {expected}, got shape {:?}",
                shape.0
            ),
        ));
    }
    Ok(())
}

fn require_minimum_opset(node: &NodeProto, opset: i64, minimum: i64) -> Result<()> {
    if opset < minimum {
        return Err(NyError::UnsupportedOp(format!(
            "standard ONNX {} node '{}' uses opset {opset}; ny's represented schema requires opset {minimum} or newer",
            node.op_type, node.name
        )));
    }
    Ok(())
}

fn require_exact_io(node: &NodeProto, inputs: usize, outputs: usize) -> Result<()> {
    if node.input.len() != inputs
        || node.input.iter().any(String::is_empty)
        || node.output.len() != outputs
        || node.output.iter().any(String::is_empty)
    {
        return Err(node_error(
            node,
            format!(
                "requires exactly {inputs} non-empty input(s) and {outputs} non-empty output(s); got inputs {:?}, outputs {:?}",
                node.input, node.output
            ),
        ));
    }
    Ok(())
}

fn require_optional_second_input(node: &NodeProto) -> Result<()> {
    if !(1..=2).contains(&node.input.len())
        || node.input[0].is_empty()
        || node.output.is_empty()
        || node.output.iter().any(String::is_empty)
    {
        return Err(node_error(
            node,
            format!(
                "requires one data input, one optional control input, and non-empty output(s); got inputs {:?}, outputs {:?}",
                node.input, node.output
            ),
        ));
    }
    Ok(())
}

fn require_no_attributes(node: &NodeProto, opset: i64) -> Result<()> {
    if let Some(attribute) = node.attribute.first() {
        return Err(unknown_attribute(node, attribute, opset));
    }
    Ok(())
}

fn require_attribute_type(
    node: &NodeProto,
    attribute: &AttributeProto,
    expected: i32,
) -> Result<()> {
    if attribute.r#type != expected {
        return Err(node_error(
            node,
            format!(
                "attribute '{}' must have ONNX AttributeProto type {expected}, got {}",
                attribute.name, attribute.r#type
            ),
        ));
    }
    Ok(())
}

fn require_int_attribute(node: &NodeProto, attribute: &AttributeProto) -> Result<i64> {
    require_attribute_type(node, attribute, onnx_proto::attribute_type::INT)?;
    Ok(attribute.i_value())
}

fn require_boolean_int_attribute(node: &NodeProto, attribute: &AttributeProto) -> Result<i64> {
    let value = require_int_attribute(node, attribute)?;
    if !matches!(value, 0 | 1) {
        return Err(node_error(
            node,
            format!(
                "attribute '{}' must be INT 0 or 1, got {value}",
                attribute.name
            ),
        ));
    }
    Ok(value)
}

fn require_ints_attribute<'a>(
    node: &NodeProto,
    attribute: &'a AttributeProto,
) -> Result<&'a [i64]> {
    require_attribute_type(node, attribute, onnx_proto::attribute_type::INTS)?;
    Ok(&attribute.ints)
}

fn require_unique_axes(node: &NodeProto, axes: &[i64], role: &str) -> Result<()> {
    let mut unique = HashSet::new();
    if let Some(axis) = axes.iter().find(|axis| !unique.insert(**axis)) {
        return Err(node_error(
            node,
            format!("{role} contains duplicate axis {axis}"),
        ));
    }
    Ok(())
}

fn validate_axes_if_rank_known(
    node: &NodeProto,
    axes: &[i64],
    shapes: &mut RawShapeResolver<'_>,
    data: &str,
    output_rank_relative: bool,
) -> Result<()> {
    let Some(shape) = shapes.resolve(data)? else {
        return Ok(());
    };
    let rank = if output_rank_relative {
        shape
            .rank()
            .checked_add(axes.len())
            .ok_or_else(|| node_error(node, "output rank overflows usize".to_string()))?
    } else {
        shape.rank()
    };
    for &axis in axes {
        validate_axis(node, axis, rank, false)?;
    }
    Ok(())
}

fn validate_axis(node: &NodeProto, axis: i64, rank: usize, inclusive_upper: bool) -> Result<()> {
    let rank =
        i64::try_from(rank).map_err(|_| node_error(node, "rank does not fit i64".to_string()))?;
    let upper = if inclusive_upper { rank } else { rank - 1 };
    if axis < -rank || axis > upper {
        return Err(node_error(
            node,
            format!(
                "axis {axis} is outside the authenticated rank-{rank} range [-{rank}, {upper}]"
            ),
        ));
    }
    Ok(())
}

fn normalize_axis(node: &NodeProto, axis: i64, rank: usize) -> Result<usize> {
    validate_axis(node, axis, rank, false)?;
    if axis >= 0 {
        usize::try_from(axis).map_err(|_| node_error(node, "axis does not fit usize".to_string()))
    } else {
        let rank = i64::try_from(rank)
            .map_err(|_| node_error(node, "rank does not fit i64".to_string()))?;
        usize::try_from(rank + axis)
            .map_err(|_| node_error(node, "normalized axis does not fit usize".to_string()))
    }
}

fn unknown_attribute(node: &NodeProto, attribute: &AttributeProto, opset: i64) -> NyError {
    node_error(
        node,
        format!(
            "does not define attribute '{}' at opset {opset}",
            attribute.name
        ),
    )
}

fn node_error(node: &NodeProto, detail: String) -> NyError {
    NyError::ModelLoad(format!(
        "standard ONNX {} node '{}' {detail}",
        node.op_type, node.name
    ))
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RawShape(Vec<Option<i64>>);

impl RawShape {
    fn rank(&self) -> usize {
        self.0.len()
    }

    fn vector_length(&self) -> Option<usize> {
        if self.rank() != 1 {
            return None;
        }
        usize::try_from(self.0[0]?).ok()
    }
}

/// Shape proof sufficient for authenticating rank-1 structural control
/// tensors.  It preserves unknown/symbolic extents while still proving rank.
struct RawShapeResolver<'a> {
    graph: &'a GraphProto,
    authored: HashMap<&'a str, Vec<RawShape>>,
    producers: HashMap<&'a str, Vec<usize>>,
    memo: HashMap<String, Option<RawShape>>,
}

impl<'a> RawShapeResolver<'a> {
    fn new(graph: &'a GraphProto) -> Self {
        let mut authored: HashMap<&str, Vec<RawShape>> = HashMap::new();
        for initializer in &graph.initializer {
            if !initializer.name.is_empty() {
                authored
                    .entry(initializer.name.as_str())
                    .or_default()
                    .push(RawShape(
                        initializer.dims.iter().copied().map(Some).collect(),
                    ));
            }
        }
        for info in graph
            .input
            .iter()
            .chain(graph.output.iter())
            .chain(graph.value_info().iter())
        {
            if !info.name.is_empty() {
                if let Some(shape) = value_info_shape(info) {
                    authored.entry(info.name.as_str()).or_default().push(shape);
                }
            }
        }
        let mut producers: HashMap<&str, Vec<usize>> = HashMap::new();
        for (index, node) in graph.node.iter().enumerate() {
            for output in node.output.iter().filter(|output| !output.is_empty()) {
                producers.entry(output.as_str()).or_default().push(index);
            }
        }
        Self {
            graph,
            authored,
            producers,
            memo: HashMap::new(),
        }
    }

    fn resolve(&mut self, value: &str) -> Result<Option<RawShape>> {
        self.resolve_inner(value, &mut HashSet::new())
    }

    fn resolve_inner(
        &mut self,
        value: &str,
        active: &mut HashSet<String>,
    ) -> Result<Option<RawShape>> {
        if let Some(shape) = self.memo.get(value) {
            return Ok(shape.clone());
        }
        if !active.insert(value.to_string()) {
            return Err(NyError::ModelLoad(format!(
                "cycle encountered while proving raw ONNX shape of '{value}'"
            )));
        }
        let mut candidates = self.authored.get(value).cloned().unwrap_or_default();
        if let Some(producers) = self.producers.get(value).cloned() {
            if producers.len() != 1 {
                return Err(NyError::ModelLoad(format!(
                    "cannot prove raw ONNX shape of '{value}': it has {} producers",
                    producers.len()
                )));
            }
            if let Some(shape) = self.infer_node_shape(producers[0], value, active)? {
                candidates.push(shape);
            }
        }
        active.remove(value);

        let mut merged = None;
        for candidate in candidates {
            merged = Some(match merged {
                None => candidate,
                Some(current) => merge_shapes(current, candidate).ok_or_else(|| {
                    NyError::ModelLoad(format!("conflicting raw ONNX shapes for value '{value}'"))
                })?,
            });
        }
        self.memo.insert(value.to_string(), merged.clone());
        Ok(merged)
    }

    fn infer_node_shape(
        &mut self,
        node_index: usize,
        output: &str,
        active: &mut HashSet<String>,
    ) -> Result<Option<RawShape>> {
        let node = self.graph.node[node_index].clone();
        if !is_standard_onnx_domain(&node.domain) {
            return Ok(None);
        }
        if node.op_type == "Split" && node.output.iter().any(|candidate| candidate == output) {
            return self.first_input_shape(&node, active);
        }
        if node.output.len() != 1 || node.output[0] != output {
            return Ok(None);
        }

        match node.op_type.as_str() {
            "Constant" => Ok(constant_shape(&node)),
            "Shape" => {
                let extent = match node.input.first().filter(|input| !input.is_empty()) {
                    Some(input) => self.resolve_inner(input, active)?.map(|shape| {
                        let rank = shape.rank() as i64;
                        let clamp = |value: i64| {
                            let relative = if value < 0 {
                                value.saturating_add(rank)
                            } else {
                                value
                            };
                            relative.clamp(0, rank)
                        };
                        let start = node
                            .attribute
                            .iter()
                            .find(|attribute| attribute.name == "start")
                            .map(AttributeProto::i_value)
                            .map(clamp)
                            .unwrap_or(0);
                        let end = node
                            .attribute
                            .iter()
                            .find(|attribute| attribute.name == "end")
                            .map(AttributeProto::i_value)
                            .map(clamp)
                            .unwrap_or(rank);
                        end.saturating_sub(start).max(0)
                    }),
                    None => None,
                };
                Ok(Some(RawShape(vec![extent])))
            }
            "Identity" | "Cast" | "Transpose" | "Slice" | "Tile" => {
                self.first_input_shape(&node, active)
            }
            "Flatten" => Ok(Some(RawShape(vec![None, None]))),
            "Concat" => self.uniform_input_rank_shape(&node, active),
            "Gather" => self.gather_shape(&node, active),
            "Reshape" => self.reshape_shape(&node, active),
            "Squeeze" => self.squeeze_shape(&node, active),
            "Unsqueeze" => self.unsqueeze_shape(&node, active),
            "Expand" => self.expand_shape(&node, active),
            // Range always returns a one-dimensional sequence. Its extent is
            // value-dependent, but rank alone is enough to authenticate a
            // downstream Reshape/ConstantOfShape control tensor before fold.
            "Range" => Ok(Some(RawShape(vec![None]))),
            "Add" | "Sub" | "Mul" | "Div" | "Pow" | "Min" | "Max" | "Equal" | "Less"
            | "Greater" | "LessOrEqual" | "GreaterOrEqual" | "And" | "Or" | "Xor" | "Where" => {
                self.broadcast_rank_shape(&node, active)
            }
            "ConstantOfShape" => self.constant_of_shape_rank(&node, active),
            _ => Ok(None),
        }
    }

    /// ONNX multidirectional (NumPy) broadcasting gives the result the maximum
    /// rank of its operands.  Extents stay unproven, which is all a rank-1
    /// control-tensor check needs.  The legacy unidirectional forms broadcast B
    /// into A, and A's rank is already that maximum, so the bound is exact
    /// there too.
    fn broadcast_rank_shape(
        &mut self,
        node: &NodeProto,
        active: &mut HashSet<String>,
    ) -> Result<Option<RawShape>> {
        let mut rank: Option<usize> = None;
        for input in node.input.iter().filter(|input| !input.is_empty()) {
            let Some(shape) = self.resolve_inner(input, active)? else {
                return Ok(None);
            };
            rank = Some(match rank {
                None => shape.rank(),
                Some(current) => current.max(shape.rank()),
            });
        }
        Ok(rank.map(|rank| RawShape(vec![None; rank])))
    }

    /// `ConstantOfShape`'s output rank is the ELEMENT COUNT of its rank-1 INT64
    /// operand.  That count is the operand's own extent, so the rank follows
    /// from shape metadata alone — no operand value is read here.
    fn constant_of_shape_rank(
        &mut self,
        node: &NodeProto,
        active: &mut HashSet<String>,
    ) -> Result<Option<RawShape>> {
        let Some(input) = node.input.first().filter(|input| !input.is_empty()) else {
            return Ok(None);
        };
        let Some(shape) = self.resolve_inner(input, active)? else {
            return Ok(None);
        };
        Ok(shape.vector_length().map(|len| RawShape(vec![None; len])))
    }

    fn first_input_shape(
        &mut self,
        node: &NodeProto,
        active: &mut HashSet<String>,
    ) -> Result<Option<RawShape>> {
        let Some(input) = node.input.first().filter(|input| !input.is_empty()) else {
            return Ok(None);
        };
        self.resolve_inner(input, active)
    }

    fn uniform_input_rank_shape(
        &mut self,
        node: &NodeProto,
        active: &mut HashSet<String>,
    ) -> Result<Option<RawShape>> {
        let mut rank = None;
        for input in node.input.iter().filter(|input| !input.is_empty()) {
            let Some(shape) = self.resolve_inner(input, active)? else {
                return Ok(None);
            };
            if rank.is_some_and(|rank| rank != shape.rank()) {
                return Ok(None);
            }
            rank = Some(shape.rank());
        }
        Ok(rank.map(|rank| RawShape(vec![None; rank])))
    }

    fn gather_shape(
        &mut self,
        node: &NodeProto,
        active: &mut HashSet<String>,
    ) -> Result<Option<RawShape>> {
        if node.input.len() != 2 {
            return Ok(None);
        }
        let Some(data) = self.resolve_inner(&node.input[0], active)? else {
            return Ok(None);
        };
        let Some(indices) = self.resolve_inner(&node.input[1], active)? else {
            return Ok(None);
        };
        let Some(rank) = data
            .rank()
            .checked_add(indices.rank())
            .and_then(|rank| rank.checked_sub(1))
        else {
            return Ok(None);
        };
        Ok(Some(RawShape(vec![None; rank])))
    }

    fn reshape_shape(
        &mut self,
        node: &NodeProto,
        active: &mut HashSet<String>,
    ) -> Result<Option<RawShape>> {
        let Some(target) = node.input.get(1).filter(|input| !input.is_empty()) else {
            return Ok(None);
        };
        let Some(target_shape) = self.resolve_inner(target, active)? else {
            return Ok(None);
        };
        Ok(target_shape
            .vector_length()
            .map(|rank| RawShape(vec![None; rank])))
    }

    fn squeeze_shape(
        &mut self,
        node: &NodeProto,
        active: &mut HashSet<String>,
    ) -> Result<Option<RawShape>> {
        let Some(data) = node.input.first().filter(|input| !input.is_empty()) else {
            return Ok(None);
        };
        let Some(input) = self.resolve_inner(data, active)? else {
            return Ok(None);
        };
        let removed = if let Some(axes) = node.input.get(1).filter(|input| !input.is_empty()) {
            self.resolve_inner(axes, active)?
                .and_then(|shape| shape.vector_length())
        } else if let Some(attribute) = node
            .attribute
            .iter()
            .find(|attribute| attribute.name == "axes")
        {
            (attribute.r#type == onnx_proto::attribute_type::INTS).then_some(attribute.ints.len())
        } else if input.0.iter().all(Option::is_some) {
            Some(
                input
                    .0
                    .iter()
                    .filter(|dimension| **dimension == Some(1))
                    .count(),
            )
        } else {
            None
        };
        Ok(removed
            .and_then(|removed| input.rank().checked_sub(removed))
            .map(|rank| RawShape(vec![None; rank])))
    }

    fn unsqueeze_shape(
        &mut self,
        node: &NodeProto,
        active: &mut HashSet<String>,
    ) -> Result<Option<RawShape>> {
        let Some(input) = self.first_input_shape(node, active)? else {
            return Ok(None);
        };
        let inserted = if let Some(axes) = node.input.get(1).filter(|input| !input.is_empty()) {
            self.resolve_inner(axes, active)?
                .and_then(|shape| shape.vector_length())
        } else {
            node.attribute
                .iter()
                .find(|attribute| {
                    attribute.name == "axes" && attribute.r#type == onnx_proto::attribute_type::INTS
                })
                .map(|attribute| attribute.ints.len())
        };
        Ok(inserted
            .and_then(|inserted| input.rank().checked_add(inserted))
            .map(|rank| RawShape(vec![None; rank])))
    }

    fn expand_shape(
        &mut self,
        node: &NodeProto,
        active: &mut HashSet<String>,
    ) -> Result<Option<RawShape>> {
        if node.input.len() != 2 {
            return Ok(None);
        }
        let Some(data) = self.resolve_inner(&node.input[0], active)? else {
            return Ok(None);
        };
        let Some(target) = self.resolve_inner(&node.input[1], active)? else {
            return Ok(None);
        };
        Ok(target
            .vector_length()
            .map(|target_rank| RawShape(vec![None; data.rank().max(target_rank)])))
    }
}

fn merge_shapes(left: RawShape, right: RawShape) -> Option<RawShape> {
    if left.rank() != right.rank() {
        return None;
    }
    left.0
        .into_iter()
        .zip(right.0)
        .map(|(left, right)| match (left, right) {
            (Some(left), Some(right)) if left != right => None,
            (Some(value), _) | (_, Some(value)) => Some(Some(value)),
            (None, None) => Some(None),
        })
        .collect::<Option<Vec<_>>>()
        .map(RawShape)
}

fn value_info_shape(info: &onnx_proto::ValueInfoProto) -> Option<RawShape> {
    use onnx_proto::tensor_shape_proto::dimension::Value;

    let dimensions = &info
        .r#type
        .as_ref()?
        .tensor_type
        .as_ref()?
        .shape
        .as_ref()?
        .dim;
    Some(RawShape(
        dimensions
            .iter()
            .map(|dimension| match dimension.value.as_ref() {
                Some(Value::DimValue(value)) if *value >= 0 => Some(*value),
                Some(Value::DimValue(_)) | Some(Value::DimParam(_)) | None => None,
            })
            .collect(),
    ))
}

fn constant_shape(node: &NodeProto) -> Option<RawShape> {
    if !node.input.is_empty() {
        return None;
    }
    let shapes = node
        .attribute
        .iter()
        .filter_map(
            |attribute| match (attribute.name.as_str(), attribute.r#type) {
                ("value", onnx_proto::attribute_type::TENSOR) => attribute
                    .t
                    .as_ref()
                    .map(|tensor| RawShape(tensor.dims.iter().copied().map(Some).collect())),
                ("value_float", onnx_proto::attribute_type::FLOAT)
                | ("value_int", onnx_proto::attribute_type::INT) => Some(RawShape(Vec::new())),
                ("value_floats", onnx_proto::attribute_type::FLOATS) => Some(RawShape(vec![Some(
                    i64::try_from(attribute.floats.len()).ok()?,
                )])),
                ("value_ints", onnx_proto::attribute_type::INTS) => Some(RawShape(vec![Some(
                    i64::try_from(attribute.ints.len()).ok()?,
                )])),
                _ => None,
            },
        )
        .collect::<Vec<_>>();
    match shapes.as_slice() {
        [shape] => Some(shape.clone()),
        _ => None,
    }
}

/// INT64 values are useful inside ONNX shape programs, but ny's runtime tensor
/// graph stores activation values as f32.  Admit layout-only INT64 chains only
/// when they originate from authored constants/Shape metadata and terminate in
/// an integer control position.  In particular they may not become model
/// outputs, flow through a FLOAT Cast, or enter an ordinary activation input.
fn validate_int64_structural_control_paths<'graph>(
    graph: &'graph GraphProto,
    dtypes: &mut RawDtypeResolver<'graph>,
    literals: &LiteralCone,
    exemptions: &mut LiteralExemptions,
) -> Result<()> {
    let mut audit = Int64ControlPathAudit::new(graph, dtypes, literals, exemptions);

    // Seed every authored INT64 constant, not just the outputs of the
    // structural operators above.  An initializer can otherwise enter an
    // unaudited value-preserving producer (notably Gather) as data and emerge
    // on ny's FLOAT runtime path without ever crossing an audited output.
    let int64_initializers = graph
        .initializer
        .iter()
        .filter(|tensor| tensor.data_type == INT64 && !tensor.name.is_empty())
        .map(|tensor| tensor.name.clone())
        .collect::<Vec<_>>();
    for value in int64_initializers {
        audit.require_control_only_consumers(&value, &mut HashSet::new())?;
    }

    for node in &graph.node {
        if !is_standard_onnx_domain(&node.domain)
            || !AUDITED_OPERATORS.contains(&node.op_type.as_str())
        {
            continue;
        }
        for index in structural_data_input_indices(node) {
            let Some(value) = node.input.get(index).filter(|value| !value.is_empty()) else {
                continue;
            };
            if audit.dtypes.resolve(value)? == Some(INT64) && node.op_type != "Shape" {
                audit.require_static_source(value, &mut HashSet::new())?;
            }
        }
    }

    // Likewise seed every standard-node INT64 result.  This covers integer
    // layout/control chains whose producer (for example Gather) is validated
    // by another schema module while preserving Shape -> Gather -> control:
    // Gather's data edge is an exact INT64 layout edge and its result must
    // still terminate at a declared integer-control input.
    for node in &graph.node {
        if !is_standard_onnx_domain(&node.domain) {
            continue;
        }
        for output in node.output.iter().filter(|output| !output.is_empty()) {
            if audit.dtypes.resolve(output)? == Some(INT64) {
                audit.require_control_only_consumers(output, &mut HashSet::new())?;
            }
        }
    }
    Ok(())
}

fn structural_data_input_indices(node: &NodeProto) -> Vec<usize> {
    // Concat and Range are the variadic cases: every input is a data edge, not
    // just input 0.
    if node.op_type == "Concat" || node.op_type == "Range" {
        (0..node.input.len()).collect()
    } else {
        vec![0]
    }
}

struct Int64ControlPathAudit<'graph, 'dtype, 'cone> {
    graph: &'graph GraphProto,
    dtypes: &'dtype mut RawDtypeResolver<'graph>,
    literals: &'cone LiteralCone,
    exemptions: &'cone mut LiteralExemptions,
    initializers: HashSet<&'graph str>,
    producers: HashMap<&'graph str, Vec<usize>>,
    graph_outputs: HashSet<&'graph str>,
}

impl<'graph, 'dtype, 'cone> Int64ControlPathAudit<'graph, 'dtype, 'cone> {
    fn new(
        graph: &'graph GraphProto,
        dtypes: &'dtype mut RawDtypeResolver<'graph>,
        literals: &'cone LiteralCone,
        exemptions: &'cone mut LiteralExemptions,
    ) -> Self {
        let initializers = graph
            .initializer
            .iter()
            .filter(|tensor| !tensor.name.is_empty())
            .map(|tensor| tensor.name.as_str())
            .collect();
        let mut producers: HashMap<&str, Vec<usize>> = HashMap::new();
        for (index, node) in graph.node.iter().enumerate() {
            for output in node.output.iter().filter(|output| !output.is_empty()) {
                producers.entry(output.as_str()).or_default().push(index);
            }
        }
        let graph_outputs = graph
            .output
            .iter()
            .filter(|output| !output.name.is_empty())
            .map(|output| output.name.as_str())
            .collect();
        Self {
            graph,
            dtypes,
            literals,
            exemptions,
            initializers,
            producers,
            graph_outputs,
        }
    }

    fn require_static_source(&mut self, value: &str, active: &mut HashSet<String>) -> Result<()> {
        if self.initializers.contains(value) {
            return Ok(());
        }
        if !active.insert(value.to_string()) {
            return Err(NyError::ModelLoad(format!(
                "cycle encountered while authenticating INT64 structural source '{value}'"
            )));
        }
        let result = (|| {
            let producers = self.producers.get(value).cloned().unwrap_or_default();
            let [producer_index] = producers.as_slice() else {
                return Err(NyError::UnsupportedOp(format!(
                    "raw ONNX INT64 structural value '{value}' is not an authored initializer and does not have exactly one authenticated producer"
                )));
            };
            let producer = self.graph.node[*producer_index].clone();
            if !is_standard_onnx_domain(&producer.domain) {
                return Err(NyError::UnsupportedOp(format!(
                    "raw ONNX INT64 structural value '{value}' comes from unauthenticated custom-domain node '{}'",
                    producer.name
                )));
            }
            match producer.op_type.as_str() {
                "Constant" | "Shape" => Ok(()),
                "Cast" if cast_target_dtype(&producer) == Some(INT64) => {
                    self.require_all_inputs_static(&producer, active)
                }
                "Identity"
                | "Gather"
                | "ConstantOfShape"
                | "Expand"
                | "Reshape"
                | "Transpose"
                | "Squeeze"
                | "Unsqueeze"
                | "Slice"
                | "Concat"
                | "Range" => self.require_all_inputs_static(&producer, active),
                _ => Err(NyError::UnsupportedOp(format!(
                    "raw ONNX INT64 structural value '{value}' is produced by '{}' ({}), which is not an exact static shape-control producer",
                    producer.name, producer.op_type
                ))),
            }
        })();
        active.remove(value);
        result
    }

    fn require_all_inputs_static(
        &mut self,
        node: &NodeProto,
        active: &mut HashSet<String>,
    ) -> Result<()> {
        for input in node.input.iter().filter(|input| !input.is_empty()) {
            self.require_static_source(input, active)?;
        }
        Ok(())
    }

    fn require_control_only_consumers(
        &mut self,
        value: &str,
        active: &mut HashSet<String>,
    ) -> Result<()> {
        if self.graph_outputs.contains(value) {
            return Err(NyError::UnsupportedOp(format!(
                "raw ONNX INT64 structural value '{value}' is a graph output; ny exposes FLOAT32 verifier outputs and cannot route it through f32 propagation"
            )));
        }
        if !active.insert(value.to_string()) {
            return Err(NyError::ModelLoad(format!(
                "cycle encountered while authenticating consumers of INT64 structural value '{value}'"
            )));
        }
        let result = (|| {
            let consumers = self
                .graph
                .node
                .iter()
                .enumerate()
                .flat_map(|(node_index, node)| {
                    node.input
                        .iter()
                        .enumerate()
                        .filter(|(_, input)| *input == value)
                        .map(move |(input_index, _)| (node_index, input_index))
                })
                .collect::<Vec<_>>();
            for (node_index, input_index) in consumers {
                let consumer = self.graph.node[node_index].clone();
                if !is_standard_onnx_domain(&consumer.domain) {
                    return Err(NyError::UnsupportedOp(format!(
                        "raw ONNX INT64 structural value '{value}' reaches custom-domain node '{}' without an exact control contract",
                        consumer.name
                    )));
                }
                if is_int64_control_input(&consumer.op_type, input_index) {
                    continue;
                }
                if !is_int64_layout_chain_input(&consumer, input_index) {
                    let refusal = NyError::UnsupportedOp(format!(
                        "raw ONNX INT64 structural value '{value}' reaches data input {input_index} of '{}' ({}); this would leak integer values into ny's f32 runtime graph",
                        consumer.name, consumer.op_type
                    ));
                    // There is no leak when the consumer itself is an exact
                    // integer computation over authored constants: its result is
                    // decided by the model bytes, and the deferred-fold
                    // obligation requires the loader to erase it into a literal
                    // before propagation. If folding does not erase it, the
                    // refusal above is re-raised (see `literal_cone`).
                    if !is_exact_integer_literal_node(&consumer, self.literals, self.dtypes)? {
                        return Err(refusal);
                    }
                    self.exemptions.record(&consumer, &refusal);
                    continue;
                }
                for output in consumer.output.iter().filter(|output| !output.is_empty()) {
                    let dtype = self.dtypes.resolve(output)?.ok_or_else(|| {
                        NyError::ModelLoad(format!(
                            "cannot authenticate dtype of '{}' while tracing INT64 structural control path",
                            output
                        ))
                    })?;
                    if dtype != INT64 {
                        return Err(NyError::UnsupportedOp(format!(
                            "raw ONNX INT64 structural value '{value}' changes to dtype {dtype} at '{}' ({}); only exact INT64 layout chains are admitted",
                            consumer.name, consumer.op_type
                        )));
                    }
                    self.require_control_only_consumers(output, active)?;
                }
            }
            Ok(())
        })();
        active.remove(value);
        result
    }
}

fn is_int64_control_input(op_type: &str, input_index: usize) -> bool {
    matches!(
        (op_type, input_index),
        (
            "Reshape" | "Expand" | "Squeeze" | "Unsqueeze" | "Split" | "Tile",
            1
        ) | ("Slice", 1..=4)
            | ("ConstantOfShape", 0)
            | ("Range", 0..=2)
            | ("Gather" | "GatherElements" | "GatherND" | "ScatterND", 1)
            | ("Trilu", 1)
            | ("Pad", 1 | 3)
            | ("Resize", 3)
            | ("TopK" | "CumSum", 1)
            | ("ReduceSum" | "ReduceMean" | "ReduceMax" | "ReduceMin", 1)
    )
}

fn is_int64_layout_chain_input(node: &NodeProto, input_index: usize) -> bool {
    match node.op_type.as_str() {
        "Shape" | "Identity" | "Transpose" | "Squeeze" | "Unsqueeze" | "Slice" | "Reshape"
        | "Expand" => input_index == 0,
        "Concat" => true,
        "Gather" => input_index == 0,
        "Cast" => input_index == 0 && cast_target_dtype(node) == Some(INT64),
        _ => false,
    }
}

fn cast_target_dtype(node: &NodeProto) -> Option<i32> {
    let attributes = node
        .attribute
        .iter()
        .filter(|attribute| attribute.name == "to")
        .collect::<Vec<_>>();
    match attributes.as_slice() {
        [attribute] if attribute.r#type == onnx_proto::attribute_type::INT => {
            i32::try_from(attribute.i_value()).ok()
        }
        _ => None,
    }
}
