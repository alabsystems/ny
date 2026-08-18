// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Raw, versioned schemas for reductions and comparisons supported by ny.
//!
//! These checks intentionally precede constant folding.  Folding can erase an
//! axes input, an obsolete broadcast attribute, or the authored tensor dtype;
//! accepting the resulting node would then guess semantics from a normalized
//! `WeightStore` value.

use crate::loader::const_fold::is_standard_onnx_domain;
use crate::onnx_proto::{self, AttributeProto, GraphProto, NodeProto};
use ny_core::{NyError, Result};
use std::collections::{HashMap, HashSet};

use super::literal_cone::{is_exact_integer_literal_node, LiteralCone, LiteralExemptions};
use super::quantization_preflight::RawDtypeResolver;

const FLOAT: i32 = 1;
const UINT8: i32 = 2;
const INT8: i32 = 3;
const UINT16: i32 = 4;
const INT16: i32 = 5;
const INT32: i32 = 6;
const INT64: i32 = 7;
const STRING: i32 = 8;
const BOOL: i32 = 9;
const FLOAT16: i32 = 10;
const DOUBLE: i32 = 11;
const UINT32: i32 = 12;
const UINT64: i32 = 13;
const COMPLEX128: i32 = 15;
const BFLOAT16: i32 = 16;

const REDUCTION_BASE_TYPES: &[i32] = &[UINT32, UINT64, INT32, INT64, FLOAT16, FLOAT, DOUBLE];
const NUMERIC_TYPES: &[i32] = &[
    UINT8, UINT16, UINT32, UINT64, INT8, INT16, INT32, INT64, FLOAT16, FLOAT, DOUBLE,
];
const NUMERIC_TYPES_IR4: &[i32] = &[
    UINT8, UINT16, UINT32, UINT64, INT8, INT16, INT32, INT64, FLOAT16, FLOAT, DOUBLE, BFLOAT16,
];

pub(super) fn validate_reduction_comparison_schemas(
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
        if !is_standard_onnx_domain(&node.domain) || !is_audited_operator(&node.op_type) {
            continue;
        }
        let opset = standard_opset
            .ok_or_else(|| node_error(node, "has no standard-domain opset import".to_string()))?;
        match node.op_type.as_str() {
            "ReduceSum" | "ReduceMean" | "ReduceMax" | "ReduceMin" => {
                validate_reduction(node, opset, &mut dtypes, &mut shapes)?
            }
            "CumSum" => validate_cumsum(node, opset, &mut dtypes, &mut shapes)?,
            "Equal" | "Less" | "Greater" | "LessOrEqual" | "GreaterOrEqual" => {
                validate_comparison(node, opset, &mut dtypes, &mut shapes, literals, exemptions)?
            }
            "Where" => validate_where(node, opset, &mut dtypes, literals, exemptions)?,
            _ => unreachable!("operator filtered above"),
        }
    }
    Ok(())
}

/// Test-only: validate and immediately discharge any literal-cone deferral.
///
/// A unit test never runs constant folding, so a deferred refusal must re-raise
/// with its original message here.
#[cfg(test)]
pub(super) fn validate_reduction_comparison_schemas_unfolded(
    graph: &GraphProto,
    opset_imports: &HashMap<String, i64>,
) -> Result<()> {
    let literals = LiteralCone::new(graph);
    let mut exemptions = LiteralExemptions::default();
    validate_reduction_comparison_schemas(graph, opset_imports, &literals, &mut exemptions)?;
    exemptions.require_all_folded(&crate::WeightStore::new())
}

fn is_audited_operator(op_type: &str) -> bool {
    matches!(
        op_type,
        "ReduceSum"
            | "ReduceMean"
            | "ReduceMax"
            | "ReduceMin"
            | "CumSum"
            | "Equal"
            | "Less"
            | "Greater"
            | "LessOrEqual"
            | "GreaterOrEqual"
            | "Where"
    )
}

fn validate_reduction(
    node: &NodeProto,
    opset: i64,
    dtypes: &mut RawDtypeResolver<'_>,
    shapes: &mut RawShapeResolver<'_>,
) -> Result<()> {
    require_minimum_opset(node, opset, 1)?;
    let dynamic_axes_opset = if node.op_type == "ReduceSum" { 13 } else { 18 };
    let dynamic_axes = opset >= dynamic_axes_opset;

    if node.output.len() != 1
        || node.output[0].is_empty()
        || node.input.first().is_none_or(String::is_empty)
        || if dynamic_axes {
            !(1..=2).contains(&node.input.len())
        } else {
            node.input.len() != 1
        }
    {
        return Err(node_error(
            node,
            format!(
                "has an invalid opset-{opset} reduction signature: inputs {:?}, outputs {:?}",
                node.input, node.output
            ),
        ));
    }

    for attribute in &node.attribute {
        match attribute.name.as_str() {
            "keepdims" => {
                require_boolean_int_attribute(node, attribute)?;
            }
            "axes" if !dynamic_axes => {
                require_attribute_type(node, attribute, onnx_proto::attribute_type::INTS)?;
            }
            "noop_with_empty_axes" if dynamic_axes => {
                require_boolean_int_attribute(node, attribute)?;
            }
            "axes" => {
                return Err(node_error(
                    node,
                    format!(
                        "uses the obsolete axes attribute at opset {opset}; axes moved to input[1] in opset {dynamic_axes_opset}"
                    ),
                ));
            }
            "noop_with_empty_axes" => {
                return Err(node_error(
                    node,
                    format!(
                        "uses noop_with_empty_axes before its opset-{dynamic_axes_opset} introduction"
                    ),
                ));
            }
            _ => return Err(unknown_attribute(node, attribute, opset)),
        }
    }

    let data_dtype = require_dtype(node, dtypes, &node.input[0], "data")?;
    if !reduction_data_type_allowed(&node.op_type, opset, data_dtype) {
        return Err(node_error(
            node,
            format!("does not allow data dtype {data_dtype} at opset {opset}"),
        ));
    }
    require_ny_float_data(node, data_dtype, "data")?;
    require_output_dtype(node, dtypes, data_dtype)?;

    if let Some(axes) = node.input.get(1).filter(|name| !name.is_empty()) {
        require_specific_dtype(node, dtypes, axes, "axes", &[INT64])?;
        require_rank(node, shapes, axes, "axes", 1)?;
    }
    Ok(())
}

fn reduction_data_type_allowed(op_type: &str, opset: i64, dtype: i32) -> bool {
    if REDUCTION_BASE_TYPES.contains(&dtype) {
        return true;
    }
    (dtype == BFLOAT16 && opset >= 13)
        || (matches!(op_type, "ReduceMax" | "ReduceMin")
            && ((opset >= 12 && matches!(dtype, UINT8 | INT8)) || (opset >= 20 && dtype == BOOL)))
}

fn validate_cumsum(
    node: &NodeProto,
    opset: i64,
    dtypes: &mut RawDtypeResolver<'_>,
    shapes: &mut RawShapeResolver<'_>,
) -> Result<()> {
    require_minimum_opset(node, opset, 11)?;
    require_exact_io(node, 2, 1)?;
    for attribute in &node.attribute {
        match attribute.name.as_str() {
            "exclusive" | "reverse" => {
                require_boolean_int_attribute(node, attribute)?;
            }
            _ => return Err(unknown_attribute(node, attribute, opset)),
        }
    }

    let data_dtype = require_dtype(node, dtypes, &node.input[0], "data")?;
    let allowed = if opset >= 14 {
        REDUCTION_BASE_TYPES
    } else {
        &[UINT32, UINT64, INT32, INT64, FLOAT, DOUBLE]
    };
    let dtype_allowed = allowed.contains(&data_dtype) || (opset >= 14 && data_dtype == BFLOAT16);
    if !dtype_allowed {
        return Err(node_error(
            node,
            format!("does not allow data dtype {data_dtype} at opset {opset}"),
        ));
    }
    require_ny_float_data(node, data_dtype, "data")?;
    require_output_dtype(node, dtypes, data_dtype)?;
    require_specific_dtype(node, dtypes, &node.input[1], "axis", &[INT32, INT64])?;
    require_rank(node, shapes, &node.input[1], "axis", 0)
}

fn validate_comparison(
    node: &NodeProto,
    opset: i64,
    dtypes: &mut RawDtypeResolver<'_>,
    shapes: &mut RawShapeResolver<'_>,
    literals: &LiteralCone,
    exemptions: &mut LiteralExemptions,
) -> Result<()> {
    let minimum = if matches!(node.op_type.as_str(), "LessOrEqual" | "GreaterOrEqual") {
        12
    } else {
        1
    };
    require_minimum_opset(node, opset, minimum)?;
    require_exact_io(node, 2, 1)?;

    let legacy_broadcast =
        opset < 7 && matches!(node.op_type.as_str(), "Equal" | "Less" | "Greater");
    if legacy_broadcast {
        validate_legacy_comparison_attributes(node, shapes)?;
    } else if let Some(attribute) = node.attribute.first() {
        return Err(unknown_attribute(node, attribute, opset));
    }

    let lhs_dtype = require_dtype(node, dtypes, &node.input[0], "A")?;
    let rhs_dtype = require_dtype(node, dtypes, &node.input[1], "B")?;
    if lhs_dtype != rhs_dtype {
        return Err(node_error(
            node,
            format!(
                "requires A and B to have one tensor type, got dtypes {lhs_dtype} and {rhs_dtype}"
            ),
        ));
    }
    if !comparison_data_type_allowed(&node.op_type, opset, lhs_dtype) {
        return Err(node_error(
            node,
            format!("does not allow input dtype {lhs_dtype} at opset {opset}"),
        ));
    }
    if let Err(refusal) = require_ny_float_data(node, lhs_dtype, "comparison operands") {
        if !is_exact_integer_literal_node(node, literals, dtypes)? {
            return Err(refusal);
        }
        exemptions.record(node, &refusal);
    }
    require_output_dtype(node, dtypes, BOOL)
}

fn validate_legacy_comparison_attributes(
    node: &NodeProto,
    shapes: &mut RawShapeResolver<'_>,
) -> Result<()> {
    let mut broadcast = 0;
    let mut axis = None;
    for attribute in &node.attribute {
        match attribute.name.as_str() {
            "broadcast" => broadcast = require_boolean_int_attribute(node, attribute)?,
            "axis" => {
                require_attribute_type(node, attribute, onnx_proto::attribute_type::INT)?;
                axis = Some(attribute.i_value());
            }
            _ => return Err(unknown_attribute(node, attribute, 1)),
        }
    }

    let lhs = require_shape(node, shapes, &node.input[0], "A")?;
    let rhs = require_shape(node, shapes, &node.input[1], "B")?;
    if broadcast == 0 {
        if !lhs.provably_equal(&rhs) {
            return Err(NyError::UnsupportedOp(format!(
                "standard ONNX {} node '{}' uses legacy broadcast=0 with shapes {:?} and {:?}; ny's NumPy-broadcast comparison is equivalent only when the shapes are provably equal",
                node.op_type, node.name, lhs, rhs
            )));
        }
        return Ok(());
    }

    let trailing_axis = lhs.rank().checked_sub(rhs.rank()).ok_or_else(|| {
        node_error(
            node,
            format!(
                "cannot broadcast rank-{} B into rank-{} A",
                rhs.rank(),
                lhs.rank()
            ),
        )
    })?;
    let authored_axis = match axis {
        Some(value) => usize::try_from(value).ok(),
        None => Some(trailing_axis),
    };
    if authored_axis != Some(trailing_axis) {
        return Err(NyError::UnsupportedOp(format!(
            "standard ONNX {} node '{}' uses legacy axis {:?}; ny represents only the trailing placement {} that is exactly NumPy-broadcast equivalent",
            node.op_type, node.name, axis, trailing_axis
        )));
    }
    if !lhs.numpy_broadcasts_rhs(&rhs) {
        return Err(node_error(
            node,
            format!(
                "has incompatible legacy broadcast shapes {:?} and {:?}",
                lhs, rhs
            ),
        ));
    }
    Ok(())
}

fn comparison_data_type_allowed(op_type: &str, opset: i64, dtype: i32) -> bool {
    match op_type {
        "Equal" if opset < 11 => matches!(dtype, BOOL | INT32 | INT64),
        "Equal" if opset < 13 => dtype == BOOL || NUMERIC_TYPES.contains(&dtype),
        "Equal" if opset < 19 => dtype == BOOL || NUMERIC_TYPES_IR4.contains(&dtype),
        "Equal" => dtype == BOOL || dtype == STRING || NUMERIC_TYPES_IR4.contains(&dtype),
        "Less" | "Greater" if opset < 9 => matches!(dtype, FLOAT16 | FLOAT | DOUBLE),
        "Less" | "Greater" if opset < 13 => NUMERIC_TYPES.contains(&dtype),
        "Less" | "Greater" => NUMERIC_TYPES_IR4.contains(&dtype),
        "LessOrEqual" | "GreaterOrEqual" if opset < 16 => NUMERIC_TYPES.contains(&dtype),
        "LessOrEqual" | "GreaterOrEqual" => NUMERIC_TYPES_IR4.contains(&dtype),
        _ => false,
    }
}

fn validate_where(
    node: &NodeProto,
    opset: i64,
    dtypes: &mut RawDtypeResolver<'_>,
    literals: &LiteralCone,
    exemptions: &mut LiteralExemptions,
) -> Result<()> {
    require_minimum_opset(node, opset, 9)?;
    require_exact_io(node, 3, 1)?;
    if let Some(attribute) = node.attribute.first() {
        return Err(unknown_attribute(node, attribute, opset));
    }

    require_specific_dtype(node, dtypes, &node.input[0], "condition", &[BOOL])?;
    let true_dtype = require_dtype(node, dtypes, &node.input[1], "X")?;
    let false_dtype = require_dtype(node, dtypes, &node.input[2], "Y")?;
    if true_dtype != false_dtype {
        return Err(node_error(
            node,
            format!(
                "requires X and Y to have one tensor type, got dtypes {true_dtype} and {false_dtype}"
            ),
        ));
    }
    let allowed = (1..=COMPLEX128).contains(&true_dtype) || (opset >= 16 && true_dtype == BFLOAT16);
    if !allowed {
        return Err(node_error(
            node,
            format!("does not allow selected-value dtype {true_dtype} at opset {opset}"),
        ));
    }
    if let Err(refusal) = require_ny_float_data(node, true_dtype, "selected values") {
        if !is_exact_integer_literal_node(node, literals, dtypes)? {
            return Err(refusal);
        }
        exemptions.record(node, &refusal);
    }
    require_output_dtype(node, dtypes, true_dtype)
}

fn require_ny_float_data(node: &NodeProto, dtype: i32, role: &str) -> Result<()> {
    if dtype != FLOAT {
        return Err(NyError::UnsupportedOp(format!(
            "standard ONNX {} node '{}' has {role} dtype {dtype}; ny represents only ONNX FLOAT (float32) data for this operator",
            node.op_type, node.name
        )));
    }
    Ok(())
}

fn require_minimum_opset(node: &NodeProto, opset: i64, minimum: i64) -> Result<()> {
    if opset < minimum {
        return Err(node_error(
            node,
            format!("requires opset {minimum} or newer, got {opset}"),
        ));
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
                "requires exactly {inputs} non-empty input(s) and {outputs} non-empty output(s); got inputs {:?} and outputs {:?}",
                node.input, node.output
            ),
        ));
    }
    Ok(())
}

fn require_boolean_int_attribute(node: &NodeProto, attribute: &AttributeProto) -> Result<i64> {
    require_attribute_type(node, attribute, onnx_proto::attribute_type::INT)?;
    let value = attribute.i_value();
    if !matches!(value, 0 | 1) {
        return Err(node_error(
            node,
            format!(
                "attribute '{}' must be encoded as INT 0 or 1, got {value}",
                attribute.name
            ),
        ));
    }
    Ok(value)
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
                "attribute '{}' must have ONNX attribute type {expected}, got {}",
                attribute.name, attribute.r#type
            ),
        ));
    }
    Ok(())
}

fn require_dtype(
    node: &NodeProto,
    dtypes: &mut RawDtypeResolver<'_>,
    value: &str,
    role: &str,
) -> Result<i32> {
    dtypes.resolve(value)?.ok_or_else(|| {
        node_error(
            node,
            format!("cannot authenticate the ONNX dtype of {role} value '{value}'"),
        )
    })
}

fn require_specific_dtype(
    node: &NodeProto,
    dtypes: &mut RawDtypeResolver<'_>,
    value: &str,
    role: &str,
    allowed: &[i32],
) -> Result<()> {
    let dtype = require_dtype(node, dtypes, value, role)?;
    if !allowed.contains(&dtype) {
        return Err(node_error(
            node,
            format!(
                "requires {role} value '{value}' to have dtype {:?}, got {dtype}",
                allowed
            ),
        ));
    }
    Ok(())
}

fn require_output_dtype(
    node: &NodeProto,
    dtypes: &mut RawDtypeResolver<'_>,
    expected: i32,
) -> Result<()> {
    let output = &node.output[0];
    let dtype = require_dtype(node, dtypes, output, "output")?;
    if dtype != expected {
        return Err(node_error(
            node,
            format!(
                "requires output '{}' to have dtype {expected}, got {dtype}",
                output
            ),
        ));
    }
    Ok(())
}

fn require_rank(
    node: &NodeProto,
    shapes: &mut RawShapeResolver<'_>,
    value: &str,
    role: &str,
    expected: usize,
) -> Result<()> {
    let shape = require_shape(node, shapes, value, role)?;
    if shape.rank() != expected {
        return Err(node_error(
            node,
            format!(
                "requires {role} value '{value}' to have rank {expected}, got shape {:?}",
                shape
            ),
        ));
    }
    Ok(())
}

fn require_shape(
    node: &NodeProto,
    shapes: &mut RawShapeResolver<'_>,
    value: &str,
    role: &str,
) -> Result<RawShape> {
    shapes.resolve(value)?.ok_or_else(|| {
        node_error(
            node,
            format!("cannot authenticate the ONNX shape of {role} value '{value}'"),
        )
    })
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
enum RawDimension {
    Known(i64),
    Symbol(String),
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RawShape(Vec<RawDimension>);

impl RawShape {
    fn rank(&self) -> usize {
        self.0.len()
    }

    fn provably_equal(&self, other: &Self) -> bool {
        self.0.len() == other.0.len()
            && self
                .0
                .iter()
                .zip(&other.0)
                .all(|(left, right)| match (left, right) {
                    (RawDimension::Known(left), RawDimension::Known(right)) => left == right,
                    (RawDimension::Symbol(left), RawDimension::Symbol(right)) => left == right,
                    _ => false,
                })
    }

    fn numpy_broadcasts_rhs(&self, rhs: &Self) -> bool {
        if rhs.rank() > self.rank() {
            return false;
        }
        self.0[self.rank() - rhs.rank()..]
            .iter()
            .zip(&rhs.0)
            .all(|(left, right)| dimensions_broadcast(left, right))
    }
}

fn dimensions_broadcast(left: &RawDimension, right: &RawDimension) -> bool {
    // Legacy unidirectional broadcasting expands B to A; unlike NumPy's
    // bidirectional rule, an A dimension of one cannot grow to match B.
    matches!(right, RawDimension::Known(1))
        || match (left, right) {
            (RawDimension::Known(left), RawDimension::Known(right)) => left == right,
            (RawDimension::Symbol(left), RawDimension::Symbol(right)) => left == right,
            _ => false,
        }
}

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
                if let Some(shape) = tensor_shape(initializer) {
                    authored
                        .entry(initializer.name.as_str())
                        .or_default()
                        .push(shape);
                }
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
            for output in node.output.iter().filter(|name| !name.is_empty()) {
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
        if !is_standard_onnx_domain(&node.domain)
            || node.output.len() != 1
            || node.output[0] != output
        {
            return Ok(None);
        }
        match node.op_type.as_str() {
            "Constant" => Ok(constant_shape(&node)),
            "Identity" | "Cast" if node.input.len() == 1 && !node.input[0].is_empty() => {
                self.resolve_inner(&node.input[0], active)
            }
            _ => Ok(None),
        }
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
            (RawDimension::Known(left), RawDimension::Known(right)) if left == right => {
                Some(RawDimension::Known(left))
            }
            (RawDimension::Symbol(left), RawDimension::Symbol(right)) if left == right => {
                Some(RawDimension::Symbol(left))
            }
            (RawDimension::Unknown, other) | (other, RawDimension::Unknown) => Some(other),
            (RawDimension::Known(value), RawDimension::Symbol(_))
            | (RawDimension::Symbol(_), RawDimension::Known(value)) => {
                Some(RawDimension::Known(value))
            }
            (RawDimension::Symbol(_), RawDimension::Symbol(_)) => Some(RawDimension::Unknown),
            _ => None,
        })
        .collect::<Option<Vec<_>>>()
        .map(RawShape)
}

fn tensor_shape(tensor: &onnx_proto::TensorProto) -> Option<RawShape> {
    tensor
        .dims
        .iter()
        .copied()
        .map(|dimension| (dimension >= 0).then_some(RawDimension::Known(dimension)))
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
    dimensions
        .iter()
        .map(|dimension| match dimension.value.as_ref() {
            Some(Value::DimValue(value)) if *value >= 0 => Some(RawDimension::Known(*value)),
            Some(Value::DimParam(symbol)) => Some(RawDimension::Symbol(symbol.clone())),
            Some(Value::DimValue(_)) => None,
            None => Some(RawDimension::Unknown),
        })
        .collect::<Option<Vec<_>>>()
        .map(RawShape)
}

fn constant_shape(node: &NodeProto) -> Option<RawShape> {
    if !node.input.is_empty() {
        return None;
    }
    let mut shapes = Vec::new();
    for attribute in &node.attribute {
        let shape = match (attribute.name.as_str(), attribute.r#type) {
            ("value", onnx_proto::attribute_type::TENSOR) => {
                attribute.t.as_ref().and_then(tensor_shape)
            }
            ("value_float", onnx_proto::attribute_type::FLOAT)
            | ("value_int", onnx_proto::attribute_type::INT) => Some(RawShape(Vec::new())),
            ("value_floats", onnx_proto::attribute_type::FLOATS) => {
                i64::try_from(attribute.floats.len())
                    .ok()
                    .map(|length| RawShape(vec![RawDimension::Known(length)]))
            }
            ("value_ints", onnx_proto::attribute_type::INTS) => i64::try_from(attribute.ints.len())
                .ok()
                .map(|length| RawShape(vec![RawDimension::Known(length)])),
            _ => None,
        };
        if let Some(shape) = shape {
            shapes.push(shape);
        }
    }
    match shapes.as_slice() {
        [shape] => Some(shape.clone()),
        _ => None,
    }
}
