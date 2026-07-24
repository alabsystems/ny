// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::onnx_proto::{
    attribute_type, tensor_shape_proto, AttributeProto, NodeProto, TensorProto, TensorShapeProto,
    TensorTypeProto, TypeProto, ValueInfoProto,
};
use crate::WeightStore;

pub(super) fn attr_tensor(name: &str, tensor: TensorProto) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        f: 0.0,
        i: 0,
        s: Vec::new(),
        t: Some(tensor),
        r#type: attribute_type::TENSOR,
        floats: Vec::new(),
        ints: Vec::new(),
    }
}

pub(super) fn attr_ints(name: &str, values: &[i64]) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        f: 0.0,
        i: 0,
        s: Vec::new(),
        t: None,
        r#type: attribute_type::INTS,
        floats: Vec::new(),
        ints: values.to_vec(),
    }
}

pub(super) fn attr_float(name: &str, value: f32) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        f: value,
        i: 0,
        s: Vec::new(),
        t: None,
        r#type: attribute_type::FLOAT,
        floats: Vec::new(),
        ints: Vec::new(),
    }
}

pub(super) fn attr_int(name: &str, value: i64) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        f: 0.0,
        i: value,
        s: Vec::new(),
        t: None,
        r#type: attribute_type::INT,
        floats: Vec::new(),
        ints: Vec::new(),
    }
}

pub(super) fn tensor_f32(name: &str, shape: &[i64], data: &[f32]) -> TensorProto {
    let elements = shape.iter().product::<i64>() as usize;
    assert_eq!(elements, data.len());
    TensorProto {
        dims: shape.to_vec(),
        data_type: 1,
        name: name.to_string(),
        raw_data: Vec::new(),
        float_data: data.to_vec(),
        ..Default::default()
    }
}

pub(super) fn node(
    name: &str,
    op_type: &str,
    inputs: &[&str],
    outputs: &[&str],
    attrs: Vec<AttributeProto>,
) -> NodeProto {
    NodeProto {
        input: inputs.iter().map(|s| s.to_string()).collect(),
        output: outputs.iter().map(|s| s.to_string()).collect(),
        op_type: op_type.to_string(),
        name: name.to_string(),
        attribute: attrs,
        ..Default::default()
    }
}

pub(super) fn tensor_value_info(name: &str, shape: &[i64]) -> ValueInfoProto {
    let dims = shape
        .iter()
        .map(|dim| tensor_shape_proto::Dimension {
            value: Some(tensor_shape_proto::dimension::Value::DimValue(*dim)),
        })
        .collect();
    ValueInfoProto {
        name: name.to_string(),
        r#type: Some(TypeProto {
            tensor_type: Some(TensorTypeProto {
                elem_type: 1,
                shape: Some(TensorShapeProto { dim: dims }),
            }),
        }),
    }
}

pub(super) fn assert_folded_tensor(
    weights: &WeightStore,
    name: &str,
    shape: &[usize],
    expected: &[f32],
) {
    let tensor = weights
        .get(name)
        .unwrap_or_else(|| panic!("missing folded tensor {name}"));
    assert_eq!(tensor.shape(), shape, "{name} shape mismatch");
    assert_eq!(tensor.len(), expected.len(), "{name} length mismatch");
    for (idx, (got, exp)) in tensor.iter().zip(expected.iter()).enumerate() {
        assert!(
            (*got - *exp).abs() < 1.0e-6,
            "{name}[{idx}] mismatch: got {got}, expected {exp}"
        );
    }
}

pub(super) fn fold(graph: &crate::onnx_proto::GraphProto, weights: &mut WeightStore) {
    super::super::fold_constant_nodes(graph, weights, &mut std::collections::HashMap::new())
}
