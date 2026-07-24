// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

mod batch_norm;
#[cfg(feature = "ort")]
mod batch_norm_ort_prop;
mod batch_norm_tail;
mod instance_norm;
mod layer_norm;
mod merge_linear;

use crate::onnx_proto::{attribute_type, AttributeProto, NodeProto, TensorProto};
use ndarray::{ArrayD, IxDyn};

fn make_node(op_type: &str, inputs: &[&str], outputs: &[&str]) -> NodeProto {
    NodeProto {
        input: inputs.iter().map(|s| (*s).to_string()).collect(),
        output: outputs.iter().map(|s| (*s).to_string()).collect(),
        name: String::new(),
        op_type: op_type.to_string(),
        domain: String::new(),
        attribute: Vec::new(),
    }
}

fn make_axes_attr(axes: &[i64]) -> AttributeProto {
    AttributeProto {
        name: "axes".to_string(),
        f: 0.0,
        i: 0,
        s: Vec::new(),
        t: None,
        r#type: 7,
        floats: Vec::new(),
        ints: axes.to_vec(),
    }
}

fn make_const_scalar(output: &str, value: f32) -> NodeProto {
    let tensor = TensorProto {
        dims: Vec::new(),
        data_type: 1,
        name: output.to_string(),
        raw_data: Vec::new(),
        float_data: vec![value],
        ..Default::default()
    };
    let attr = AttributeProto {
        name: "value".to_string(),
        f: 0.0,
        i: 0,
        s: Vec::new(),
        t: Some(tensor),
        r#type: 4,
        floats: Vec::new(),
        ints: Vec::new(),
    };
    let mut node = make_node("Constant", &[], &[output]);
    node.attribute.push(attr);
    node
}

fn make_float_attr(name: &str, value: f32) -> AttributeProto {
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

fn make_int_attr(name: &str, value: i64) -> AttributeProto {
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

fn make_weight(shape: &[usize], data: &[f32]) -> ArrayD<f32> {
    ArrayD::from_shape_vec(IxDyn(shape), data.to_vec()).expect("valid test weight shape")
}
