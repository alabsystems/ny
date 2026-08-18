// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::prepare::lower_reduce_l2_nodes;
use crate::onnx_proto::{AttributeProto, NodeProto};

fn make_node(op: &str, inputs: &[&str], outputs: &[&str]) -> NodeProto {
    NodeProto {
        op_type: op.to_string(),
        input: inputs.iter().map(|value| value.to_string()).collect(),
        output: outputs.iter().map(|value| value.to_string()).collect(),
        name: String::new(),
        domain: String::new(),
        attribute: Vec::new(),
    }
}

fn make_int_attr(name: &str, value: i64) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        i: Some(value),
        ..Default::default()
    }
}

#[test]
fn test_lower_reduce_l2_nodes_rewrites_to_supported_primitives() {
    let mut reduce_l2 = make_node("ReduceL2", &["x"], &["norm"]);
    reduce_l2.name = "reduce_l2".to_string();
    reduce_l2.attribute.push(make_int_attr("keepdims", 0));
    reduce_l2.attribute.push(AttributeProto {
        name: "axes".to_string(),
        ints: vec![-1],
        ..Default::default()
    });
    let mut nodes = vec![reduce_l2];

    lower_reduce_l2_nodes(&mut nodes);

    assert_eq!(nodes.len(), 4);

    assert_eq!(nodes[0].op_type, "Constant");
    assert!(nodes[0].input.is_empty());
    assert_eq!(
        nodes[0].output,
        vec!["reduce_l2__reduce_l2_exponent".to_string()]
    );
    assert_eq!(nodes[0].attribute.len(), 1);
    assert_eq!(nodes[0].attribute[0].name, "value_float");
    assert_eq!(nodes[0].attribute[0].f, Some(2.0));

    assert_eq!(nodes[1].op_type, "Pow");
    assert_eq!(
        nodes[1].input,
        vec!["x".to_string(), "reduce_l2__reduce_l2_exponent".to_string()]
    );
    assert_eq!(
        nodes[1].output,
        vec!["reduce_l2__reduce_l2_square".to_string()]
    );

    assert_eq!(nodes[2].op_type, "ReduceSum");
    assert_eq!(
        nodes[2].input,
        vec!["reduce_l2__reduce_l2_square".to_string()]
    );
    assert_eq!(
        nodes[2].output,
        vec!["reduce_l2__reduce_l2_sum".to_string()]
    );
    assert_eq!(nodes[2].attribute.len(), 2);

    assert_eq!(nodes[3].op_type, "Sqrt");
    assert_eq!(nodes[3].input, vec!["reduce_l2__reduce_l2_sum".to_string()]);
    assert_eq!(nodes[3].output, vec!["norm".to_string()]);
}
