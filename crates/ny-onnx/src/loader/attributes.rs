// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::onnx_proto;
use crate::onnx_proto::attribute_type;
use crate::AttributeValue;
use std::collections::HashMap;

pub(super) fn node_attr_ints(node: &onnx_proto::NodeProto, name: &str) -> Option<Vec<i64>> {
    for attr in &node.attribute {
        if attr.name != name {
            continue;
        }
        if attr.r#type == attribute_type::INTS {
            return Some(attr.ints.clone());
        }
        if attr.r#type == attribute_type::INT {
            return Some(vec![attr.i]);
        }
    }
    None
}

pub(super) fn node_attr_int(node: &onnx_proto::NodeProto, name: &str) -> Option<i64> {
    for attr in &node.attribute {
        if attr.name != name {
            continue;
        }
        if attr.r#type == attribute_type::INT {
            return Some(attr.i);
        }
        if attr.r#type == attribute_type::INTS && attr.ints.len() == 1 {
            return Some(attr.ints[0]);
        }
    }
    None
}

pub(super) fn parse_node_attributes(
    node: &onnx_proto::NodeProto,
) -> HashMap<String, AttributeValue> {
    let mut out = HashMap::new();

    for attr in &node.attribute {
        let value = match attr.r#type {
            attribute_type::FLOAT => Some(AttributeValue::Float(attr.f)),
            attribute_type::INT => Some(AttributeValue::Int(attr.i)),
            attribute_type::STRING => Some(AttributeValue::String(
                String::from_utf8_lossy(&attr.s).to_string(),
            )),
            attribute_type::FLOATS => Some(AttributeValue::Floats(attr.floats.clone())),
            attribute_type::INTS => Some(AttributeValue::Ints(attr.ints.clone())),
            _ => None,
        };
        if let Some(value) = value {
            out.insert(attr.name.clone(), value);
        }
    }

    out
}
