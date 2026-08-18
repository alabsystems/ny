// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::onnx_proto;
use crate::onnx_proto::attribute_type;
use crate::AttributeValue;
use ny_core::{NyError, Result};
use std::collections::HashMap;

/// Reject protobuf attributes that populate storage fields other than the one
/// selected by `type`.
///
/// AttributeProto is not encoded as a protobuf `oneof`: malformed producers
/// can retain several payload fields simultaneously. Selecting one field and
/// then folding the node would erase that ambiguity. Proto3 scalar presence is
/// represented explicitly so an inactive encoded zero is not erased. ONNX
/// permits the selected scalar field to be absent when its semantic value is
/// the proto3 default zero.
pub(super) fn validate_attribute_storage(
    node: &onnx_proto::NodeProto,
    attribute: &onnx_proto::AttributeProto,
) -> Result<()> {
    let f_empty = attribute.f.is_none();
    let i_empty = attribute.i.is_none();
    let s_empty = attribute.s.is_none();
    let t_empty = attribute.t.is_none();
    let g_empty = attribute.g.is_none();
    let sparse_tensor_empty = attribute.sparse_tensor.is_none();
    let tp_empty = attribute.tp.is_none();
    let floats_empty = attribute.floats.is_empty();
    let ints_empty = attribute.ints.is_empty();
    let strings_empty = attribute.strings.is_empty();
    let tensors_empty = attribute.tensors.is_empty();
    let graphs_empty = attribute.graphs.is_empty();
    let sparse_tensors_empty = attribute.sparse_tensors.is_empty();
    let type_protos_empty = attribute.type_protos.is_empty();
    // Attribute references are valid only inside FunctionProto bodies, which
    // this loader does not ingest. In a ModelProto graph they must never hide
    // or replace a local value.
    let reference_empty = attribute.ref_attr_name.is_empty();
    let singular_messages_empty = g_empty && sparse_tensor_empty && tp_empty;
    let repeated_messages_empty =
        strings_empty && tensors_empty && graphs_empty && sparse_tensors_empty && type_protos_empty;
    let canonical = match attribute.r#type {
        attribute_type::FLOAT => {
            reference_empty
                && i_empty
                && s_empty
                && t_empty
                && singular_messages_empty
                && floats_empty
                && ints_empty
                && repeated_messages_empty
        }
        attribute_type::INT => {
            reference_empty
                && f_empty
                && s_empty
                && t_empty
                && singular_messages_empty
                && floats_empty
                && ints_empty
                && repeated_messages_empty
        }
        attribute_type::STRING => {
            reference_empty
                && f_empty
                && i_empty
                && t_empty
                && singular_messages_empty
                && floats_empty
                && ints_empty
                && repeated_messages_empty
        }
        attribute_type::TENSOR => {
            reference_empty
                && f_empty
                && i_empty
                && s_empty
                && attribute.t.is_some()
                && singular_messages_empty
                && floats_empty
                && ints_empty
                && repeated_messages_empty
        }
        attribute_type::GRAPH => {
            reference_empty
                && f_empty
                && i_empty
                && s_empty
                && t_empty
                && attribute.g.is_some()
                && sparse_tensor_empty
                && tp_empty
                && floats_empty
                && ints_empty
                && repeated_messages_empty
        }
        attribute_type::SPARSE_TENSOR => {
            reference_empty
                && f_empty
                && i_empty
                && s_empty
                && t_empty
                && g_empty
                && attribute.sparse_tensor.is_some()
                && tp_empty
                && floats_empty
                && ints_empty
                && repeated_messages_empty
        }
        attribute_type::TYPE_PROTO => {
            reference_empty
                && f_empty
                && i_empty
                && s_empty
                && t_empty
                && g_empty
                && sparse_tensor_empty
                && attribute.tp.is_some()
                && floats_empty
                && ints_empty
                && repeated_messages_empty
        }
        attribute_type::FLOATS => {
            reference_empty
                && f_empty
                && i_empty
                && s_empty
                && t_empty
                && singular_messages_empty
                && ints_empty
                && repeated_messages_empty
        }
        attribute_type::INTS => {
            reference_empty
                && f_empty
                && i_empty
                && s_empty
                && t_empty
                && singular_messages_empty
                && floats_empty
                && repeated_messages_empty
        }
        attribute_type::STRINGS => {
            reference_empty
                && f_empty
                && i_empty
                && s_empty
                && t_empty
                && singular_messages_empty
                && floats_empty
                && ints_empty
                && tensors_empty
                && graphs_empty
                && sparse_tensors_empty
                && type_protos_empty
        }
        attribute_type::TENSORS => {
            reference_empty
                && f_empty
                && i_empty
                && s_empty
                && t_empty
                && singular_messages_empty
                && floats_empty
                && ints_empty
                && strings_empty
                && graphs_empty
                && sparse_tensors_empty
                && type_protos_empty
        }
        attribute_type::GRAPHS => {
            reference_empty
                && f_empty
                && i_empty
                && s_empty
                && t_empty
                && singular_messages_empty
                && floats_empty
                && ints_empty
                && strings_empty
                && tensors_empty
                && sparse_tensors_empty
                && type_protos_empty
        }
        attribute_type::SPARSE_TENSORS => {
            reference_empty
                && f_empty
                && i_empty
                && s_empty
                && t_empty
                && singular_messages_empty
                && floats_empty
                && ints_empty
                && strings_empty
                && tensors_empty
                && graphs_empty
                && type_protos_empty
        }
        attribute_type::TYPE_PROTOS => {
            reference_empty
                && f_empty
                && i_empty
                && s_empty
                && t_empty
                && singular_messages_empty
                && floats_empty
                && ints_empty
                && strings_empty
                && tensors_empty
                && graphs_empty
                && sparse_tensors_empty
        }
        _ => false,
    };
    if canonical {
        Ok(())
    } else {
        Err(NyError::ModelLoad(format!(
            "ONNX {} node '{}' in domain '{}' attribute '{}' has ambiguous or unsupported raw storage for declared type {}",
            node.op_type, node.name, node.domain, attribute.name, attribute.r#type
        )))
    }
}

pub(super) fn parse_node_attributes(
    node: &onnx_proto::NodeProto,
) -> HashMap<String, AttributeValue> {
    let mut out = HashMap::new();

    for attr in &node.attribute {
        let value = match attr.r#type {
            attribute_type::FLOAT => Some(AttributeValue::Float(attr.f_value())),
            attribute_type::INT => Some(AttributeValue::Int(attr.i_value())),
            attribute_type::STRING => Some(AttributeValue::String(
                String::from_utf8_lossy(attr.s_value()).to_string(),
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

#[cfg(test)]
mod tests {
    use super::validate_attribute_storage;
    use crate::onnx_proto::{self, attribute_type};
    use prost::Message;

    fn node() -> onnx_proto::NodeProto {
        onnx_proto::NodeProto {
            name: "quantize".to_string(),
            op_type: "QuantizeLinear".to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn declared_payload_must_be_the_only_populated_storage_field() {
        let clean = onnx_proto::AttributeProto {
            name: "axis".to_string(),
            i: Some(0),
            r#type: attribute_type::INT,
            ..Default::default()
        };
        validate_attribute_storage(&node(), &clean).expect("canonical INT attribute");

        for malformed in [
            onnx_proto::AttributeProto {
                floats: vec![1.0],
                ..clean.clone()
            },
            onnx_proto::AttributeProto {
                f: Some(-0.0),
                ..clean.clone()
            },
            onnx_proto::AttributeProto {
                ints: vec![7],
                ..clean
            },
        ] {
            assert!(validate_attribute_storage(&node(), &malformed).is_err());
        }

        let malformed_constant = onnx_proto::AttributeProto {
            name: "value_float".to_string(),
            f: Some(1.0),
            ints: vec![2],
            r#type: attribute_type::FLOAT,
            ..Default::default()
        };
        assert!(validate_attribute_storage(&node(), &malformed_constant).is_err());
    }

    #[test]
    fn serialized_inactive_graph_payload_is_retained_and_rejected() {
        // AttributeProto { i: 7, type: INT, g: GraphProto {} }. This is
        // intentionally authored as bytes: a decoder that omits official tag
        // 6 would silently discard the conflicting graph before validation.
        let wire = [0x18, 0x07, 0xa0, 0x01, 0x02, 0x32, 0x00];
        let attribute = onnx_proto::AttributeProto::decode(wire.as_slice())
            .expect("decode malformed AttributeProto wire");

        assert_eq!(attribute.i, Some(7));
        assert!(attribute.g.is_some(), "official graph tag must be retained");
        assert!(validate_attribute_storage(&node(), &attribute).is_err());
    }

    #[test]
    fn all_official_payload_and_reference_fields_participate_in_union_validation() {
        let clean = onnx_proto::AttributeProto {
            name: "axis".to_string(),
            i: Some(1),
            r#type: attribute_type::INT,
            ..Default::default()
        };

        let malformed = [
            onnx_proto::AttributeProto {
                ref_attr_name: "parent_axis".to_string(),
                ..clean.clone()
            },
            onnx_proto::AttributeProto {
                g: Some(Default::default()),
                ..clean.clone()
            },
            onnx_proto::AttributeProto {
                sparse_tensor: Some(Default::default()),
                ..clean.clone()
            },
            onnx_proto::AttributeProto {
                tp: Some(Default::default()),
                ..clean.clone()
            },
            onnx_proto::AttributeProto {
                s: Some(Vec::new()),
                ..clean.clone()
            },
            onnx_proto::AttributeProto {
                strings: vec![b"hidden".to_vec()],
                ..clean.clone()
            },
            onnx_proto::AttributeProto {
                tensors: vec![Default::default()],
                ..clean.clone()
            },
            onnx_proto::AttributeProto {
                graphs: vec![Default::default()],
                ..clean.clone()
            },
            onnx_proto::AttributeProto {
                sparse_tensors: vec![Default::default()],
                ..clean.clone()
            },
            onnx_proto::AttributeProto {
                type_protos: vec![Default::default()],
                ..clean.clone()
            },
        ];

        for attribute in malformed {
            assert!(validate_attribute_storage(&node(), &attribute).is_err());
        }

        let documented = onnx_proto::AttributeProto {
            doc_string: "metadata is not a payload".to_string(),
            ..clean
        };
        validate_attribute_storage(&node(), &documented)
            .expect("documentation does not make payload storage ambiguous");
    }

    #[test]
    fn newly_retained_payload_types_validate_when_they_are_selected() {
        for attribute in [
            onnx_proto::AttributeProto {
                name: "body".to_string(),
                r#type: attribute_type::GRAPH,
                g: Some(Default::default()),
                ..Default::default()
            },
            onnx_proto::AttributeProto {
                name: "labels".to_string(),
                r#type: attribute_type::STRINGS,
                strings: vec![b"a".to_vec(), b"b".to_vec()],
                ..Default::default()
            },
            onnx_proto::AttributeProto {
                name: "sparse".to_string(),
                r#type: attribute_type::SPARSE_TENSOR,
                sparse_tensor: Some(Default::default()),
                ..Default::default()
            },
            onnx_proto::AttributeProto {
                name: "types".to_string(),
                r#type: attribute_type::TYPE_PROTOS,
                type_protos: vec![Default::default()],
                ..Default::default()
            },
        ] {
            validate_attribute_storage(&node(), &attribute)
                .expect("selected payload must remain canonical");
        }
    }

    #[test]
    fn selected_scalar_payload_may_omit_its_default_zero_wire_tag() {
        for (declared_type, missing_wire) in [
            (
                attribute_type::INT,
                // type: INT, with no tag 3 payload.
                vec![0xa0, 0x01, 0x02],
            ),
            (
                attribute_type::FLOAT,
                // type: FLOAT, with no tag 2 payload.
                vec![0xa0, 0x01, 0x01],
            ),
        ] {
            let attribute = onnx_proto::AttributeProto::decode(missing_wire.as_slice())
                .expect("decode missing-scalar AttributeProto wire");
            assert_eq!(attribute.r#type, declared_type);
            validate_attribute_storage(&node(), &attribute)
                .expect("ONNX permits an omitted selected scalar at its default zero");
            match declared_type {
                attribute_type::INT => assert_eq!(attribute.i_value(), 0),
                attribute_type::FLOAT => assert_eq!(attribute.f_value(), 0.0),
                _ => unreachable!(),
            }
        }

        // Explicit encoded zeros remain valid scalar payloads.
        for wire in [
            vec![0x18, 0x00, 0xa0, 0x01, 0x02],
            vec![0x15, 0x00, 0x00, 0x00, 0x00, 0xa0, 0x01, 0x01],
        ] {
            let attribute = onnx_proto::AttributeProto::decode(wire.as_slice())
                .expect("decode explicit-zero AttributeProto wire");
            validate_attribute_storage(&node(), &attribute)
                .expect("an explicit zero is a present scalar payload");
        }
    }

    #[test]
    fn empty_string_wire_presence_is_retained_for_union_validation() {
        // type: INT plus explicitly present empty `s` (tag 4). The empty byte
        // string still counts as a conflicting value field.
        let inactive_wire = [0xa0, 0x01, 0x02, 0x22, 0x00];
        let inactive = onnx_proto::AttributeProto::decode(inactive_wire.as_slice())
            .expect("decode inactive empty STRING payload");
        assert_eq!(inactive.s, Some(Vec::new()));
        assert!(validate_attribute_storage(&node(), &inactive).is_err());

        // A selected empty STRING may be encoded either with no tag 4 or an
        // explicit zero-length tag; both have the same ONNX value.
        for wire in [vec![0xa0, 0x01, 0x03], vec![0xa0, 0x01, 0x03, 0x22, 0x00]] {
            let selected = onnx_proto::AttributeProto::decode(wire.as_slice())
                .expect("decode selected empty STRING payload");
            validate_attribute_storage(&node(), &selected)
                .expect("selected empty STRING is canonical");
            assert_eq!(selected.s_value(), b"");
        }
    }
}
