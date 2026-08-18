// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Exact ONNX selection-operator canonicalization.

use super::{lookup_opset_version, parse_node_attributes};
use crate::loader::const_fold::common::read_tensor_i64s_and_shape;
use crate::onnx_proto::{self, NodeProto};
use crate::{AttributeValue, LayerSpec, WeightStore};
use ny_core::{LayerType, NyError, Result};

/// Split ONNX TopK's two heterogeneous outputs into two ordinary ny graph
/// nodes. Mapping both tensor names to one node aliases Values and Indices,
/// while retaining K as a runtime input does not match ny's embedded-k layer.
pub(super) fn canonicalize_standard_topk(
    node: &NodeProto,
    node_index: usize,
    weights: &WeightStore,
    opset_imports: &std::collections::HashMap<String, i64>,
) -> Result<Option<[LayerSpec; 2]>> {
    if node.op_type != "TopK" || !matches!(node.domain.as_str(), "" | "ai.onnx") {
        return Ok(None);
    }
    let opset = lookup_opset_version(opset_imports, &node.domain).ok_or_else(|| {
        NyError::ModelLoad(format!(
            "standard ONNX TopK node '{}' has no standard-domain opset authority",
            node.name
        ))
    })?;
    if node.output.len() != 2 || node.output.iter().any(String::is_empty) {
        return Err(NyError::ModelLoad(format!(
            "standard ONNX TopK node '{}' requires two non-empty outputs",
            node.name
        )));
    }

    let k = if opset < 10 {
        node.attribute
            .iter()
            .find(|attribute| attribute.name == "k")
            .map(onnx_proto::AttributeProto::i_value)
            .ok_or_else(|| {
                NyError::ModelLoad(format!(
                    "legacy standard ONNX TopK node '{}' is missing required k",
                    node.name
                ))
            })?
    } else {
        let k_name = node.input.get(1).ok_or_else(|| {
            NyError::ModelLoad(format!(
                "standard ONNX TopK node '{}' is missing K input",
                node.name
            ))
        })?;
        let (values, shape) = read_tensor_i64s_and_shape(weights, k_name).ok_or_else(|| {
            NyError::UnsupportedOp(format!(
                "standard ONNX TopK node '{}' requires an exact constant INT64 K input, got '{}'",
                node.name, k_name
            ))
        })?;
        if shape != [1] || values.len() != 1 {
            return Err(NyError::ModelLoad(format!(
                "standard ONNX TopK node '{}' requires K to have shape [1], got {:?}",
                node.name, shape
            )));
        }
        values[0]
    };
    if k <= 0 {
        return Err(NyError::ModelLoad(format!(
            "standard ONNX TopK node '{}' requires positive K, got {k}",
            node.name
        )));
    }

    let data = node.input.first().cloned().ok_or_else(|| {
        NyError::ModelLoad(format!(
            "standard ONNX TopK node '{}' is missing data input",
            node.name
        ))
    })?;
    let axis = node
        .attribute
        .iter()
        .find(|attribute| attribute.name == "axis")
        .map(onnx_proto::AttributeProto::i_value)
        .unwrap_or(-1);
    let base_name = if node.name.is_empty() {
        format!("topk_{node_index}")
    } else {
        node.name.clone()
    };

    let make = |output_index: usize, output_kind: &str| {
        let mut attributes = parse_node_attributes(node);
        attributes.remove("largest");
        attributes.remove("sorted");
        attributes.insert("axis".to_string(), AttributeValue::Int(axis));
        attributes.insert("k".to_string(), AttributeValue::Int(k));
        attributes.insert(
            "output".to_string(),
            AttributeValue::String(output_kind.to_string()),
        );
        LayerSpec {
            name: format!("{base_name}__ny_{output_kind}_{node_index}"),
            layer_type: LayerType::Topk,
            inputs: vec![data.clone()],
            outputs: vec![node.output[output_index].clone()],
            weights: None,
            attributes,
        }
    };

    Ok(Some([make(0, "values"), make(1, "indices")]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;
    use std::collections::HashMap;

    fn int_attr(name: &str, value: i64) -> onnx_proto::AttributeProto {
        onnx_proto::AttributeProto {
            name: name.to_string(),
            i: Some(value),
            r#type: onnx_proto::attribute_type::INT,
            ..Default::default()
        }
    }

    #[test]
    fn modern_topk_splits_values_and_indices_without_aliasing_k() {
        let node = NodeProto {
            name: "select".to_string(),
            op_type: "TopK".to_string(),
            input: vec!["x".to_string(), "k".to_string()],
            output: vec!["values".to_string(), "indices".to_string()],
            attribute: vec![int_attr("axis", -1)],
            ..Default::default()
        };
        let mut weights = WeightStore::new();
        weights.insert_integers("k".to_string(), arr1(&[3_i64]).into_dyn());
        let layers =
            canonicalize_standard_topk(&node, 4, &weights, &HashMap::from([(String::new(), 13)]))
                .unwrap()
                .unwrap();

        assert_eq!(layers[0].outputs, ["values"]);
        assert_eq!(layers[1].outputs, ["indices"]);
        assert_eq!(layers[0].inputs, ["x"]);
        assert_eq!(layers[1].inputs, ["x"]);
        assert_eq!(layers[0].attributes["k"], AttributeValue::Int(3));
        assert_eq!(
            layers[0].attributes["output"],
            AttributeValue::String("values".to_string())
        );
        assert_eq!(
            layers[1].attributes["output"],
            AttributeValue::String("indices".to_string())
        );
    }

    #[test]
    fn modern_topk_requires_exact_rank_one_singleton_k() {
        let node = NodeProto {
            name: "select".to_string(),
            op_type: "TopK".to_string(),
            input: vec!["x".to_string(), "k".to_string()],
            output: vec!["values".to_string(), "indices".to_string()],
            ..Default::default()
        };
        for k in [
            arr1(&[1_i64, 2]).into_dyn(),
            ndarray::arr0(1_i64).into_dyn(),
        ] {
            let mut weights = WeightStore::new();
            weights.insert_integers("k".to_string(), k);
            assert!(canonicalize_standard_topk(
                &node,
                0,
                &weights,
                &HashMap::from([(String::new(), 13)]),
            )
            .is_err());
        }
    }
}
