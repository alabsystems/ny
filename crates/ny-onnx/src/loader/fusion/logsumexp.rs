// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::onnx_proto::{self, attribute_type};
use crate::{AttributeValue, LayerSpec};
use ny_core::LayerType;
use std::collections::{HashMap, HashSet};

use super::helpers::fused_subgraph_is_closed;

pub(in crate::loader) fn try_fuse_logsumexp(
    nodes: &[onnx_proto::NodeProto],
    log_idx: usize,
    producer_by_output: &HashMap<&str, usize>,
    consumers_by_input: &HashMap<&str, Vec<usize>>,
    graph_output_names: &HashSet<String>,
) -> Option<(usize, LayerSpec, Vec<usize>)> {
    let log_node = &nodes[log_idx];
    if log_node.op_type != "Log"
        || log_node.input.len() != 1
        || log_node.output.len() != 1
        || log_node.input[0].is_empty()
        || log_node.output[0].is_empty()
        || !log_node.attribute.is_empty()
    {
        return None;
    }

    let log_input = log_node.input.first()?.as_str();
    let reduce_idx = *producer_by_output.get(log_input)?;
    let reduce_node = &nodes[reduce_idx];
    if reduce_node.op_type != "ReduceSum"
        || reduce_node.input.len() != 1
        || reduce_node.output.len() != 1
        || reduce_node.input[0].is_empty()
        || reduce_node.output[0].is_empty()
    {
        return None;
    }

    let reduce_input = reduce_node.input.first()?.as_str();
    let exp_idx = *producer_by_output.get(reduce_input)?;
    let exp_node = &nodes[exp_idx];
    if exp_node.op_type != "Exp"
        || exp_node.input.len() != 1
        || exp_node.output.len() != 1
        || exp_node.input[0].is_empty()
        || exp_node.output[0].is_empty()
        || !exp_node.attribute.is_empty()
    {
        return None;
    }

    let exp_output = exp_node.output.first()?.as_str();
    let exp_consumers = consumers_by_input.get(exp_output)?;
    if exp_consumers.as_slice() != [reduce_idx] {
        return None;
    }

    let reduce_output = reduce_node.output.first()?.as_str();
    let reduce_consumers = consumers_by_input.get(reduce_output)?;
    if reduce_consumers.as_slice() != [log_idx] {
        return None;
    }

    let (axes, keepdims) = reduce_sum_attributes(reduce_node)?;

    let mut attributes = HashMap::new();
    attributes.insert("axes".to_string(), AttributeValue::Ints(axes));
    attributes.insert("keepdims".to_string(), AttributeValue::Int(keepdims));

    let name = if log_node.name.is_empty() {
        log_node.output.first().cloned().unwrap_or_default()
    } else {
        log_node.name.clone()
    };

    let spec = LayerSpec {
        name,
        layer_type: LayerType::LogSumExp,
        inputs: vec![exp_node.input.first()?.clone()],
        outputs: log_node.output.clone(),
        weights: None,
        attributes,
    };

    let fused = vec![exp_idx, reduce_idx, log_idx];
    if !fused_subgraph_is_closed(
        nodes,
        &fused,
        &spec.outputs,
        consumers_by_input,
        graph_output_names,
    ) {
        return None;
    }

    Some((exp_idx, spec, fused))
}

fn reduce_sum_attributes(node: &onnx_proto::NodeProto) -> Option<(Vec<i64>, i64)> {
    let mut axes = None;
    let mut keepdims = None;
    let mut noop_with_empty_axes = None;
    for attribute in &node.attribute {
        match attribute.name.as_str() {
            "axes" if axes.is_none() && attribute.r#type == attribute_type::INTS => {
                axes = Some(attribute.ints.clone());
            }
            "keepdims"
                if keepdims.is_none()
                    && attribute.r#type == attribute_type::INT
                    && matches!(attribute.i_value(), 0 | 1) =>
            {
                keepdims = Some(attribute.i_value());
            }
            "noop_with_empty_axes"
                if noop_with_empty_axes.is_none()
                    && attribute.r#type == attribute_type::INT
                    && matches!(attribute.i_value(), 0 | 1) =>
            {
                noop_with_empty_axes = Some(attribute.i_value());
            }
            _ => return None,
        }
    }

    let axes = axes.unwrap_or_default();
    if noop_with_empty_axes.unwrap_or(0) == 1 && axes.is_empty() {
        // ReduceSum is the identity here, so Log(ReduceSum(Exp(x))) == x,
        // not a LogSumExp reduction.
        return None;
    }
    Some((axes, keepdims.unwrap_or(1)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(op_type: &str, input: &str, output: &str) -> onnx_proto::NodeProto {
        onnx_proto::NodeProto {
            op_type: op_type.to_string(),
            input: vec![input.to_string()],
            output: vec![output.to_string()],
            ..Default::default()
        }
    }

    fn int_attr(name: &str, value: i64) -> onnx_proto::AttributeProto {
        onnx_proto::AttributeProto {
            name: name.to_string(),
            i: Some(value),
            r#type: attribute_type::INT,
            ..Default::default()
        }
    }

    fn nodes() -> Vec<onnx_proto::NodeProto> {
        vec![
            node("Exp", "x", "exp"),
            node("ReduceSum", "exp", "sum"),
            node("Log", "sum", "out"),
        ]
    }

    fn fuse(
        nodes: &[onnx_proto::NodeProto],
        graph_outputs: &[&str],
    ) -> Option<(usize, LayerSpec, Vec<usize>)> {
        let mut producers = HashMap::new();
        let mut consumers: HashMap<&str, Vec<usize>> = HashMap::new();
        for (idx, node) in nodes.iter().enumerate() {
            for output in &node.output {
                producers.insert(output.as_str(), idx);
            }
            for input in &node.input {
                consumers.entry(input.as_str()).or_default().push(idx);
            }
        }
        let graph_outputs = graph_outputs
            .iter()
            .map(|output| (*output).to_string())
            .collect();
        try_fuse_logsumexp(nodes, 2, &producers, &consumers, &graph_outputs)
    }

    #[test]
    fn missing_reduce_axes_means_reduce_all() {
        let (_, spec, _) = fuse(&nodes(), &[]).expect("canonical LogSumExp must fuse");
        assert_eq!(spec.attributes["axes"], AttributeValue::Ints(Vec::new()));
        assert_eq!(spec.attributes["keepdims"], AttributeValue::Int(1));
    }

    #[test]
    fn noop_empty_axes_and_malformed_attributes_decline_fusion() {
        let mut noop = nodes();
        noop[1].attribute.push(int_attr("noop_with_empty_axes", 1));
        assert!(fuse(&noop, &[]).is_none());

        let mut malformed = nodes();
        malformed[1].attribute.push(int_attr("keepdims", 2));
        assert!(fuse(&malformed, &[]).is_none());

        let mut unknown = nodes();
        unknown[1].attribute.push(int_attr("mystery", 0));
        assert!(fuse(&unknown, &[]).is_none());
    }

    #[test]
    fn logsumexp_preserves_graph_output_intermediates() {
        for intermediate in ["exp", "sum"] {
            assert!(fuse(&nodes(), &[intermediate]).is_none());
        }
    }
}
