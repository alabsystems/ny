// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::onnx_proto;
use crate::{AttributeValue, LayerSpec};
use ny_core::LayerType;
use std::collections::HashMap;

use super::super::attributes::{node_attr_int, node_attr_ints};

pub(in crate::loader) fn try_fuse_logsumexp(
    nodes: &[onnx_proto::NodeProto],
    log_idx: usize,
    producer_by_output: &HashMap<&str, usize>,
    consumers_by_input: &HashMap<&str, Vec<usize>>,
) -> Option<(usize, LayerSpec, Vec<usize>)> {
    let log_node = &nodes[log_idx];
    if log_node.op_type != "Log" || log_node.input.is_empty() {
        return None;
    }

    let log_input = log_node.input.first()?.as_str();
    let reduce_idx = *producer_by_output.get(log_input)?;
    let reduce_node = &nodes[reduce_idx];
    if reduce_node.op_type != "ReduceSum" || reduce_node.input.is_empty() {
        return None;
    }
    if reduce_node.input.len() > 1 {
        return None;
    }

    let reduce_input = reduce_node.input.first()?.as_str();
    let exp_idx = *producer_by_output.get(reduce_input)?;
    let exp_node = &nodes[exp_idx];
    if exp_node.op_type != "Exp" || exp_node.input.is_empty() {
        return None;
    }

    let exp_output = exp_node.output.first()?.as_str();
    let exp_consumers = consumers_by_input.get(exp_output)?;
    if exp_consumers.len() != 1 {
        return None;
    }

    let reduce_output = reduce_node.output.first()?.as_str();
    let reduce_consumers = consumers_by_input.get(reduce_output)?;
    if reduce_consumers.len() != 1 {
        return None;
    }

    let axes = node_attr_ints(reduce_node, "axes").unwrap_or_else(|| vec![-1]);
    let keepdims = node_attr_int(reduce_node, "keepdims").unwrap_or(1);

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

    Some((exp_idx, spec, vec![exp_idx, reduce_idx, log_idx]))
}
