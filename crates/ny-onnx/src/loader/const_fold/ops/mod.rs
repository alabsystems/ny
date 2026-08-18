// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

mod cast;
mod constants;
mod elementwise;
mod quantization;
mod range;
mod reductions;
mod shape_ops;
#[cfg(test)]
mod shape_ops_tests;
mod slice;
mod trilu;

use crate::onnx_proto;
use crate::WeightStore;

use super::shape_inference::ConstFoldLookups;
use super::FoldedTensor;

pub(super) fn try_fold_shape_node(
    node: &onnx_proto::NodeProto,
    graph: &onnx_proto::GraphProto,
    lookups: &ConstFoldLookups,
    weights: &WeightStore,
) -> Option<FoldedTensor> {
    shape_ops::try_fold_shape_node(node, graph, lookups, weights)
}

pub(super) fn try_fold_all_const_node(
    node: &onnx_proto::NodeProto,
    weights: &WeightStore,
    model_unbatched: bool,
) -> Option<FoldedTensor> {
    quantization::try_fold(node, weights)
        .or_else(|| trilu::try_fold(node, weights))
        .or_else(|| elementwise::try_fold(node, weights))
        .or_else(|| reductions::try_fold(node, weights))
        .or_else(|| constants::try_fold(node, weights, model_unbatched))
        .or_else(|| range::try_fold(node, weights))
        .or_else(|| shape_ops::try_fold(node, weights, model_unbatched))
}
