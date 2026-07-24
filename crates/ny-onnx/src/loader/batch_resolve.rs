// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Resolve a symbolic leading (batch) dimension on graph inputs to a concrete
//! verification value before shape inference and const-folding.
//!
//! Many exported transformer graphs (ViT, cGAN, smart_turn, …) declare their
//! runtime input as `[?batch_size, C, H, W]`: axis 0 is a symbolic `dim_param`
//! while the trailing axes are concrete. With a free leading symbol, ONNX
//! Runtime cannot resolve ranks through the attention
//! `Shape → Gather(axis 0) → Concat → Reshape → Transpose` chain, so the whole
//! shape-inference pass aborts (e.g. ViT's `[TypeInferenceError] Invalid
//! attribute perm {0, 2, 1}, input shape = {48}`) and every downstream tensor
//! falls back to conservative unbounded bounds. The const-folding path likewise
//! leaves the symbolic axis as `-1`, so the `Shape` op folds to copy-axis
//! sentinels instead of a concrete value and the attention reshape→transpose
//! chain mis-folds to the wrong rank.
//!
//! For VNN-COMP, the VNN-LIB property pins a single fixed-size input image, so
//! the verification batch is exactly 1. Pinning axis 0 of each runtime graph
//! input to that concrete value lets all three consumers (the ORT bytes, the
//! const-fold graph-shape lookup, and `value_info_to_tensor_spec`) see the same
//! static dim and fold the attention chain correctly.
//!
//! # Soundness
//!
//! This rewrite is *semantics-preserving*: it only sets a static value on the
//! batch dimension that was already implicitly 1 for verification. It changes
//! shape metadata only — never a weight, op, or attribute — so the computed
//! function is unchanged (verified empirically: ORT outputs of the original
//! dynamic-batch model and the batch-1-fixed model on the same batch-1 input are
//! byte-identical). Guard rails keep it sound:
//!
//! * Only axis 0 of runtime graph **inputs** is touched. Initializers (weights)
//!   are skipped, and no inner axis is ever modified — an inner symbolic dim
//!   (e.g. an LLM sequence length) must keep its dynamic shape.
//! * The axis is rewritten only when it is symbolic (`dim_param`) or
//!   non-positive (`dim_value <= 0` / unset). A positive concrete leading dim is
//!   left untouched (the helper is a no-op in that case).

use crate::onnx_proto;

/// Default verification batch size for VNN-COMP: the VNN-LIB property fixes a
/// single input, so the batch is exactly 1.
pub(super) const VERIFICATION_BATCH_SIZE: i64 = 1;

/// Resolve the leading (batch) dimension of every runtime graph input to
/// `batch`, in place.
///
/// Returns `true` if any dimension was rewritten (useful for logging / tests).
/// See the module docs for the exact rule and soundness argument.
pub(super) fn resolve_batch_dim(graph: &mut onnx_proto::GraphProto, batch: i64) -> bool {
    if batch <= 0 {
        return false;
    }

    let initializer_names: std::collections::HashSet<&str> = graph
        .initializer
        .iter()
        .map(|init| init.name.as_str())
        .collect();

    let mut changed = false;
    for input in graph.input.iter_mut() {
        // Skip initializers exposed as graph inputs (weights/constants): their
        // shape is the data's real shape and must never be rewritten.
        if initializer_names.contains(input.name.as_str()) {
            continue;
        }
        let Some(shape) = input
            .r#type
            .as_mut()
            .and_then(|ty| ty.tensor_type.as_mut())
            .and_then(|tt| tt.shape.as_mut())
        else {
            continue;
        };
        let Some(leading) = shape.dim.first_mut() else {
            continue;
        };
        if dim_is_positive_concrete(leading) {
            continue; // already static; nothing to do.
        }
        leading.value = Some(onnx_proto::tensor_shape_proto::dimension::Value::DimValue(
            batch,
        ));
        changed = true;
    }
    changed
}

/// True when this dimension is a positive concrete `dim_value` (so it is already
/// static and must not be overwritten).
fn dim_is_positive_concrete(dim: &onnx_proto::tensor_shape_proto::Dimension) -> bool {
    matches!(
        dim.value,
        Some(onnx_proto::tensor_shape_proto::dimension::Value::DimValue(value)) if value > 0
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::onnx_proto::{
        tensor_shape_proto, TensorProto, TensorShapeProto, TensorTypeProto, TypeProto,
        ValueInfoProto,
    };

    fn dim_value(value: i64) -> tensor_shape_proto::Dimension {
        tensor_shape_proto::Dimension {
            value: Some(tensor_shape_proto::dimension::Value::DimValue(value)),
        }
    }

    fn dim_param(name: &str) -> tensor_shape_proto::Dimension {
        tensor_shape_proto::Dimension {
            value: Some(tensor_shape_proto::dimension::Value::DimParam(
                name.to_string(),
            )),
        }
    }

    fn dim_unset() -> tensor_shape_proto::Dimension {
        tensor_shape_proto::Dimension { value: None }
    }

    fn input(name: &str, dims: Vec<tensor_shape_proto::Dimension>) -> ValueInfoProto {
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

    fn read_dims(info: &ValueInfoProto) -> Vec<Option<i64>> {
        info.r#type
            .as_ref()
            .unwrap()
            .tensor_type
            .as_ref()
            .unwrap()
            .shape
            .as_ref()
            .unwrap()
            .dim
            .iter()
            .map(|d| match &d.value {
                Some(tensor_shape_proto::dimension::Value::DimValue(v)) => Some(*v),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn resolves_symbolic_batch_dim_param() {
        let mut graph = onnx_proto::GraphProto {
            input: vec![input(
                "x",
                vec![
                    dim_param("batch_size"),
                    dim_value(3),
                    dim_value(32),
                    dim_value(32),
                ],
            )],
            ..Default::default()
        };
        assert!(resolve_batch_dim(&mut graph, 1));
        assert_eq!(
            read_dims(&graph.input[0]),
            vec![Some(1), Some(3), Some(32), Some(32)]
        );
    }

    #[test]
    fn resolves_nonpositive_leading_dim() {
        let mut graph = onnx_proto::GraphProto {
            input: vec![input("x", vec![dim_value(-1), dim_value(10)])],
            ..Default::default()
        };
        assert!(resolve_batch_dim(&mut graph, 1));
        assert_eq!(read_dims(&graph.input[0]), vec![Some(1), Some(10)]);
    }

    #[test]
    fn resolves_unset_leading_dim() {
        let mut graph = onnx_proto::GraphProto {
            input: vec![input("x", vec![dim_unset(), dim_value(10)])],
            ..Default::default()
        };
        assert!(resolve_batch_dim(&mut graph, 1));
        assert_eq!(read_dims(&graph.input[0]), vec![Some(1), Some(10)]);
    }

    #[test]
    fn noop_when_leading_dim_already_positive() {
        let mut graph = onnx_proto::GraphProto {
            input: vec![input("x", vec![dim_value(1), dim_value(10)])],
            ..Default::default()
        };
        assert!(!resolve_batch_dim(&mut graph, 1));
        assert_eq!(read_dims(&graph.input[0]), vec![Some(1), Some(10)]);
    }

    #[test]
    fn never_touches_inner_symbolic_dim() {
        // [1, ?seq, 1024]: leading already concrete; inner symbolic seq must
        // survive untouched (LLM sequence length).
        let mut graph = onnx_proto::GraphProto {
            input: vec![input(
                "hidden",
                vec![dim_value(1), dim_param("seq"), dim_value(1024)],
            )],
            ..Default::default()
        };
        assert!(!resolve_batch_dim(&mut graph, 1));
        assert_eq!(read_dims(&graph.input[0]), vec![Some(1), None, Some(1024)]);
    }

    #[test]
    fn resolves_only_leading_when_both_axes_symbolic() {
        // [?batch, ?seq, 1024]: only the batch axis is pinned; the inner
        // symbolic seq dim is preserved.
        let mut graph = onnx_proto::GraphProto {
            input: vec![input(
                "hidden",
                vec![dim_param("batch"), dim_param("seq"), dim_value(1024)],
            )],
            ..Default::default()
        };
        assert!(resolve_batch_dim(&mut graph, 1));
        assert_eq!(read_dims(&graph.input[0]), vec![Some(1), None, Some(1024)]);
    }

    #[test]
    fn skips_initializer_inputs() {
        let mut graph = onnx_proto::GraphProto {
            input: vec![input("w", vec![dim_value(-1), dim_value(48)])],
            initializer: vec![TensorProto {
                name: "w".to_string(),
                ..Default::default()
            }],
            ..Default::default()
        };
        // The only input is an initializer, so nothing is rewritten.
        assert!(!resolve_batch_dim(&mut graph, 1));
        assert_eq!(read_dims(&graph.input[0]), vec![Some(-1), Some(48)]);
    }

    #[test]
    fn skips_scalar_input_without_dims() {
        let mut graph = onnx_proto::GraphProto {
            input: vec![input("scalar", vec![])],
            ..Default::default()
        };
        assert!(!resolve_batch_dim(&mut graph, 1));
    }

    #[test]
    fn noop_for_nonpositive_batch_arg() {
        let mut graph = onnx_proto::GraphProto {
            input: vec![input("x", vec![dim_param("batch_size"), dim_value(3)])],
            ..Default::default()
        };
        assert!(!resolve_batch_dim(&mut graph, 0));
        assert_eq!(read_dims(&graph.input[0]), vec![None, Some(3)]);
    }
}
