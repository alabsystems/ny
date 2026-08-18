// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::shape_inference::ConstFoldLookups;
use super::shape_ops::{try_fold, try_fold_shape_node};
use crate::onnx_proto::{
    tensor_shape_proto, AttributeProto, GraphProto, NodeProto, TensorShapeProto, TensorTypeProto,
    TypeProto, ValueInfoProto,
};
use crate::WeightStore;
use ndarray::{ArrayD, IxDyn};
use std::collections::HashMap;

fn tensor_value_info(name: &str, shape: &[i64]) -> ValueInfoProto {
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

fn attr_int(name: &str, value: i64) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        i: Some(value),
        r#type: crate::onnx_proto::attribute_type::INT,
        ..Default::default()
    }
}

fn squeeze_node() -> NodeProto {
    NodeProto {
        input: vec!["data".to_string()],
        output: vec!["scalar".to_string()],
        op_type: "Squeeze".to_string(),
        attribute: vec![AttributeProto {
            name: "axes".to_string(),
            ints: vec![0],
            r#type: crate::onnx_proto::attribute_type::INTS,
            ..Default::default()
        }],
        ..Default::default()
    }
}

#[test]
fn squeeze_singleton_vector_produces_float_scalar() {
    let mut weights = WeightStore::new();
    weights.insert(
        "data".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![42.0]).unwrap(),
    );

    let folded = try_fold(&squeeze_node(), &weights, false).expect("Squeeze should fold");
    assert!(folded.float_data.shape().is_empty());
    assert_eq!(
        folded.float_data.iter().copied().collect::<Vec<_>>(),
        vec![42.0]
    );
}

#[test]
fn squeeze_singleton_vector_preserves_integer_scalar_rank() {
    let mut weights = WeightStore::new();
    weights.insert(
        "data".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![16_777_216.0]).unwrap(),
    );
    weights.insert_integers(
        "data".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![16_777_217]).unwrap(),
    );

    let folded = try_fold(&squeeze_node(), &weights, false).expect("Squeeze should fold");
    assert!(folded.float_data.shape().is_empty());
    let integer_data = folded.integer_data.expect("exact integer payload");
    assert!(integer_data.shape().is_empty());
    assert_eq!(
        integer_data.iter().copied().collect::<Vec<_>>(),
        vec![16_777_217]
    );
}

#[test]
fn cast_int64_identity_preserves_internal_shape_marker() {
    let sentinel = ny_core::reshape_copy_axis_sentinel(1).expect("axis sentinel");
    let mut weights = WeightStore::new();
    weights.insert(
        "shape".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0]).unwrap(),
    );
    weights.insert_integers(
        "shape".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![sentinel]).unwrap(),
    );
    weights.insert_integer_range("shape".to_string(), i64::MIN, i64::MAX);
    let cast = NodeProto {
        input: vec!["shape".to_string()],
        output: vec!["cast_shape".to_string()],
        op_type: "Cast".to_string(),
        attribute: vec![attr_int("to", 7)],
        ..Default::default()
    };

    let folded = try_fold(&cast, &weights, false).expect("INT64 identity Cast should fold");

    assert_eq!(
        folded.float_data.iter().copied().collect::<Vec<_>>(),
        vec![0.0]
    );
    assert_eq!(
        folded
            .integer_data
            .expect("exact sentinel payload")
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        vec![sentinel]
    );
}

#[test]
fn shape_const_fold_preserves_declared_dynamic_input_dim_over_ort_placeholder() {
    let graph = GraphProto {
        input: vec![tensor_value_info("hidden_states", &[1, -1, 1024])],
        ..Default::default()
    };
    let inferred_shapes = HashMap::from([("hidden_states".to_string(), vec![1, 1, 1024])]);
    let lookups = ConstFoldLookups::new(&graph, &inferred_shapes, false);
    let shape_node = NodeProto {
        input: vec!["hidden_states".to_string()],
        output: vec!["shape_out".to_string()],
        op_type: "Shape".to_string(),
        ..Default::default()
    };

    let weights = WeightStore::new();
    let shape = try_fold_shape_node(&shape_node, &graph, &lookups, &weights)
        .expect("Shape should infer declared graph input dimensions");
    let integer_data = shape
        .integer_data
        .expect("Shape output should preserve integer dimensions");
    assert_eq!(
        integer_data.iter().copied().collect::<Vec<_>>(),
        vec![
            1,
            ny_core::reshape_copy_axis_sentinel(1).expect("axis in range"),
            1024
        ],
        "dynamic graph-input dimensions must not be replaced by ORT placeholders"
    );
}

#[test]
fn shape_const_fold_honors_start_end_attributes() {
    // ONNX Shape (opset >= 15) with start/end slices the reported shape vector.
    // Folding the FULL shape here is a wrong constant (e.g. a batch-extracting
    // Shape(end=1) reporting [1, 80, 3000] instead of [1]) that poisons every
    // downstream Concat/Reshape fold.
    let graph = GraphProto {
        input: vec![tensor_value_info("input_features", &[1, 80, 3000])],
        ..Default::default()
    };
    let lookups = ConstFoldLookups::new(&graph, &HashMap::new(), false);
    let weights = WeightStore::new();

    let batch_only = NodeProto {
        input: vec!["input_features".to_string()],
        output: vec!["batch_dim".to_string()],
        op_type: "Shape".to_string(),
        attribute: vec![attr_int("start", 0), attr_int("end", 1)],
        ..Default::default()
    };
    let folded = try_fold_shape_node(&batch_only, &graph, &lookups, &weights)
        .expect("Shape(start=0, end=1) should fold");
    assert_eq!(
        folded
            .integer_data
            .expect("integer dims")
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        vec![1],
        "Shape(end=1) must report only the leading dim"
    );

    // Negative indices are rank-relative: start=-2 -> [80, 3000].
    let trailing = NodeProto {
        input: vec!["input_features".to_string()],
        output: vec!["trailing_dims".to_string()],
        op_type: "Shape".to_string(),
        attribute: vec![attr_int("start", -2)],
        ..Default::default()
    };
    let folded = try_fold_shape_node(&trailing, &graph, &lookups, &weights)
        .expect("Shape(start=-2) should fold");
    assert_eq!(
        folded
            .integer_data
            .expect("integer dims")
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        vec![80, 3000],
        "Shape(start=-2) must report the trailing two dims"
    );
}

#[test]
fn shape_const_fold_start_after_end_is_empty_vector() {
    let graph = GraphProto {
        input: vec![tensor_value_info("input", &[2, 3, 4])],
        ..Default::default()
    };
    let lookups = ConstFoldLookups::new(&graph, &HashMap::new(), false);
    let weights = WeightStore::new();
    let node = NodeProto {
        input: vec!["input".to_string()],
        output: vec!["empty_shape".to_string()],
        op_type: "Shape".to_string(),
        attribute: vec![attr_int("start", 2), attr_int("end", 1)],
        ..Default::default()
    };

    let folded = try_fold_shape_node(&node, &graph, &lookups, &weights)
        .expect("Shape(start > end) is a valid empty vector");
    assert_eq!(folded.float_data.shape(), &[0]);
    let integers = folded.integer_data.expect("exact INT64 empty vector");
    assert_eq!(integers.shape(), &[0]);
    assert!(integers.is_empty());
}

#[test]
fn shape_const_fold_start_end_keeps_original_axis_sentinels() {
    // Symbolic dims inside a start/end window must keep sentinels that name the
    // ORIGINAL tensor axis (sentinels identify the source axis to copy from,
    // not the output position).
    let graph = GraphProto {
        input: vec![tensor_value_info("hidden_states", &[2, -1, 512])],
        ..Default::default()
    };
    let lookups = ConstFoldLookups::new(&graph, &HashMap::new(), false);
    let weights = WeightStore::new();
    let node = NodeProto {
        input: vec!["hidden_states".to_string()],
        output: vec!["tail_dims".to_string()],
        op_type: "Shape".to_string(),
        attribute: vec![attr_int("start", 1)],
        ..Default::default()
    };
    let folded = try_fold_shape_node(&node, &graph, &lookups, &weights)
        .expect("Shape(start=1) should fold with sentinel for the symbolic dim");
    assert_eq!(
        folded
            .integer_data
            .expect("integer dims")
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        vec![
            ny_core::reshape_copy_axis_sentinel(1).expect("axis in range"),
            512
        ],
        "sentinel must reference original axis 1, not window offset 0"
    );
}

#[test]
fn shape_slice_preserves_copy_axis_sentinel_integer_store() {
    let mut weights = WeightStore::new();
    weights.insert(
        "shape".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[5]), vec![1.0, 8.0, 2.0, 0.0, 128.0]).unwrap(),
    );
    weights.insert_integers(
        "shape".to_string(),
        ArrayD::from_shape_vec(
            IxDyn(&[5]),
            vec![
                1,
                8,
                2,
                ny_core::reshape_copy_axis_sentinel(3).expect("axis in range"),
                128,
            ],
        )
        .unwrap(),
    );
    weights.insert(
        "starts".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![3.0]).unwrap(),
    );
    weights.insert(
        "ends".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![5.0]).unwrap(),
    );
    weights.insert(
        "axes".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0]).unwrap(),
    );

    let slice_node = NodeProto {
        input: vec![
            "shape".to_string(),
            "starts".to_string(),
            "ends".to_string(),
            "axes".to_string(),
        ],
        output: vec!["tail_shape".to_string()],
        op_type: "Slice".to_string(),
        ..Default::default()
    };

    let folded = try_fold(&slice_node, &weights, false).expect("Slice should fold");
    let integer_data = folded
        .integer_data
        .expect("Slice should preserve shape integer payload");
    assert_eq!(
        integer_data.iter().copied().collect::<Vec<_>>(),
        vec![
            ny_core::reshape_copy_axis_sentinel(3).expect("axis in range"),
            128
        ]
    );
}

#[test]
fn shape_then_gather_preserves_large_dimension_integer_store_2360() {
    let graph = GraphProto {
        input: vec![tensor_value_info("activation", &[1, 16_777_217])],
        ..Default::default()
    };
    let lookups = ConstFoldLookups::new(&graph, &HashMap::new(), false);
    let shape_node = NodeProto {
        input: vec!["activation".to_string()],
        output: vec!["shape_out".to_string()],
        op_type: "Shape".to_string(),
        ..Default::default()
    };

    let mut weights = WeightStore::new();
    weights.insert(
        "gather_index".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[]), vec![1.0]).unwrap(),
    );
    weights.insert_integers(
        "gather_index".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[]), vec![1]).unwrap(),
    );

    let shape = try_fold_shape_node(&shape_node, &graph, &lookups, &weights)
        .expect("Shape should infer graph input dimensions");
    let shape_ints = shape
        .integer_data
        .clone()
        .expect("Shape output should preserve integer dimensions");
    weights.insert("shape_out".to_string(), shape.float_data);
    weights.insert_integers("shape_out".to_string(), shape_ints);

    let gather_node = NodeProto {
        input: vec!["shape_out".to_string(), "gather_index".to_string()],
        output: vec!["axis_size".to_string()],
        op_type: "Gather".to_string(),
        attribute: vec![attr_int("axis", 0)],
        ..Default::default()
    };

    let axis_size = try_fold(&gather_node, &weights, false).expect("Gather should fold over Shape");
    let integer_data = axis_size
        .integer_data
        .expect("Gather should preserve exact integer payload");
    assert!(integer_data.shape().is_empty());
    assert_eq!(
        integer_data.iter().copied().collect::<Vec<_>>(),
        vec![16_777_217]
    );
}

#[test]
fn shape_cast_then_gather_preserves_large_dimension_integer_store_2360() {
    let graph = GraphProto {
        input: vec![tensor_value_info("activation", &[1, 16_777_217])],
        ..Default::default()
    };
    let lookups = ConstFoldLookups::new(&graph, &HashMap::new(), false);
    let shape_node = NodeProto {
        input: vec!["activation".to_string()],
        output: vec!["shape_out".to_string()],
        op_type: "Shape".to_string(),
        ..Default::default()
    };

    let mut weights = WeightStore::new();
    weights.insert(
        "gather_index".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[]), vec![1.0]).unwrap(),
    );
    weights.insert_integers(
        "gather_index".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[]), vec![1]).unwrap(),
    );

    let shape = try_fold_shape_node(&shape_node, &graph, &lookups, &weights)
        .expect("Shape should infer graph input dimensions");
    let shape_ints = shape
        .integer_data
        .clone()
        .expect("Shape output should preserve integer dimensions");
    weights.insert("shape_out".to_string(), shape.float_data);
    weights.insert_integers("shape_out".to_string(), shape_ints);

    let cast_node = NodeProto {
        input: vec!["shape_out".to_string()],
        output: vec!["shape_i32".to_string()],
        op_type: "Cast".to_string(),
        attribute: vec![attr_int("to", 6)],
        ..Default::default()
    };

    let cast_shape = try_fold(&cast_node, &weights, false).expect("Cast should fold over Shape");
    let cast_ints = cast_shape
        .integer_data
        .clone()
        .expect("integer Cast should preserve the exact shape payload");
    assert_eq!(
        cast_ints.iter().copied().collect::<Vec<_>>(),
        vec![1, 16_777_217]
    );
    weights.insert("shape_i32".to_string(), cast_shape.float_data);
    weights.insert_integers("shape_i32".to_string(), cast_ints);

    let gather_node = NodeProto {
        input: vec!["shape_i32".to_string(), "gather_index".to_string()],
        output: vec!["axis_size".to_string()],
        op_type: "Gather".to_string(),
        attribute: vec![attr_int("axis", 0)],
        ..Default::default()
    };

    let axis_size = try_fold(&gather_node, &weights, false).expect("Gather should fold over Cast");
    let integer_data = axis_size
        .integer_data
        .expect("Gather should preserve exact integer payload after Cast");
    assert!(integer_data.shape().is_empty());
    assert_eq!(
        integer_data.iter().copied().collect::<Vec<_>>(),
        vec![16_777_217]
    );
}

/// Regression test for Prover audit finding: INT64→INT32 Cast with values
/// outside i32 range must materialize the actual wrapping result, not keep
/// the pre-cast float_data.
/// See: P1 commit 5e7a2f0 "Re: #2360, Re: #3769 audit Cast overflow semantics"
#[test]
fn cast_int64_to_int32_overflow_materializes_wrapped_value_2360() {
    let mut weights = WeightStore::new();
    // 2_147_483_700 exceeds i32::MAX (2_147_483_647) by 53.
    // Wrapping cast: 0x80000034 interpreted as i32 = -2_147_483_596.
    let overflow_val: i64 = 2_147_483_700;
    weights.insert(
        "src".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![overflow_val as f32]).unwrap(),
    );
    weights.insert_integers(
        "src".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![overflow_val]).unwrap(),
    );

    let cast_node = NodeProto {
        input: vec!["src".to_string()],
        output: vec!["dst".to_string()],
        op_type: "Cast".to_string(),
        attribute: vec![attr_int("to", 6)], // ONNX INT32 = 6
        ..Default::default()
    };

    let result = try_fold(&cast_node, &weights, false).expect("Cast should fold");

    // Integer payload must contain the wrapped i32 value widened back to i64.
    let expected_i32 = overflow_val as i32; // wrapping: -2_147_483_596
    let expected_i64 = expected_i32 as i64;
    let ints = result
        .integer_data
        .expect("Cast INT32 should always produce integer_data");
    assert_eq!(
        ints.iter().copied().collect::<Vec<_>>(),
        vec![expected_i64],
        "integer_data must reflect the wrapped INT32 value, not the original INT64"
    );

    // float_data must be derived from the casted integer, not the pre-cast value.
    let expected_f32 = expected_i64 as f32; // -2147483648.0 (nearest f32)
    assert_eq!(
        result.float_data.iter().copied().collect::<Vec<_>>(),
        vec![expected_f32],
        "float_data must match the casted INT32 value, not the pre-cast INT64"
    );
}

/// Edge-case coverage for INT64→INT32 wrapping: boundary values at i32::MIN-1,
/// i32::MAX+1, and i64::MAX.  Verifies Rust `(v as i32) as i64` truncation
/// matches C++ static_cast semantics used by ONNX Runtime.
///
/// Requested by W4 in commit 4a5f1c17d ## Next: "verify wrapping correctness
/// for edge cases (i32::MIN-1, i32::MAX+1, i64::MAX)".
#[test]
fn cast_int64_to_int32_boundary_edge_cases_2360() {
    let cases: &[(i64, i64, &str)] = &[
        // (input_i64, expected_after_wrap, label)
        (
            i32::MAX as i64 + 1,
            i32::MIN as i64,
            "i32::MAX+1 → i32::MIN",
        ),
        (
            i32::MIN as i64 - 1,
            i32::MAX as i64,
            "i32::MIN-1 → i32::MAX",
        ),
        (i64::MAX, -1i64, "i64::MAX → -1"),
        // In-range boundary values should pass through unchanged.
        (i32::MAX as i64, i32::MAX as i64, "i32::MAX → unchanged"),
        (i32::MIN as i64, i32::MIN as i64, "i32::MIN → unchanged"),
        (0i64, 0i64, "zero → unchanged"),
    ];

    for &(input_val, expected_i64, label) in cases {
        let mut weights = WeightStore::new();
        weights.insert(
            "src".to_string(),
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![input_val as f32]).unwrap(),
        );
        weights.insert_integers(
            "src".to_string(),
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![input_val]).unwrap(),
        );

        let cast_node = NodeProto {
            input: vec!["src".to_string()],
            output: vec!["dst".to_string()],
            op_type: "Cast".to_string(),
            attribute: vec![attr_int("to", 6)], // ONNX INT32 = 6
            ..Default::default()
        };

        let result = try_fold(&cast_node, &weights, false)
            .unwrap_or_else(|| panic!("{label}: Cast should fold"));

        let ints = result
            .integer_data
            .unwrap_or_else(|| panic!("{label}: Cast INT32 should produce integer_data"));
        assert_eq!(
            ints.iter().copied().collect::<Vec<_>>(),
            vec![expected_i64],
            "{label}: integer_data mismatch"
        );

        let expected_f32 = expected_i64 as f32;
        assert_eq!(
            result.float_data.iter().copied().collect::<Vec<_>>(),
            vec![expected_f32],
            "{label}: float_data mismatch"
        );
    }
}

/// Float->int Cast const-fold must apply trunc-toward-zero to the float view
/// (#cctsdb B1): folding Cast(0.7 -> INT64) as 0.7 bakes a wrong constant
/// into the network. Input has only a float payload (no integer view).
#[test]
fn cast_float_to_int_const_fold_truncates_float_view() {
    let mut weights = WeightStore::new();
    weights.insert(
        "src".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[4]), vec![0.7_f32, -1.5, 2.0, -0.2]).unwrap(),
    );

    let cast_node = NodeProto {
        input: vec!["src".to_string()],
        output: vec!["dst".to_string()],
        op_type: "Cast".to_string(),
        attribute: vec![attr_int("to", 7)], // ONNX INT64 = 7
        ..Default::default()
    };

    let result = try_fold(&cast_node, &weights, false).expect("Cast should fold");
    assert_eq!(
        result.float_data.iter().copied().collect::<Vec<_>>(),
        vec![0.0, -1.0, 2.0, -0.0],
        "float view must be truncated toward zero, not passed through"
    );
}

/// Cast->BOOL const-fold must materialize the indicator `x != 0`, not pass the
/// value through: folding Cast(2.0 -> BOOL) as 2.0 bakes a wrong constant into
/// the network exactly as folding Cast(0.7 -> INT64) as 0.7 would. It is also
/// NOT truncation — trunc(0.5) = 0 but bool(0.5) = 1.
#[test]
fn cast_to_bool_const_fold_materializes_indicator() {
    let mut weights = WeightStore::new();
    weights.insert(
        "src".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[5]), vec![2.0_f32, 0.0, -3.5, 0.5, 1.0]).unwrap(),
    );

    let cast_node = NodeProto {
        input: vec!["src".to_string()],
        output: vec!["dst".to_string()],
        op_type: "Cast".to_string(),
        attribute: vec![attr_int("to", 9)], // ONNX BOOL = 9
        ..Default::default()
    };

    let result = try_fold(&cast_node, &weights, false).expect("Cast should fold");
    assert_eq!(
        result.float_data.iter().copied().collect::<Vec<_>>(),
        vec![1.0, 0.0, 1.0, 1.0, 1.0],
        "BOOL cast is x != 0, neither identity nor truncation"
    );
    assert_eq!(
        result
            .integer_data
            .as_ref()
            .expect("BOOL cast should produce integer_data")
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        vec![1_i64, 0, 1, 1, 1]
    );
    assert_eq!(result.integer_range, Some((0, 1)));
}

/// Float->FLOAT Cast const-fold stays identity (no truncation).
#[test]
fn cast_float_to_float_const_fold_stays_identity() {
    let mut weights = WeightStore::new();
    weights.insert(
        "src".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.7_f32, -1.5]).unwrap(),
    );

    let cast_node = NodeProto {
        input: vec!["src".to_string()],
        output: vec!["dst".to_string()],
        op_type: "Cast".to_string(),
        attribute: vec![attr_int("to", 1)], // ONNX FLOAT = 1
        ..Default::default()
    };

    let result = try_fold(&cast_node, &weights, false).expect("Cast should fold");
    assert_eq!(
        result.float_data.iter().copied().collect::<Vec<_>>(),
        vec![0.7, -1.5]
    );
}
