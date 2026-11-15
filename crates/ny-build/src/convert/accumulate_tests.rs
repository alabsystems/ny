// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Conversion tests for `LayerType::ScatterAdd` / `LayerType::IndexAdd`.

use super::*;
use crate::{AttributeValue, LayerSpec, WeightStore};
use ndarray::{ArrayD, IxDyn};
use ny_core::LayerType;
use ny_propagate::layers::BoundPropagation;
use ny_tensor::BoundedTensor;
use std::collections::{HashMap, HashSet};

fn spec(name: &str, layer_type: LayerType, axis: Option<i64>) -> LayerSpec {
    let mut attributes = HashMap::new();
    if let Some(axis) = axis {
        attributes.insert("axis".to_string(), AttributeValue::Int(axis));
    }
    LayerSpec {
        name: name.to_string(),
        layer_type,
        inputs: vec!["data".to_string(), "index".to_string(), "src".to_string()],
        outputs: vec!["out".to_string()],
        weights: None,
        attributes,
    }
}

fn scatter_add_spec(axis: i64) -> LayerSpec {
    spec("sa", LayerType::ScatterAdd, Some(axis))
}

fn index_add_spec(axis: i64) -> LayerSpec {
    spec("ia", LayerType::IndexAdd, Some(axis))
}

fn f32_array(shape: &[usize], values: Vec<f32>) -> ArrayD<f32> {
    ArrayD::from_shape_vec(IxDyn(shape), values).expect("valid f32 array")
}

fn i64_array(shape: &[usize], values: Vec<i64>) -> ArrayD<i64> {
    ArrayD::from_shape_vec(IxDyn(shape), values).expect("valid i64 array")
}

/// Insert an index tensor as both the exact i64 payload and its f32 mirror,
/// the way the ONNX loader and the trace bridge both record index constants.
fn insert_index(weights: &mut WeightStore, name: &str, shape: &[usize], values: Vec<i64>) {
    let floats: Vec<f32> = values.iter().map(|&v| v as f32).collect();
    weights.insert(name.to_string(), f32_array(shape, floats));
    weights.insert_integers(name.to_string(), i64_array(shape, values));
}

fn context<'a>(
    weights: &'a WeightStore,
    shapes: &'a HashMap<String, Vec<i64>>,
    constants: &'a HashSet<String>,
) -> ConvertContext<'a> {
    ConvertContext::new(weights, shapes, constants)
}

fn point(values: &[f32], shape: &[usize]) -> BoundedTensor {
    let arr = f32_array(shape, values.to_vec());
    BoundedTensor::new(arr.clone(), arr).expect("valid point tensor")
}

fn interval(lower: &[f32], upper: &[f32], shape: &[usize]) -> BoundedTensor {
    BoundedTensor::new(
        f32_array(shape, lower.to_vec()),
        f32_array(shape, upper.to_vec()),
    )
    .expect("valid interval tensor")
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

#[test]
fn scatter_add_embeds_constant_data_and_index_leaving_src_dynamic() {
    let mut weights = WeightStore::new();
    weights.insert("data".to_string(), f32_array(&[4], vec![0.0; 4]));
    insert_index(&mut weights, "index", &[2], vec![1, 3]);
    let shapes = HashMap::from([("data".to_string(), vec![4_i64])]);
    let constants = HashSet::new();

    let layer = context(&weights, &shapes, &constants)
        .convert_scatter_add(&scatter_add_spec(-1))
        .expect("ScatterAdd should convert");
    let Layer::ScatterAdd(scatter) = layer else {
        panic!("expected ScatterAdd layer");
    };
    // data + index embedded; only src remains an activation input.
    assert_eq!(scatter.activation_input_count(), 1);
}

#[test]
fn index_add_embeds_constant_data_and_index_leaving_src_dynamic() {
    let mut weights = WeightStore::new();
    weights.insert("data".to_string(), f32_array(&[3, 2], vec![0.0; 6]));
    insert_index(&mut weights, "index", &[2], vec![0, 2]);
    let shapes = HashMap::from([("data".to_string(), vec![3_i64, 2])]);
    let constants = HashSet::new();

    let layer = context(&weights, &shapes, &constants)
        .convert_index_add(&index_add_spec(-2))
        .expect("IndexAdd should convert");
    let Layer::IndexAdd(index_add) = layer else {
        panic!("expected IndexAdd layer");
    };
    assert_eq!(index_add.activation_input_count(), 1);
}

/// A trailing-relative (negative) axis — what the trace bridge emits — must
/// pass through the ONNX batch remap unchanged rather than being shifted again.
#[test]
fn negative_axis_passes_through_the_batch_remap_unchanged() {
    let mut weights = WeightStore::new();
    weights.insert("data".to_string(), f32_array(&[2, 4], vec![0.0; 8]));
    insert_index(&mut weights, "index", &[2, 1], vec![3, 0]);
    weights.insert("src".to_string(), f32_array(&[2, 1], vec![5.0, 7.0]));
    let shapes = HashMap::from([("data".to_string(), vec![2_i64, 4])]);
    let constants = HashSet::new();

    let layer = context(&weights, &shapes, &constants)
        .convert_scatter_add(&scatter_add_spec(-1))
        .expect("ScatterAdd should convert");
    let Layer::ScatterAdd(scatter) = layer else {
        panic!("expected ScatterAdd layer");
    };
    // Everything constant: no activation inputs, and the axis resolved to the
    // LAST axis (length 4), so index 3 is in range and index 0 writes column 0.
    assert_eq!(scatter.activation_input_count(), 0);
    let dummy = point(&[0.0], &[1]);
    let out = scatter
        .propagate_ibp(&dummy)
        .expect_err("no activation inputs are expected");
    // The arity check fires because the layer needs zero activation inputs.
    assert!(
        out.to_string().contains("expects 0 activation input"),
        "unexpected error: {out}"
    );
}

/// Axis 0 on a recorded rank-2 tensor is the (possibly stripped) batch axis:
/// no single encoding is right for both runtime layouts, so conversion refuses
/// rather than guessing (the Gather rule).
#[test]
fn positive_axis_zero_on_rank2_data_is_refused() {
    let mut weights = WeightStore::new();
    insert_index(&mut weights, "index", &[2], vec![0, 1]);
    let shapes = HashMap::from([("data".to_string(), vec![2_i64, 4])]);
    let constants = HashSet::new();

    let err = context(&weights, &shapes, &constants)
        .convert_scatter_add(&scatter_add_spec(0))
        .expect_err("axis 0 on rank-2 data must be refused");
    assert!(
        err.to_string().contains("batch"),
        "expected a batch-axis refusal, got: {err}"
    );
}

#[test]
fn missing_axis_attribute_is_refused_rather_than_defaulted() {
    let mut weights = WeightStore::new();
    insert_index(&mut weights, "index", &[2], vec![0, 1]);
    let shapes = HashMap::from([("data".to_string(), vec![4_i64])]);
    let constants = HashSet::new();
    let ctx = context(&weights, &shapes, &constants);

    for (layer_type, additive) in [(LayerType::ScatterAdd, true), (LayerType::IndexAdd, false)] {
        let bare = spec("no_axis", layer_type, None);
        let err = if additive {
            ctx.convert_scatter_add(&bare)
        } else {
            ctx.convert_index_add(&bare)
        }
        .expect_err("a missing axis attribute must be refused");
        assert!(
            err.to_string().contains("axis"),
            "expected a missing-axis refusal, got: {err}"
        );
    }
}

#[test]
fn wrong_input_count_is_refused() {
    let weights = WeightStore::new();
    let shapes = HashMap::new();
    let constants = HashSet::new();
    let mut two_inputs = scatter_add_spec(-1);
    two_inputs.inputs.pop();

    let err = context(&weights, &shapes, &constants)
        .convert_scatter_add(&two_inputs)
        .expect_err("a 2-input ScatterAdd must be refused");
    assert!(
        err.to_string().contains("requires 3 inputs"),
        "unexpected error: {err}"
    );
}

/// `IndexAdd`'s index is 1-D by definition. A src-shaped (rank > 1) constant
/// index is a `ScatterAdd`, and accepting it here would bound a different
/// function, so it is refused at conversion instead of at propagation.
#[test]
fn index_add_refuses_a_higher_rank_constant_index() {
    let mut weights = WeightStore::new();
    weights.insert("data".to_string(), f32_array(&[4], vec![0.0; 4]));
    insert_index(&mut weights, "index", &[2, 1], vec![0, 1]);
    let shapes = HashMap::from([("data".to_string(), vec![4_i64])]);
    let constants = HashSet::new();

    let err = context(&weights, &shapes, &constants)
        .convert_index_add(&index_add_spec(-1))
        .expect_err("a rank-2 IndexAdd index must be refused");
    assert!(
        err.to_string().contains("1-D index"),
        "unexpected error: {err}"
    );
}

/// A tensor that is constant BY NAME but whose value never materialized would
/// be dropped from the node's activation inputs by the graph builder while the
/// layer still expects to receive it. Refuse where the operand can be named.
#[test]
fn constant_marked_operand_without_a_value_is_refused() {
    let mut weights = WeightStore::new();
    insert_index(&mut weights, "index", &[2], vec![0, 1]);
    let shapes = HashMap::from([("data".to_string(), vec![4_i64])]);
    // `src` is declared constant but carries no payload anywhere.
    let constants = HashSet::from(["src".to_string()]);

    let err = context(&weights, &shapes, &constants)
        .convert_scatter_add(&scatter_add_spec(-1))
        .expect_err("a value-less constant operand must be refused");
    assert!(
        err.to_string().contains("marked constant") && err.to_string().contains("src"),
        "unexpected error: {err}"
    );
}

// ---------------------------------------------------------------------------
// Adversarial semantics
// ---------------------------------------------------------------------------

/// Duplicate destination indices are DETERMINISTIC for the additive family:
/// both contributions land on the same cell and sum, in either visit order.
/// The converted layer must reproduce that sum exactly, not a union.
#[test]
fn duplicate_indices_accumulate_in_both_orders() {
    // index = [1, 1]: both src elements target data[1].
    for (order, values) in [
        ("ascending", vec![2.0_f32, 5.0]),
        ("descending", vec![5.0, 2.0]),
    ] {
        let mut weights = WeightStore::new();
        weights.insert("data".to_string(), f32_array(&[3], vec![10.0, 20.0, 30.0]));
        insert_index(&mut weights, "index", &[2], vec![1, 1]);
        let shapes = HashMap::from([("data".to_string(), vec![3_i64])]);
        let constants = HashSet::new();

        let layer = context(&weights, &shapes, &constants)
            .convert_scatter_add(&scatter_add_spec(-1))
            .expect("ScatterAdd should convert");
        let Layer::ScatterAdd(scatter) = layer else {
            panic!("expected ScatterAdd layer");
        };
        let out = scatter
            .propagate_ibp(&point(&values, &[2]))
            .expect("ScatterAdd IBP should succeed");
        // 20 + 2 + 5 == 27 regardless of which src element is applied first.
        assert_eq!(
            out.lower().as_slice().unwrap(),
            &[10.0, 27.0, 30.0],
            "{order} order"
        );
        assert_eq!(
            out.upper().as_slice().unwrap(),
            &[10.0, 27.0, 30.0],
            "{order} order"
        );
    }
}

/// With intervals, duplicate targets accumulate endpoint-wise: the cell's
/// bounds are `data + sum(src_lower) .. data + sum(src_upper)`, which is the
/// exact interval image of the (order-independent) sum.
#[test]
fn duplicate_indices_accumulate_interval_endpoints() {
    let mut weights = WeightStore::new();
    weights.insert("data".to_string(), f32_array(&[2], vec![0.0, 0.0]));
    insert_index(&mut weights, "index", &[3], vec![0, 0, 1]);
    let shapes = HashMap::from([("data".to_string(), vec![2_i64])]);
    let constants = HashSet::new();

    let layer = context(&weights, &shapes, &constants)
        .convert_scatter_add(&scatter_add_spec(-1))
        .expect("ScatterAdd should convert");
    let Layer::ScatterAdd(scatter) = layer else {
        panic!("expected ScatterAdd layer");
    };
    let src = interval(&[-1.0, 2.0, 4.0], &[1.0, 3.0, 5.0], &[3]);
    let out = scatter
        .propagate_ibp(&src)
        .expect("ScatterAdd IBP should succeed");
    // cell 0 collects src[0] + src[1] = [-1+2, 1+3] = [1, 4]; cell 1 gets src[2].
    assert_eq!(out.lower().as_slice().unwrap(), &[1.0, 4.0]);
    assert_eq!(out.upper().as_slice().unwrap(), &[4.0, 5.0]);
}

/// A NON-constant index converts to the dynamic path rather than being
/// refused. The resulting hull must enclose EVERY index assignment: each src
/// element lands on exactly one cell, so widening every cell by the total
/// negative / positive contribution of src is sound.
#[test]
fn dynamic_index_yields_the_conservative_additive_union() {
    let mut weights = WeightStore::new();
    weights.insert("data".to_string(), f32_array(&[3], vec![0.0, 0.0, 0.0]));
    let shapes = HashMap::from([
        ("data".to_string(), vec![3_i64]),
        ("index".to_string(), vec![2_i64]),
    ]);
    let constants = HashSet::new();

    let layer = context(&weights, &shapes, &constants)
        .convert_scatter_add(&scatter_add_spec(-1))
        .expect("ScatterAdd with a dynamic index should convert");
    let Layer::ScatterAdd(scatter) = layer else {
        panic!("expected ScatterAdd layer");
    };
    // data embedded; index + src remain activation inputs.
    assert_eq!(scatter.activation_input_count(), 2);

    let index = interval(&[0.0, 0.0], &[2.0, 2.0], &[2]);
    let src = interval(&[-2.0, 1.0], &[-1.0, 4.0], &[2]);
    let out = scatter
        .propagate_ibp_binary(&index, &src)
        .expect("dynamic ScatterAdd IBP should succeed");

    // Lower: only the negative parts can pull a cell down => -2. Upper: only
    // the positive parts can push it up => 4.
    let lower = out.lower().as_slice().unwrap().to_vec();
    let upper = out.upper().as_slice().unwrap().to_vec();
    assert_eq!(lower, vec![-2.0, -2.0, -2.0]);
    assert_eq!(upper, vec![4.0, 4.0, 4.0]);

    // Enclosure check: enumerate every concrete index assignment over the
    // point-src corners and confirm each realizable output is inside the hull.
    for i0 in 0..3usize {
        for i1 in 0..3usize {
            for s0 in [-2.0_f32, -1.0] {
                for s1 in [1.0_f32, 4.0] {
                    let mut concrete = [0.0_f32; 3];
                    concrete[i0] += s0;
                    concrete[i1] += s1;
                    for (cell, value) in concrete.iter().enumerate() {
                        assert!(
                            lower[cell] <= *value && *value <= upper[cell],
                            "dynamic hull missed {value} at cell {cell} for index ({i0},{i1})"
                        );
                    }
                }
            }
        }
    }
}

/// Out-of-range and negative-beyond-range constant indices must fail closed at
/// propagation time rather than silently wrapping or clamping to a neighbour.
#[test]
fn out_of_range_constant_indices_fail_closed() {
    for bad in [3_i64, -4, i64::MAX, i64::MIN] {
        let mut weights = WeightStore::new();
        weights.insert("data".to_string(), f32_array(&[3], vec![0.0; 3]));
        insert_index(&mut weights, "index", &[1], vec![bad]);
        let shapes = HashMap::from([("data".to_string(), vec![3_i64])]);
        let constants = HashSet::new();

        // Fail-closed at EITHER boundary is acceptable — what must never
        // happen is a successful propagation that silently wrapped or clamped
        // the index onto a real cell. Assert one of the two refusals fired.
        let refusal = match context(&weights, &shapes, &constants)
            .convert_scatter_add(&scatter_add_spec(-1))
        {
            Err(err) => err.to_string(),
            Ok(Layer::ScatterAdd(scatter)) => scatter
                .propagate_ibp(&point(&[1.0], &[1]))
                .map(|out| {
                    panic!(
                        "index {bad} must fail closed, but IBP produced {:?}",
                        out.lower()
                    )
                })
                .unwrap_err()
                .to_string(),
            Ok(other) => panic!("expected ScatterAdd layer, got {other:?}"),
        };
        assert!(
            refusal.contains("out of bounds") || refusal.contains("out of range"),
            "index {bad}: expected an out-of-range refusal, got: {refusal}"
        );
    }
}

/// Negative indices inside range follow the Python convention (`-1` is the
/// last position) — the same rule `ScatterND` and `Gather` already use.
#[test]
fn negative_in_range_constant_index_wraps_from_the_end() {
    let mut weights = WeightStore::new();
    weights.insert("data".to_string(), f32_array(&[3], vec![0.0, 0.0, 0.0]));
    insert_index(&mut weights, "index", &[1], vec![-1]);
    let shapes = HashMap::from([("data".to_string(), vec![3_i64])]);
    let constants = HashSet::new();

    let layer = context(&weights, &shapes, &constants)
        .convert_scatter_add(&scatter_add_spec(-1))
        .expect("ScatterAdd should convert");
    let Layer::ScatterAdd(scatter) = layer else {
        panic!("expected ScatterAdd layer");
    };
    let out = scatter
        .propagate_ibp(&point(&[7.0], &[1]))
        .expect("ScatterAdd IBP should succeed");
    assert_eq!(out.lower().as_slice().unwrap(), &[0.0, 0.0, 7.0]);
}

/// gptoss-shaped MoE combine (`crates/nn-gptoss/src/moe_dispatch.rs`:
/// `output = output.index_add(0, &ids_tensor, &weighted)`): dim 0, a 1-D u32
/// token-id index, and a `[num_routed, hidden]` src accumulated into a
/// `[num_tokens, hidden]` destination. Row-wise accumulation must be exact,
/// including when two routed rows share a token id.
#[test]
fn gptoss_shaped_index_add_combines_rows_along_dim_zero() {
    let num_tokens = 4usize;
    let hidden = 3usize;
    let mut weights = WeightStore::new();
    weights.insert(
        "data".to_string(),
        f32_array(&[num_tokens, hidden], vec![0.0; num_tokens * hidden]),
    );
    // Three routed rows; rows 0 and 2 both land on token 1 (an expert combine
    // that revisits a token), row 1 lands on token 3.
    insert_index(&mut weights, "index", &[3], vec![1, 3, 1]);
    let shapes = HashMap::from([("data".to_string(), vec![num_tokens as i64, hidden as i64])]);
    let constants = HashSet::new();

    // dim 0 of a rank-2 tensor is trailing axis -2, exactly what the bridge's
    // `trailing_axis(0, 2)` emits.
    let layer = context(&weights, &shapes, &constants)
        .convert_index_add(&index_add_spec(-2))
        .expect("IndexAdd should convert");
    let Layer::IndexAdd(index_add) = layer else {
        panic!("expected IndexAdd layer");
    };

    let weighted = point(
        &[
            1.0, 2.0, 3.0, // -> token 1
            4.0, 5.0, 6.0, // -> token 3
            10.0, 20.0, 30.0, // -> token 1 (duplicate)
        ],
        &[3, hidden],
    );
    let out = index_add
        .propagate_ibp(&weighted)
        .expect("IndexAdd IBP should succeed");
    assert_eq!(
        out.shape(),
        &[num_tokens, hidden],
        "index_add preserves the destination shape"
    );
    assert_eq!(
        out.lower().as_slice().unwrap(),
        &[
            0.0, 0.0, 0.0, // token 0 untouched
            11.0, 22.0, 33.0, // token 1: both routed rows summed
            0.0, 0.0, 0.0, // token 2 untouched
            4.0, 5.0, 6.0, // token 3
        ]
    );
}

/// The same gptoss combine with interval-valued expert outputs: the duplicate
/// token accumulates endpoint-wise, and untouched rows keep the destination.
#[test]
fn gptoss_shaped_index_add_accumulates_intervals_endpointwise() {
    let mut weights = WeightStore::new();
    weights.insert("data".to_string(), f32_array(&[2, 2], vec![0.0; 4]));
    insert_index(&mut weights, "index", &[2], vec![0, 0]);
    let shapes = HashMap::from([("data".to_string(), vec![2_i64, 2])]);
    let constants = HashSet::new();

    let layer = context(&weights, &shapes, &constants)
        .convert_index_add(&index_add_spec(-2))
        .expect("IndexAdd should convert");
    let Layer::IndexAdd(index_add) = layer else {
        panic!("expected IndexAdd layer");
    };
    let src = interval(&[-1.0, 0.5, 2.0, -3.0], &[1.0, 1.5, 2.5, -2.0], &[2, 2]);
    let out = index_add
        .propagate_ibp(&src)
        .expect("IndexAdd IBP should succeed");
    // Row 0 gets both routed rows; row 1 is untouched.
    assert_eq!(out.lower().as_slice().unwrap(), &[1.0, -2.5, 0.0, 0.0]);
    assert_eq!(out.upper().as_slice().unwrap(), &[3.5, -0.5, 0.0, 0.0]);
}
