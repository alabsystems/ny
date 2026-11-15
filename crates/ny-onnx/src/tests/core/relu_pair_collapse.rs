// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! `#relu-pair-collapse`: the loader undoes an exact ReLU split, and declines
//! on every near miss.
//!
//! The fixture is a 3-neuron layer `z = ReLU(W x + b)` split the relusplitter
//! way: rows 0 and 2 are SPLIT (`+1`/`-1` over a column pair whose rows are
//! exact negations), row 1 is a PASS-THROUGH (`+1` alone, zero merge bias), and
//! the hidden columns are permuted.

use super::onnx_fixture::{f32_initializer, int_attr, model_bytes, node, value_info};
use crate::load_onnx_bytes;
use crate::onnx_proto;
use ndarray::arr2;

const W0: [[f32; 2]; 3] = [[1.5, -2.0], [0.5, 0.25], [-1.0, 3.0]];
const A0: [f32; 3] = [0.1, -0.2, 0.3];
const C: [f32; 3] = [0.05, 0.0, -0.125];

fn neg(row: [f32; 2]) -> [f32; 2] {
    [-row[0], -row[1]]
}

/// The authored split: hidden columns
/// `[W0[2], -W0[0], W0[1], W0[0], -W0[2]]` and the merge matrix below.
fn split_parts() -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let a_rows = [W0[2], neg(W0[0]), W0[1], W0[0], neg(W0[2])];
    let a_bias = vec![A0[2], -A0[0], A0[1], A0[0], -A0[2]];
    let b = vec![
        0.0, -1.0, 0.0, 1.0, 0.0, // row 0: +col3 -col1  (split of neuron 0)
        0.0, 0.0, 1.0, 0.0, 0.0, // row 1: +col2         (pass-through of neuron 1)
        1.0, 0.0, 0.0, 0.0, -1.0, // row 2: +col0 -col4  (split of neuron 2)
    ];
    (a_rows.concat(), a_bias, b, C.to_vec())
}

fn gemm(name: &str, inputs: &[&str], output: &str, alpha: f32) -> onnx_proto::NodeProto {
    let mut attrs = vec![int_attr("transB", 1)];
    if alpha != 1.0 {
        attrs.push(onnx_proto::AttributeProto {
            name: "alpha".to_string(),
            f: Some(alpha),
            r#type: onnx_proto::attribute_type::FLOAT,
            ..Default::default()
        });
    }
    node(name, "Gemm", inputs, &[output], attrs)
}

fn split_model(a: &[f32], a_bias: &[f32], b: &[f32], c: &[f32], alpha2: f32) -> Vec<u8> {
    model_bytes(
        "relu_pair_collapse_fixture",
        13,
        vec![
            gemm("g1", &["input", "w1", "b1"], "t", 1.0),
            node("r1", "Relu", &["t"], &["h"], Vec::new()),
            gemm("g2", &["h", "w2", "b2"], "u", alpha2),
            node("r2", "Relu", &["u"], &["output"], Vec::new()),
        ],
        vec![
            f32_initializer("w1", &[5, 2], a),
            f32_initializer("b1", &[5], a_bias),
            f32_initializer("w2", &[3, 5], b),
            f32_initializer("b2", &[3], c),
        ],
        vec![value_info("input", &[1, 2])],
        vec![value_info("output", &[1, 3])],
    )
}

/// The authored model's value, computed directly in f64 from its definition.
fn authored(a: &[f32], a_bias: &[f32], b: &[f32], c: &[f32], alpha2: f32, x: [f32; 2]) -> Vec<f64> {
    let h: Vec<f64> = (0..5)
        .map(|j| {
            let t = f64::from(a[2 * j]) * f64::from(x[0])
                + f64::from(a[2 * j + 1]) * f64::from(x[1])
                + f64::from(a_bias[j]);
            t.max(0.0)
        })
        .collect();
    (0..3)
        .map(|i| {
            let u: f64 = (0..5).map(|j| f64::from(b[5 * i + j]) * h[j]).sum::<f64>()
                * f64::from(alpha2)
                + f64::from(c[i]);
            u.max(0.0)
        })
        .collect()
}

fn relu_layers(bytes: &[u8]) -> usize {
    let model = load_onnx_bytes("relu_pair_collapse.onnx", bytes).expect("loads");
    model
        .network
        .layers
        .iter()
        .filter(|layer| layer.layer_type == ny_core::LayerType::ReLU)
        .count()
}

fn evaluate(bytes: &[u8], x: [f32; 2]) -> Vec<f32> {
    let model = load_onnx_bytes("relu_pair_collapse.onnx", bytes).expect("loads");
    let net = model.to_propagate_network().expect("converts");
    let point = arr2(&[x]).into_dyn();
    let out = net
        .propagate_ibp(&ny_tensor::BoundedTensor::new(point.clone(), point).expect("point"))
        .expect("propagates");
    out.lower().iter().copied().collect()
}

const POINTS: [[f32; 2]; 6] = [
    [0.0, 0.0],
    [1.0, -1.0],
    [-1.0, 1.0],
    [0.3, 0.7],
    [-0.8, -0.4],
    [2.0, 0.5],
];

/// The exact split collapses (the hidden ReLU disappears) and the loaded
/// network computes the AUTHORED function at every point, on both sides of
/// every split neuron's kink.
#[test]
fn an_exact_relu_split_collapses_to_the_same_function() {
    let (a, ab, b, c) = split_parts();
    let bytes = split_model(&a, &ab, &b, &c, 1.0);
    assert_eq!(
        relu_layers(&bytes),
        1,
        "the split's hidden ReLU must collapse"
    );
    for x in POINTS {
        let got = evaluate(&bytes, x);
        let want = authored(&a, &ab, &b, &c, 1.0, x);
        for (g, w) in got.iter().zip(&want) {
            assert!(
                (f64::from(*g) - w).abs() <= 1e-5,
                "x={x:?} got={got:?} want={want:?}"
            );
        }
    }
}

/// A pass-through row is exact only under the following ReLU: without it the
/// chain must NOT collapse (the authored model below ends at `u`).
#[test]
fn pass_through_rows_need_the_following_relu() {
    let (a, ab, b, c) = split_parts();
    let bytes = model_bytes(
        "relu_pair_collapse_no_tail_relu",
        13,
        vec![
            gemm("g1", &["input", "w1", "b1"], "t", 1.0),
            node("r1", "Relu", &["t"], &["h"], Vec::new()),
            gemm("g2", &["h", "w2", "b2"], "output", 1.0),
        ],
        vec![
            f32_initializer("w1", &[5, 2], &a),
            f32_initializer("b1", &[5], &ab),
            f32_initializer("w2", &[3, 5], &b),
            f32_initializer("b2", &[3], &c),
        ],
        vec![value_info("input", &[1, 2])],
        vec![value_info("output", &[1, 3])],
    );
    assert_eq!(relu_layers(&bytes), 1, "the hidden ReLU must stay");
    for x in POINTS {
        let got = evaluate(&bytes, x);
        // authored: u = B relu(A x + a) + c, no tail ReLU
        let h: Vec<f64> = (0..5)
            .map(|j| {
                (f64::from(a[2 * j]) * f64::from(x[0])
                    + f64::from(a[2 * j + 1]) * f64::from(x[1])
                    + f64::from(ab[j]))
                .max(0.0)
            })
            .collect();
        for i in 0..3 {
            let want: f64 =
                (0..5).map(|j| f64::from(b[5 * i + j]) * h[j]).sum::<f64>() + f64::from(c[i]);
            assert!((f64::from(got[i]) - want).abs() <= 1e-5, "x={x:?} i={i}");
        }
    }
}

/// Every near miss declines the rewrite (both ReLUs stay) and the loaded
/// network is still the authored function.
#[test]
fn every_near_miss_declines_and_keeps_the_authored_function() {
    let (a, ab, b, c) = split_parts();
    // one bit off in a negated row
    let mut a_bit = a.clone();
    a_bit[2] = f32::from_bits(a_bit[2].to_bits() ^ 1);
    // a pass-through row with a nonzero merge bias is NOT ReLU(t + c)
    let mut c_pass = c.clone();
    c_pass[1] = 0.5;
    // a merge weight that is not exactly +-1
    let mut b_scaled = b.clone();
    b_scaled[3] = 0.999;
    let cases: [(&str, Vec<f32>, Vec<f32>, Vec<f32>, f32); 4] = [
        ("bit-off negation", a_bit, b.clone(), c.clone(), 1.0),
        ("pass-through bias", a.clone(), b.clone(), c_pass, 1.0),
        ("non-unit merge", a.clone(), b_scaled, c.clone(), 1.0),
        ("gemm alpha", a, b, c, 2.0),
    ];
    for (why, a, b, c, alpha2) in cases {
        let bytes = split_model(&a, &ab, &b, &c, alpha2);
        assert_eq!(relu_layers(&bytes), 2, "{why}: must decline the collapse");
        if alpha2 != 1.0 {
            // ny's network conversion refuses non-unit Gemm alpha outright;
            // the collapse declining is the whole claim for this case.
            continue;
        }
        for x in POINTS {
            let got = evaluate(&bytes, x);
            let want = authored(&a, &ab, &b, &c, alpha2, x);
            for (g, w) in got.iter().zip(&want) {
                assert!(
                    (f64::from(*g) - w).abs() <= 1e-5,
                    "{why}: x={x:?} got={got:?} want={want:?}"
                );
            }
        }
    }
}
