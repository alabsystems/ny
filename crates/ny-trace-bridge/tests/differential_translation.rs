// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Differential-translation oracle for the trace bridge.
//!
//! Feed the *same logical network* through the two independent op→`LayerSpec`
//! intake cores NY owns and require the resulting verifier bounds to agree:
//!
//! - **ny-onnx** parses ONNX proto ops (`convert_node_to_layer`) →
//!   `to_graph_network()` → `ny_propagate::GraphNetwork`.
//! - **this bridge** lowers `TraceOp`s (`translate_node`) → `GraphModel` →
//!   `build_graph_network()` → `ny_propagate::GraphNetwork`.
//!
//! ## Scope — what this guards, and what it deliberately does NOT
//!
//! Both paths lower a `LayerSpec` list into a `GraphNetwork` through the SAME
//! downstream `ny_build` graph builder. They are therefore *not* two independent
//! end-to-end translations — the shared `ny_build` stage is common to both and a
//! bug there is invisible here by construction. What IS independent is the FRONT
//! stage: op parsing into a `LayerSpec`. ny-onnx maps ONNX ops; the bridge maps
//! `TraceOp`s. Both were written separately and can drift. This harness pins the
//! two op→`LayerSpec` cores against each other: a bound divergence means the
//! bridge lowered an op's weights, axes, or attributes differently than the
//! independently-written ONNX path — exactly the "axis/attribute drift" the
//! migration roadmap (docs/TRACE_BRIDGE_MIGRATION.md) flags as the risk when
//! porting arms (INC-5..11).
//!
//! The MLP case is additionally anchored to a *hand-derived* IBP interval, so
//! agreement can never be vacuous (two identically-wrong translations both
//! passing): the shared-anchor asserts the ny-onnx path hits the exact interval
//! computed by hand below, and the differential then ties the bridge to it.
//!
//! ## Extending
//!
//! As each op family is ported into `translate_node`, add a fixture that builds
//! the family both ways (ONNX + `TraceOp` schema) and calls
//! [`assert_networks_agree`]. Bridge fixtures use unbatched `[C, …]` shapes (the
//! trace convention) while ONNX fixtures use batched `[N, C, …]`; the comparison
//! is on the *flattened* output vector, which is shape-invariant, so the two
//! natural conventions compare apples-to-apples.

use ndarray::{ArrayD, IxDyn};
use ny_onnx::onnx_proto;
use ny_propagate::GraphNetwork;
use ny_tensor::BoundedTensor;
use ny_trace_bridge::schema::{ComputationGraph, DType, NodeId, TraceNode, TraceOp, WeightPayload};
use ny_trace_bridge::translate::translate as lower_trace;
use prost::Message;

// ------------------------------------------------------------------------
// ONNX proto builders (mirror crates/ny-onnx/src/bin/gen_test_fixtures.rs so the
// synthesized bytes are byte-compatible with the checked-in fixtures).
// ------------------------------------------------------------------------

fn tensor_value_info(name: &str, shape: &[i64]) -> onnx_proto::ValueInfoProto {
    let dim = shape
        .iter()
        .map(|d| onnx_proto::tensor_shape_proto::Dimension {
            value: Some(onnx_proto::tensor_shape_proto::dimension::Value::DimValue(
                *d,
            )),
        })
        .collect();
    onnx_proto::ValueInfoProto {
        name: name.to_string(),
        r#type: Some(onnx_proto::TypeProto {
            tensor_type: Some(onnx_proto::TensorTypeProto {
                elem_type: 1, // f32
                shape: Some(onnx_proto::TensorShapeProto { dim }),
            }),
        }),
    }
}

fn tensor_f32(name: &str, shape: &[i64], data: &[f32]) -> onnx_proto::TensorProto {
    assert_eq!(
        shape.iter().product::<i64>() as usize,
        data.len(),
        "initializer {name}: shape/data length mismatch"
    );
    onnx_proto::TensorProto {
        dims: shape.to_vec(),
        data_type: 1, // f32
        name: name.to_string(),
        float_data: data.to_vec(),
        ..Default::default()
    }
}

fn attr_int(name: &str, value: i64) -> onnx_proto::AttributeProto {
    onnx_proto::AttributeProto {
        name: name.to_string(),
        i: value,
        r#type: onnx_proto::attribute_type::INT,
        ..Default::default()
    }
}

fn attr_ints(name: &str, values: &[i64]) -> onnx_proto::AttributeProto {
    onnx_proto::AttributeProto {
        name: name.to_string(),
        r#type: onnx_proto::attribute_type::INTS,
        ints: values.to_vec(),
        ..Default::default()
    }
}

fn onnx_node(
    name: &str,
    op_type: &str,
    inputs: &[&str],
    outputs: &[&str],
    attrs: Vec<onnx_proto::AttributeProto>,
) -> onnx_proto::NodeProto {
    onnx_proto::NodeProto {
        input: inputs.iter().map(|s| (*s).to_string()).collect(),
        output: outputs.iter().map(|s| (*s).to_string()).collect(),
        name: name.to_string(),
        op_type: op_type.to_string(),
        domain: String::new(),
        attribute: attrs,
    }
}

/// Encode a `GraphProto` into ONNX bytes, load it through ny-onnx, and lower to
/// a `GraphNetwork` (the ONNX intake core).
fn onnx_graph_network(graph: onnx_proto::GraphProto) -> GraphNetwork {
    let model = onnx_proto::ModelProto {
        ir_version: 9,
        opset_import: vec![onnx_proto::OperatorSetIdProto {
            domain: String::new(),
            version: 17,
        }],
        graph: Some(graph),
        ..Default::default()
    };
    let mut buf = Vec::new();
    model.encode(&mut buf).expect("encode synthesized ONNX");
    ny_onnx::load_onnx_bytes("differential", &buf)
        .expect("ny-onnx loads the synthesized model")
        .to_graph_network()
        .expect("ny-onnx lowers to a GraphNetwork")
}

// ------------------------------------------------------------------------
// Trace-schema builder (the bridge intake core).
// ------------------------------------------------------------------------

fn tnode(id: u64, name: &str, op: TraceOp, inputs: &[u64], shape: &[usize]) -> TraceNode {
    TraceNode::new(
        NodeId(id),
        name,
        op,
        inputs.iter().map(|&i| NodeId(i)).collect(),
        shape.to_vec(),
        DType::F32,
    )
}

/// Lower a trace-schema graph to a `GraphNetwork` (the bridge intake core).
fn bridge_graph_network(graph: &ComputationGraph) -> GraphNetwork {
    lower_trace(graph)
        .expect("bridge translates the schema")
        .build_graph_network(ny_build::GraphNetworkOptions::default())
        .expect("bridge GraphModel builds a GraphNetwork")
}

// ------------------------------------------------------------------------
// Bound-propagation comparison harness.
// ------------------------------------------------------------------------

/// Flattened output (lower, upper) bounds under IBP.
fn ibp_bounds(net: &GraphNetwork, input: &BoundedTensor) -> (Vec<f32>, Vec<f32>) {
    let out = net.propagate_ibp(input).expect("IBP propagation succeeds");
    (
        out.lower().iter().copied().collect(),
        out.upper().iter().copied().collect(),
    )
}

/// Flattened output (lower, upper) bounds under CROWN.
fn crown_bounds(net: &GraphNetwork, input: &BoundedTensor) -> (Vec<f32>, Vec<f32>) {
    let out = net
        .propagate_crown(input)
        .expect("CROWN propagation succeeds");
    (
        out.lower().iter().copied().collect(),
        out.upper().iter().copied().collect(),
    )
}

fn assert_close(a: &[f32], b: &[f32], tol: f32, ctx: &str) {
    assert_eq!(
        a.len(),
        b.len(),
        "{ctx}: output length mismatch ({} vs {})",
        a.len(),
        b.len()
    );
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        // Exact-equality short-circuit so *agreeing* infinities pass: (inf - inf)
        // is NaN and `NaN <= tol` is false, which would otherwise false-fail two
        // legitimately-equal ±inf bounds (relevant once SoundButLoose ops like
        // Exp/Div — which can produce infinite bounds — get fixtures here).
        // `partial_cmp == Some(Equal)` is true for inf==inf but None for NaN, so a
        // genuine NaN divergence still fails, and it avoids the float_cmp lint.
        if x.partial_cmp(y) == Some(std::cmp::Ordering::Equal) {
            continue;
        }
        assert!(
            (x - y).abs() <= tol,
            "{ctx}: element {i} differs: {x} vs {y} (tol {tol})"
        );
    }
}

/// Non-panicking IBP-agreement predicate (used by the sensitivity test that a
/// *drifted* translation is actually DETECTED). Returns false on any length or
/// value divergence beyond `tol`.
fn ibp_agree(
    a: &GraphNetwork,
    b: &GraphNetwork,
    ia: &BoundedTensor,
    ib: &BoundedTensor,
    tol: f32,
) -> bool {
    let (lo_a, hi_a) = ibp_bounds(a, ia);
    let (lo_b, hi_b) = ibp_bounds(b, ib);
    if lo_a.len() != lo_b.len() {
        return false;
    }
    lo_a.iter().zip(&lo_b).all(|(x, y)| (x - y).abs() <= tol)
        && hi_a.iter().zip(&hi_b).all(|(x, y)| (x - y).abs() <= tol)
}

/// Assert the ONNX-intake network and the bridge-intake network agree, on both
/// IBP and CROWN, over the given input boxes (one per network, since the two
/// intakes use their natural batched/unbatched shape conventions). Also runs a
/// self-consistency check that CROWN is no looser than IBP — this holds for the
/// feedforward affine/ReLU/Conv nets exercised here, but it is a *tightness*
/// expectation, NOT a soundness property: each method is independently sound and
/// neither dominates the other in general (which is why CROWN-IBP hybrids exist).
fn assert_networks_agree(
    onnx_net: &GraphNetwork,
    bridge_net: &GraphNetwork,
    onnx_input: &BoundedTensor,
    bridge_input: &BoundedTensor,
    tol: f32,
    label: &str,
) {
    let (ibp_lo_a, ibp_hi_a) = ibp_bounds(onnx_net, onnx_input);
    let (ibp_lo_b, ibp_hi_b) = ibp_bounds(bridge_net, bridge_input);
    assert_close(
        &ibp_lo_a,
        &ibp_lo_b,
        tol,
        &format!("{label}: IBP lower A vs B"),
    );
    assert_close(
        &ibp_hi_a,
        &ibp_hi_b,
        tol,
        &format!("{label}: IBP upper A vs B"),
    );

    let (crown_lo_a, crown_hi_a) = crown_bounds(onnx_net, onnx_input);
    let (crown_lo_b, crown_hi_b) = crown_bounds(bridge_net, bridge_input);
    assert_close(
        &crown_lo_a,
        &crown_lo_b,
        tol,
        &format!("{label}: CROWN lower A vs B"),
    );
    assert_close(
        &crown_hi_a,
        &crown_hi_b,
        tol,
        &format!("{label}: CROWN upper A vs B"),
    );

    // CROWN is expected no looser than IBP for these feedforward nets (a cheap
    // self-consistency check that flags a broken propagation path). This is a
    // per-fixture tightness expectation, not a general soundness guarantee.
    for (i, ((clo, chi), (ilo, ihi))) in crown_lo_a
        .iter()
        .zip(&crown_hi_a)
        .zip(ibp_lo_a.iter().zip(&ibp_hi_a))
        .enumerate()
    {
        assert!(
            *clo >= *ilo - tol && *chi <= *ihi + tol,
            "{label}: CROWN not ⊆ IBP at output {i}: crown [{clo}, {chi}] vs ibp [{ilo}, {ihi}]"
        );
    }
}

/// Box `[-1, 1]^n` with the given shape.
fn unit_box(shape: &[usize]) -> BoundedTensor {
    let n: usize = shape.iter().product();
    let lower = ArrayD::from_shape_vec(IxDyn(shape), vec![-1.0_f32; n]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(shape), vec![1.0_f32; n]).unwrap();
    BoundedTensor::new(lower, upper).expect("valid input box")
}

/// A box whose per-element interval varies by flattened (row-major) index, so
/// every input position/channel is distinguishable. Under a *uniform* box the
/// output bounds of an affine op are permutation-invariant, which hides
/// axis/channel/group drift (an equal-everywhere input can't tell a transposed
/// kernel from the original); a structured box makes any reindexing observable.
/// The onnx `[N, C, …]` and bridge `[C, …]` shapes share the same row-major data
/// order (N = 1), so both intakes receive identical per-element intervals.
fn structured_box(shape: &[usize]) -> BoundedTensor {
    let n: usize = shape.iter().product();
    let lower: Vec<f32> = (0..n).map(|k| -1.0 - 0.05 * (k % 11) as f32).collect();
    let upper: Vec<f32> = (0..n).map(|k| 0.5 + 0.07 * (k % 13) as f32).collect();
    let lower = ArrayD::from_shape_vec(IxDyn(shape), lower).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(shape), upper).unwrap();
    BoundedTensor::new(lower, upper).expect("valid structured input box")
}

/// Deterministic, asymmetric values in `[-1, 1]` (period 17, so no short axis is
/// symmetric) — a kernel/channel/spatial permutation of these changes the conv
/// output, which is what makes the conv differential able to see axis drift.
fn varied(n: usize) -> Vec<f32> {
    (0..n)
        .map(|k| (((k as i64 * 7 + 3).rem_euclid(17)) - 8) as f32 * 0.125)
        .collect()
}

/// Distinct per-channel biases so a bias mis-order is also observable.
fn varied_bias(n: usize) -> Vec<f32> {
    (0..n).map(|k| 0.1 * k as f32 - 0.15).collect()
}

// ------------------------------------------------------------------------
// Fixtures. ONNX weights mirror gen_test_fixtures.rs; the bridge schema carries
// the identical weight arrays so the two intakes describe the same net.
// ------------------------------------------------------------------------

// simple_mlp: input[.,2] → Gemm(W1[4,2],b1) → ReLU → Gemm(W2[2,4],b2) → out[.,2].
const W1: [f32; 8] = [1.0, 0.5, -1.0, 0.5, 0.5, 1.0, 0.5, -1.0];
const B1: [f32; 4] = [0.1, 0.1, 0.1, 0.1];
const W2: [f32; 8] = [1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0];
const B2: [f32; 2] = [0.0, 0.0];

fn onnx_mlp() -> GraphNetwork {
    let graph = onnx_proto::GraphProto {
        node: vec![
            onnx_node(
                "gemm1",
                "Gemm",
                &["input", "w1", "b1"],
                &["fc1"],
                vec![attr_int("transB", 1)],
            ),
            onnx_node("relu", "Relu", &["fc1"], &["act"], Vec::new()),
            onnx_node(
                "gemm2",
                "Gemm",
                &["act", "w2", "b2"],
                &["output"],
                vec![attr_int("transB", 1)],
            ),
        ],
        name: "simple_mlp".to_string(),
        initializer: vec![
            tensor_f32("w1", &[4, 2], &W1),
            tensor_f32("b1", &[4], &B1),
            tensor_f32("w2", &[2, 4], &W2),
            tensor_f32("b2", &[2], &B2),
        ],
        input: vec![tensor_value_info("input", &[1, 2])],
        output: vec![tensor_value_info("output", &[1, 2])],
        ..Default::default()
    };
    onnx_graph_network(graph)
}

fn bridge_mlp() -> GraphNetwork {
    let graph = ComputationGraph::from_nodes(vec![
        tnode(0, "input", TraceOp::Input, &[], &[2]),
        tnode(
            1,
            "fc1",
            TraceOp::Linear {
                weight: WeightPayload::f32(W1.to_vec(), vec![4, 2]),
                bias: Some(WeightPayload::f32(B1.to_vec(), vec![4])),
            },
            &[0],
            &[4],
        ),
        tnode(2, "act", TraceOp::Relu, &[1], &[4]),
        tnode(
            3,
            "fc2",
            TraceOp::Linear {
                weight: WeightPayload::f32(W2.to_vec(), vec![2, 4]),
                bias: Some(WeightPayload::f32(B2.to_vec(), vec![2])),
            },
            &[2],
            &[2],
        ),
    ]);
    bridge_graph_network(&graph)
}

#[test]
fn mlp_matches_ground_truth_and_agrees_across_intakes() {
    let onnx_net = onnx_mlp();
    let bridge_net = bridge_mlp();
    let onnx_input = unit_box(&[1, 2]);
    let bridge_input = unit_box(&[2]);

    // --- Ground-truth IBP anchor (hand-derived over x ∈ [-1, 1]^2) ---
    // Each fc1 row has |w| sum 1.5 and bias 0.1 → pre-ReLU ∈ [-1.4, 1.6];
    // ReLU → [0, 1.6] on all four hidden units.
    // out0 = Σ hidden (all +1) → [0, 6.4].
    // out1 = -h0 + h1 - h2 + h3 → [-3.2, 3.2].
    let (ibp_lo, ibp_hi) = ibp_bounds(&onnx_net, &onnx_input);
    assert_close(
        &ibp_lo,
        &[0.0, -3.2],
        1e-3,
        "ONNX IBP lower vs hand-derived",
    );
    assert_close(&ibp_hi, &[6.4, 3.2], 1e-3, "ONNX IBP upper vs hand-derived");

    // --- Differential: bridge must reproduce the same net on IBP and CROWN ---
    assert_networks_agree(
        &onnx_net,
        &bridge_net,
        &onnx_input,
        &bridge_input,
        1e-4,
        "mlp",
    );
}

fn onnx_linear_relu() -> GraphNetwork {
    let graph = onnx_proto::GraphProto {
        node: vec![
            onnx_node(
                "gemm",
                "Gemm",
                &["input", "w", "b"],
                &["lin"],
                vec![attr_int("transB", 1)],
            ),
            onnx_node("relu", "Relu", &["lin"], &["output"], Vec::new()),
        ],
        name: "linear_relu".to_string(),
        initializer: vec![
            tensor_f32("w", &[3, 2], &[1.0, 2.0, 3.0, -1.0, -2.0, 1.0]),
            tensor_f32("b", &[3], &[0.5, -0.5, 1.0]),
        ],
        input: vec![tensor_value_info("input", &[1, 2])],
        output: vec![tensor_value_info("output", &[1, 3])],
        ..Default::default()
    };
    onnx_graph_network(graph)
}

fn bridge_linear_relu() -> GraphNetwork {
    let graph = ComputationGraph::from_nodes(vec![
        tnode(0, "input", TraceOp::Input, &[], &[2]),
        tnode(
            1,
            "lin",
            TraceOp::Linear {
                weight: WeightPayload::f32(vec![1.0, 2.0, 3.0, -1.0, -2.0, 1.0], vec![3, 2]),
                bias: Some(WeightPayload::f32(vec![0.5, -0.5, 1.0], vec![3])),
            },
            &[0],
            &[3],
        ),
        tnode(2, "relu", TraceOp::Relu, &[1], &[3]),
    ]);
    bridge_graph_network(&graph)
}

#[test]
fn linear_relu_agrees_across_intakes() {
    assert_networks_agree(
        &onnx_linear_relu(),
        &bridge_linear_relu(),
        &unit_box(&[1, 2]),
        &unit_box(&[2]),
        1e-4,
        "linear_relu",
    );
}

fn onnx_single_linear() -> GraphNetwork {
    let graph = onnx_proto::GraphProto {
        node: vec![onnx_node(
            "gemm",
            "Gemm",
            &["input", "w", "b"],
            &["output"],
            vec![attr_int("transB", 1)],
        )],
        name: "single_linear".to_string(),
        initializer: vec![
            tensor_f32("w", &[3, 2], &[1.0, 2.0, 3.0, -1.0, -2.0, 1.0]),
            tensor_f32("b", &[3], &[0.5, -0.5, 1.0]),
        ],
        input: vec![tensor_value_info("input", &[1, 2])],
        output: vec![tensor_value_info("output", &[1, 3])],
        ..Default::default()
    };
    onnx_graph_network(graph)
}

fn bridge_single_linear() -> GraphNetwork {
    let graph = ComputationGraph::from_nodes(vec![
        tnode(0, "input", TraceOp::Input, &[], &[2]),
        tnode(
            1,
            "lin",
            TraceOp::Linear {
                weight: WeightPayload::f32(vec![1.0, 2.0, 3.0, -1.0, -2.0, 1.0], vec![3, 2]),
                bias: Some(WeightPayload::f32(vec![0.5, -0.5, 1.0], vec![3])),
            },
            &[0],
            &[3],
        ),
    ]);
    bridge_graph_network(&graph)
}

#[test]
fn single_linear_agrees_across_intakes() {
    assert_networks_agree(
        &onnx_single_linear(),
        &bridge_single_linear(),
        &unit_box(&[1, 2]),
        &unit_box(&[2]),
        1e-4,
        "single_linear",
    );
}

// Conv2d — the classic axis/attribute-drift target. Deliberately NON-TRIVIAL so
// a real op→LayerSpec drift actually diverges: asymmetric pad (pH≠pW), asymmetric
// stride (sH≠sW), non-square input (H≠W), asymmetric kernel (kH≠kW), and >1
// in/out channels, propagated over a structured (non-uniform) box. Under these
// params a pad-H/W swap or stride-H/W swap changes the OUTPUT SHAPE (length
// mismatch → fail), and a kernel spatial/channel transpose or bias mis-order
// changes the numeric bounds (value mismatch → fail). A trivial symmetric conv
// (all pads 0 / strides 1 / single channel / uniform box) would pass under ALL
// of those drifts, which is why this fixture avoids every one of them.
// input[.,2,5,4] → Conv(K[3,2,3,2], pad[1,0], stride[2,1]) → out[.,3,3,3].
fn onnx_conv2d() -> GraphNetwork {
    let graph = onnx_proto::GraphProto {
        node: vec![onnx_node(
            "conv",
            "Conv",
            &["input", "kernel", "bias"],
            &["output"],
            vec![
                attr_ints("kernel_shape", &[3, 2]),
                attr_ints("pads", &[1, 0, 1, 0]), // [top, left, bottom, right] = pH,pW,pH,pW
                attr_ints("strides", &[2, 1]),
                attr_ints("dilations", &[1, 1]),
                attr_int("group", 1),
            ],
        )],
        name: "asym_conv2d".to_string(),
        initializer: vec![
            tensor_f32("kernel", &[3, 2, 3, 2], &varied(3 * 2 * 3 * 2)),
            tensor_f32("bias", &[3], &varied_bias(3)),
        ],
        input: vec![tensor_value_info("input", &[1, 2, 5, 4])],
        output: vec![tensor_value_info("output", &[1, 3, 3, 3])],
        ..Default::default()
    };
    onnx_graph_network(graph)
}

fn bridge_conv2d() -> GraphNetwork {
    let graph = ComputationGraph::from_nodes(vec![
        tnode(0, "input", TraceOp::Input, &[], &[2, 5, 4]),
        tnode(
            1,
            "conv",
            TraceOp::Conv2d {
                weight: WeightPayload::f32(varied(3 * 2 * 3 * 2), vec![3, 2, 3, 2]),
                bias: Some(WeightPayload::f32(varied_bias(3), vec![3])),
                padding: [1, 0],
                stride: [2, 1],
                dilation: [1, 1],
                groups: 1,
            },
            &[0],
            &[3, 3, 3],
        ),
    ]);
    bridge_graph_network(&graph)
}

#[test]
fn conv2d_asymmetric_agrees_across_intakes() {
    assert_networks_agree(
        &onnx_conv2d(),
        &bridge_conv2d(),
        &structured_box(&[1, 2, 5, 4]),
        &structured_box(&[2, 5, 4]),
        1e-4,
        "conv2d_asymmetric",
    );
}

// Grouped Conv2d: groups=2 routes input channel c to the kernel of group c. With
// a structured box (channels have distinct intervals), a group MIS-route feeds
// the wrong channel's box to a kernel and changes the bounds — invisible under a
// uniform box or with groups=1. input[.,2,4,4] → Conv(K[2,1,2,2], groups=2) → out[.,2,3,3].
fn onnx_conv2d_grouped() -> GraphNetwork {
    let graph = onnx_proto::GraphProto {
        node: vec![onnx_node(
            "conv",
            "Conv",
            &["input", "kernel", "bias"],
            &["output"],
            vec![
                attr_ints("kernel_shape", &[2, 2]),
                attr_ints("pads", &[0, 0, 0, 0]),
                attr_ints("strides", &[1, 1]),
                attr_ints("dilations", &[1, 1]),
                attr_int("group", 2),
            ],
        )],
        name: "grouped_conv2d".to_string(),
        initializer: vec![
            // weight [C_out, C_in/groups, kH, kW] = [2, 1, 2, 2].
            tensor_f32("kernel", &[2, 1, 2, 2], &varied(8)),
            tensor_f32("bias", &[2], &varied_bias(2)),
        ],
        input: vec![tensor_value_info("input", &[1, 2, 4, 4])],
        output: vec![tensor_value_info("output", &[1, 2, 3, 3])],
        ..Default::default()
    };
    onnx_graph_network(graph)
}

fn bridge_conv2d_grouped() -> GraphNetwork {
    let graph = ComputationGraph::from_nodes(vec![
        tnode(0, "input", TraceOp::Input, &[], &[2, 4, 4]),
        tnode(
            1,
            "conv",
            TraceOp::Conv2d {
                weight: WeightPayload::f32(varied(8), vec![2, 1, 2, 2]),
                bias: Some(WeightPayload::f32(varied_bias(2), vec![2])),
                padding: [0, 0],
                stride: [1, 1],
                dilation: [1, 1],
                groups: 2,
            },
            &[0],
            &[2, 3, 3],
        ),
    ]);
    bridge_graph_network(&graph)
}

#[test]
fn conv2d_grouped_agrees_across_intakes() {
    assert_networks_agree(
        &onnx_conv2d_grouped(),
        &bridge_conv2d_grouped(),
        &structured_box(&[1, 2, 4, 4]),
        &structured_box(&[2, 4, 4]),
        1e-4,
        "conv2d_grouped",
    );
}

/// Sensitivity / non-vacuity proof: the asymmetric conv fixture has TEETH. A
/// bridge conv with a deliberately swapped stride ([1,2] instead of the correct
/// [2,1]) must DISAGREE with the correct ONNX conv — otherwise the differential
/// could not detect the exact axis/attribute drift it exists to catch. (Under a
/// trivial symmetric fixture this assertion would fail, because the drift would
/// be invisible — which is precisely why the fixtures above are non-trivial.)
#[test]
fn oracle_detects_conv_stride_drift() {
    let correct_onnx = onnx_conv2d(); // stride [2, 1]
    let drifted_bridge = {
        let graph = ComputationGraph::from_nodes(vec![
            tnode(0, "input", TraceOp::Input, &[], &[2, 5, 4]),
            tnode(
                1,
                "conv",
                TraceOp::Conv2d {
                    weight: WeightPayload::f32(varied(3 * 2 * 3 * 2), vec![3, 2, 3, 2]),
                    bias: Some(WeightPayload::f32(varied_bias(3), vec![3])),
                    padding: [1, 0],
                    stride: [1, 2], // DRIFT: axes swapped vs the correct [2, 1]
                    dilation: [1, 1],
                    groups: 1,
                },
                &[0],
                &[3, 5, 2], // stride swap changes the output shape too
            ),
        ]);
        bridge_graph_network(&graph)
    };
    assert!(
        !ibp_agree(
            &correct_onnx,
            &drifted_bridge,
            &structured_box(&[1, 2, 5, 4]),
            &structured_box(&[2, 5, 4]),
            1e-4,
        ),
        "oracle is vacuous: it did not detect a swapped-stride conv drift"
    );
}
