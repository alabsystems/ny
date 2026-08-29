// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Integer-index Embedding: the dtype-gate carve-out and the soundness of the
//! midpoint-Linear + EpsilonExpand surrogate.
//!
//! The bridge's graph-wide F32 gate refuses every reachable non-F32 node,
//! with one narrow carve-out: an integer-typed `Input`/`Constant` consumed
//! ONLY as the index operand of `Embedding` nodes (and not itself a graph
//! output). The Embedding surrogate is index-value-independent — its output
//! interval is the per-dimension vocabulary hull — so the index value never
//! reaches the emitted graph's arithmetic and its integer dtype cannot affect
//! soundness. These tests pin down both sides:
//!
//! - the carve-out admits exactly the value-independent pattern (integer
//!   index → Embedding) and nothing else — any other consumer, a non-leaf
//!   op, a float dtype, or an index that is also a graph output keeps the
//!   verbatim refusal;
//! - the translated surrogate's IBP output bounds CONTAIN every actual
//!   embedding row, including under adversarial float magnitudes where a
//!   naively rounded half-spread would round INWARD past a row.

use ndarray::{ArrayD, IxDyn};
use ny_build::{GraphModel, GraphNetworkOptions};
use ny_core::{LayerType, NyError};
use ny_tensor::BoundedTensor;
use ny_trace_bridge::schema::{ComputationGraph, DType, NodeId, TraceNode, TraceOp, WeightPayload};
use ny_trace_bridge::translate::translate;

fn node(id: u64, name: &str, op: TraceOp, inputs: &[u64], shape: &[usize]) -> TraceNode {
    TraceNode::new(
        NodeId(id),
        name,
        op,
        inputs.iter().map(|&i| NodeId(i)).collect(),
        shape.to_vec(),
        DType::F32,
    )
}

fn count(model: &GraphModel, lt: &LayerType) -> usize {
    model
        .network
        .layers
        .iter()
        .filter(|l| &l.layer_type == lt)
        .count()
}

/// Adversarial 4x3 vocabulary table.
///
/// - dim 0 spans [-2^26, 1.0]: `f32::midpoint` rounds the exact midpoint
///   -33554431.5 to -2^25, and the exact gap `hi - mid` = 2^25 + 1 is NOT
///   f32-representable — a round-to-nearest half-spread would land at 2^25
///   and exclude the actual row value 1.0 (midpoint + 2^25 = 0 < 1).
/// - dim 1 is all-negative with an exactly representable spread.
/// - dim 2 is zero-spread (every row 0.25), which must pin eps to a true 0.
fn adversarial_table() -> (Vec<f32>, [(f32, f32); 3]) {
    let weight = vec![
        -67_108_864.0,
        -3.5,
        0.25, // row 0
        1.0,
        -1.25,
        0.25, // row 1
        -3.0,
        -2.0,
        0.25, // row 2
        0.5,
        -3.5,
        0.25, // row 3
    ];
    let hulls = [(-67_108_864.0, 1.0), (-3.5, -1.25), (0.25, 0.25)];
    (weight, hulls)
}

/// Index-input → Embedding graph over the adversarial table: [1, 2] indices
/// (T = 2 positions) → [1, 2, 3] embeddings.
fn embedding_graph(index_dtype: DType) -> ComputationGraph {
    let (weight, _) = adversarial_table();
    let mut index = node(0, "idx", TraceOp::Input, &[], &[1, 2]);
    index.output_dtype = index_dtype;
    ComputationGraph::from_nodes(vec![
        index,
        node(
            1,
            "emb",
            TraceOp::Embedding {
                weight: WeightPayload::f32(weight, vec![4, 3]),
            },
            &[0],
            &[1, 2, 3],
        ),
    ])
}

/// Every integer index dtype translates through the carve-out, into the
/// Linear + EpsilonExpand + Reshape surrogate, and builds a GraphNetwork.
#[test]
fn integer_index_embedding_translates_and_builds() {
    for dtype in [DType::U32, DType::I64, DType::I32, DType::U8] {
        let model = translate(&embedding_graph(dtype))
            .unwrap_or_else(|e| panic!("{dtype:?}-index Embedding must translate: {e}"));
        assert_eq!(count(&model, &LayerType::Linear), 1, "{dtype:?}");
        assert_eq!(count(&model, &LayerType::EpsilonExpand), 1, "{dtype:?}");
        assert_eq!(count(&model, &LayerType::Reshape), 1, "{dtype:?}");
        model
            .build_graph_network(GraphNetworkOptions::default())
            .unwrap_or_else(|e| panic!("{dtype:?}-index Embedding must build: {e}"));
    }
}

/// The F32 index path (no carve-out needed) keeps translating identically.
#[test]
fn f32_index_embedding_still_translates() {
    let model = translate(&embedding_graph(DType::F32)).expect("F32-index Embedding translates");
    assert_eq!(count(&model, &LayerType::Linear), 1);
    assert_eq!(count(&model, &LayerType::EpsilonExpand), 1);
    assert_eq!(count(&model, &LayerType::Reshape), 1);
    model
        .build_graph_network(GraphNetworkOptions::default())
        .expect("F32-index Embedding builds");
}

/// SOUNDNESS: the stored midpoint/epsilon pairs satisfy the containment
/// inequalities in exact (f64) arithmetic, and the end-to-end IBP output
/// interval contains every actual embedding row — per position, per
/// dimension, for every vocabulary row. The zero-spread dim stays pinned
/// exactly (eps = 0 must not widen at all).
#[test]
fn translated_bounds_contain_every_embedding_row() {
    let (weight, hulls) = adversarial_table();
    let model = translate(&embedding_graph(DType::U32)).expect("U32-index Embedding translates");

    // The bias/epsilon weights are tiled T=2 times over D=3 dims.
    let bias = model
        .weights
        .get("layer0_trace_1_bias")
        .expect("midpoint bias stored");
    let eps = model
        .weights
        .get("layer0_trace_1_epsilons")
        .expect("epsilon radii stored");
    assert_eq!(bias.shape(), &[6]);
    assert_eq!(eps.shape(), &[6]);
    for (d, &(lo, hi)) in hulls.iter().enumerate() {
        let mid = bias[[d]];
        let radius = eps[[d]];
        // Containment arithmetic, checked exactly in f64: the outward-rounded
        // radius must dominate BOTH gaps around the rounded midpoint.
        assert!(
            f64::from(mid) - f64::from(radius) <= f64::from(lo),
            "dim {d}: midpoint {mid} - eps {radius} must reach lo {lo}"
        );
        assert!(
            f64::from(mid) + f64::from(radius) >= f64::from(hi),
            "dim {d}: midpoint {mid} + eps {radius} must reach hi {hi}"
        );
        // Both positions carry the same per-dim values (row-major tiling).
        assert_eq!(bias[[d + 3]], mid, "dim {d}: bias tiling");
        assert_eq!(eps[[d + 3]], radius, "dim {d}: epsilon tiling");
    }
    assert_eq!(eps[[2]], 0.0, "zero-spread dim gets a true 0 radius");
    // The adversarial dim: a round-to-nearest half-spread (2^25) would
    // exclude the row value 1.0; the outward-rounded radius must not.
    assert!(
        f64::from(eps[[0]]) > 33_554_432.0,
        "dim 0 radius {} must exceed the inward-rounded half-spread 2^25",
        eps[[0]]
    );

    // End-to-end IBP: the output box contains every row at every position,
    // regardless of the (arbitrary) index-input bounds.
    let network = model
        .build_graph_network(GraphNetworkOptions::default())
        .expect("Embedding surrogate builds");
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, 2]), 0.0_f32),
        ArrayD::from_elem(IxDyn(&[1, 2]), 3.0_f32),
    )
    .expect("valid index box");
    let out = network.propagate_ibp(&input).expect("IBP propagates");
    assert_eq!(out.lower().shape(), &[1, 2, 3]);
    for t in 0..2 {
        for d in 0..3 {
            for row in 0..4 {
                let val = weight[row * 3 + d];
                let lower = out.lower()[[0, t, d]];
                let upper = out.upper()[[0, t, d]];
                assert!(
                    lower <= val && val <= upper,
                    "SOUNDNESS VIOLATION: row {row} dim {d} value {val} \
                     not in [{lower}, {upper}] at position {t}"
                );
            }
        }
        // eps = 0 adds NO widening on the zero-spread dim: the interval is at
        // most a handful of ulps wide, all of it the midpoint Linear's own
        // directed-rounding/summation-error slack (never the hull, which is
        // zero-width here — dim 0's hull is ~26 orders of magnitude wider).
        let width = f64::from(out.upper()[[0, t, 2]]) - f64::from(out.lower()[[0, t, 2]]);
        assert!(
            (0.0..=16.0 * f64::from(f32::EPSILON) * 0.25).contains(&width),
            "position {t}: zero-spread width {width} exceeds directed-rounding slack"
        );
    }
}

/// The degenerate equal-rank branch (output rank == input rank, no Reshape)
/// carries the same EpsilonExpand widening, un-tiled, and its IBP output
/// still contains every vocabulary row.
#[test]
fn degenerate_equal_rank_embedding_is_widened_and_sound() {
    let (weight, hulls) = adversarial_table();
    let mut index = node(0, "idx", TraceOp::Input, &[], &[2]);
    index.output_dtype = DType::U32;
    let graph = ComputationGraph::from_nodes(vec![
        index,
        node(
            1,
            "emb",
            TraceOp::Embedding {
                weight: WeightPayload::f32(weight.clone(), vec![4, 3]),
            },
            &[0],
            &[3],
        ),
    ]);
    let model = translate(&graph).expect("equal-rank Embedding translates");
    assert_eq!(count(&model, &LayerType::Linear), 1);
    assert_eq!(count(&model, &LayerType::EpsilonExpand), 1);
    assert_eq!(count(&model, &LayerType::Reshape), 0, "no rank change");

    // Un-tiled per-dim radii, same containment arithmetic as the tiled case.
    let bias = model
        .weights
        .get("layer0_trace_1_bias")
        .expect("midpoint bias stored");
    let eps = model
        .weights
        .get("layer0_trace_1_epsilons")
        .expect("epsilon radii stored");
    assert_eq!(bias.shape(), &[3]);
    assert_eq!(eps.shape(), &[3]);
    for (d, &(lo, hi)) in hulls.iter().enumerate() {
        assert!(
            f64::from(bias[[d]]) - f64::from(eps[[d]]) <= f64::from(lo)
                && f64::from(bias[[d]]) + f64::from(eps[[d]]) >= f64::from(hi),
            "dim {d}: [{} - {}, + {}] must cover [{lo}, {hi}]",
            bias[[d]],
            eps[[d]],
            eps[[d]]
        );
    }
    assert_eq!(eps[[2]], 0.0, "zero-spread dim gets a true 0 radius");

    let network = model
        .build_graph_network(GraphNetworkOptions::default())
        .expect("equal-rank Embedding surrogate builds");
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[2]), 0.0_f32),
        ArrayD::from_elem(IxDyn(&[2]), 3.0_f32),
    )
    .expect("valid index box");
    let out = network.propagate_ibp(&input).expect("IBP propagates");
    assert_eq!(out.lower().shape(), &[3]);
    for d in 0..3 {
        for row in 0..4 {
            let val = weight[row * 3 + d];
            let lower = out.lower()[[d]];
            let upper = out.upper()[[d]];
            assert!(
                lower <= val && val <= upper,
                "SOUNDNESS VIOLATION: row {row} dim {d} value {val} \
                 not in [{lower}, {upper}]"
            );
        }
    }
}

fn assert_f32_gate_refusal(err: &NyError, what: &str) {
    assert!(
        matches!(err, NyError::UnsupportedOp(m)
            if m.contains("output dtype") && m.contains("only soundly supports F32")),
        "{what}: expected the verbatim F32 gate refusal, got {err:?}"
    );
}

/// An integer input consumed by anything other than an Embedding index
/// operand keeps the verbatim refusal (the value would reach arithmetic).
#[test]
fn integer_input_feeding_non_embedding_is_refused() {
    let mut index = node(0, "idx", TraceOp::Input, &[], &[1, 2]);
    index.output_dtype = DType::U32;
    let graph =
        ComputationGraph::from_nodes(vec![index, node(1, "sq", TraceOp::Mul, &[0, 0], &[1, 2])]);
    let err = translate(&graph).expect_err("U32 input feeding Mul must be refused");
    assert_f32_gate_refusal(&err, "U32 -> Mul");
}

/// An integer input referenced at a NON-first Embedding position is refused:
/// only the index operand has a value-independent lowering.
#[test]
fn integer_input_at_non_index_embedding_position_is_refused() {
    let (weight, _) = adversarial_table();
    let mut index = node(0, "idx", TraceOp::Input, &[], &[1, 2]);
    index.output_dtype = DType::U32;
    let graph = ComputationGraph::from_nodes(vec![
        index,
        node(
            1,
            "emb",
            TraceOp::Embedding {
                weight: WeightPayload::f32(weight, vec![4, 3]),
            },
            &[0, 0],
            &[1, 2, 3],
        ),
    ]);
    let err = translate(&graph).expect_err("non-index Embedding edge must be refused");
    assert_f32_gate_refusal(&err, "U32 at Embedding position 1");
}

/// An integer input that is ALSO a marked graph output is refused: its value
/// would be consumed by the caller, so it is not value-independent.
#[test]
fn integer_input_that_is_graph_output_is_refused() {
    let mut graph = embedding_graph(DType::U32);
    graph.output_nodes.push(NodeId(0));
    let err = translate(&graph).expect_err("U32 input marked as output must be refused");
    assert_f32_gate_refusal(&err, "U32 output");
}

/// A bare integer input that is the graph's fallback output (no marked
/// outputs, last node) is refused the same way.
#[test]
fn integer_input_as_fallback_output_is_refused() {
    let mut index = node(0, "idx", TraceOp::Input, &[], &[1, 2]);
    index.output_dtype = DType::U32;
    let graph = ComputationGraph {
        nodes: vec![index],
        output_nodes: vec![],
    };
    let err = translate(&graph).expect_err("U32 fallback output must be refused");
    assert_f32_gate_refusal(&err, "U32 fallback output");
}

/// The carve-out is integer-only: a float (or Bool) index dtype keeps the
/// refusal even in the exact Input → Embedding shape.
#[test]
fn non_integer_non_f32_index_stays_refused() {
    for dtype in [DType::F16, DType::Bf16, DType::F64, DType::Bool] {
        let err = translate(&embedding_graph(dtype))
            .expect_err("non-integer non-F32 index must be refused");
        assert_f32_gate_refusal(&err, "float/bool index");
    }
}

/// A U32 ConstantWeight feeding ONLY an IndexSelect/Gather index slot
/// translates: the index values ride the captured f32 weight payload and are
/// resolved at translate/build time, never lowered into the runtime graph
/// (the kokoro sinegen downsample records exactly this shape).
#[test]
fn integer_constant_weight_gather_index_translates() {
    for op in [TraceOp::IndexSelect { dim: 1 }, TraceOp::Gather { dim: 1 }] {
        let mut idx = node(
            1,
            "indices",
            TraceOp::ConstantWeight {
                weight: WeightPayload::f32(vec![0.0, 2.0], vec![2]),
            },
            &[],
            &[2],
        );
        idx.output_dtype = DType::U32;
        let graph = ComputationGraph {
            nodes: vec![
                node(0, "data", TraceOp::Input, &[], &[1, 4]),
                idx,
                node(2, "sel", op, &[0, 1], &[1, 2]),
            ],
            output_nodes: vec![NodeId(2)],
        };
        translate(&graph).expect("constant integer gather index must translate");
    }
}

/// A RUNTIME U32 Input feeding a Gather index slot stays refused — it is not
/// translate-time-resolvable, so the value-independence argument fails.
#[test]
fn integer_runtime_input_gather_index_stays_refused() {
    let mut idx = node(1, "indices", TraceOp::Input, &[], &[2]);
    idx.output_dtype = DType::U32;
    let graph = ComputationGraph {
        nodes: vec![
            node(0, "data", TraceOp::Input, &[], &[1, 4]),
            idx,
            node(2, "sel", TraceOp::Gather { dim: 1 }, &[0, 1], &[1, 2]),
        ],
        output_nodes: vec![NodeId(2)],
    };
    let err = translate(&graph).expect_err("runtime integer gather index must be refused");
    assert_f32_gate_refusal(&err, "runtime U32 gather index");
}

/// A U32 ConstantWeight consumed by a NON-index position (binary Add data
/// operand) stays refused — value-dependence is real there.
#[test]
fn integer_constant_weight_in_value_position_stays_refused() {
    let mut cw = node(
        1,
        "cw",
        TraceOp::ConstantWeight {
            weight: WeightPayload::f32(vec![1.0, 2.0], vec![2]),
        },
        &[],
        &[2],
    );
    cw.output_dtype = DType::U32;
    let graph = ComputationGraph {
        nodes: vec![
            node(0, "x", TraceOp::Input, &[], &[2]),
            cw,
            node(2, "sum", TraceOp::Add, &[0, 1], &[2]),
        ],
        output_nodes: vec![NodeId(2)],
    };
    let err = translate(&graph).expect_err("U32 constant weight as Add operand must be refused");
    assert_f32_gate_refusal(&err, "U32 value-position constant");
}

/// A Compare node stored as U8/Bool passes the gate: its lowering models the
/// exact {0,1} value set with [0,1]-hull IBP regardless of the stored dtype
/// (nn's tracer records comparison masks as U8).
#[test]
fn boolean_mask_compare_translates_regardless_of_carrier_dtype() {
    use ny_trace_bridge::schema::CompareOp;
    for dtype in [DType::U8, DType::Bool, DType::I64] {
        let mut cmp = node(
            1,
            "mask",
            TraceOp::Compare {
                op: CompareOp::Gt,
                value: 0.5,
            },
            &[0],
            &[4],
        );
        cmp.output_dtype = dtype;
        let graph = ComputationGraph {
            nodes: vec![
                node(0, "x", TraceOp::Input, &[], &[4]),
                cmp,
                node(2, "masked", TraceOp::Mul, &[0, 1], &[4]),
            ],
            output_nodes: vec![NodeId(2)],
        };
        translate(&graph)
            .unwrap_or_else(|e| panic!("{dtype:?}-carried compare mask must translate: {e:?}"));
    }
}

/// Float non-F32 compare carriers stay refused — their rounding story is the
/// general one the gate exists for.
#[test]
fn float_non_f32_compare_carrier_stays_refused() {
    use ny_trace_bridge::schema::CompareOp;
    let mut cmp = node(
        1,
        "mask",
        TraceOp::Compare {
            op: CompareOp::Gt,
            value: 0.5,
        },
        &[0],
        &[4],
    );
    cmp.output_dtype = DType::F16;
    let graph = ComputationGraph {
        nodes: vec![
            node(0, "x", TraceOp::Input, &[], &[4]),
            cmp,
            node(2, "masked", TraceOp::Mul, &[0, 1], &[4]),
        ],
        output_nodes: vec![NodeId(2)],
    };
    let err = translate(&graph).expect_err("F16 compare carrier must be refused");
    assert_f32_gate_refusal(&err, "F16 compare carrier");
}
