// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! `ToDtype F32 -> I32/I64`: the guarded round-toward-zero `Trunc` lowering,
//! end to end through `GraphModel::build_graph_network` and IBP.
//!
//! The bridge used to refuse this cast outright even though ny-onnx lowers the
//! IDENTICAL ONNX `Cast` to a `Trunc` carrying the target code (`to = 6`/`7`),
//! which `ny_build::convert::elementwise` turns into
//! `TruncLayer::for_int32_cast()` / `for_int64_cast()`: round toward zero, plus
//! a proof obligation that the whole pre-activation interval is finite and
//! inside the destination integer domain.
//!
//! What these tests pin down, in the order the soundness argument runs:
//!
//! 1. **the relaxation is a valid enclosure** — the emitted bounds contain
//!    `x as i64 as f32` for sampled concrete points across the input box,
//!    including the sign-asymmetric cases where `trunc` and `floor` disagree;
//! 2. **it is TIGHTER than the vacuous alternative, and no looser than it** —
//!    the same graph translated with a deliberately vacuous carrier
//!    (`TraceOp::Custom`) in place of the cast yields `[-inf, +inf]`, so the
//!    new arm strictly improves on the only other sound option while still
//!    enclosing every sampled evaluation;
//! 3. **the proof obligation is live** — an input box that leaves the
//!    destination integer domain fails CLOSED at propagation (a
//!    `SoundnessRefusal`) instead of silently picking Rust's `trunc` where
//!    ONNX leaves the value undefined and ORT wraps around;
//! 4. **the domain endpoint is exactly the producer's** — nn's own
//!    `f32_to_i64` accepts `[-2^63, next_down(2^63)]` and errors outside it,
//!    which is the same closed set the guarded layer admits.

use ndarray::{ArrayD, IxDyn};
use ny_build::GraphNetworkOptions;
use ny_core::{LayerType, NyError};
use ny_tensor::BoundedTensor;
use ny_trace_bridge::schema::{ComputationGraph, DType, NodeId, TraceNode, TraceOp};
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

/// `Input -> ToDtype(target) -> Relu`, with the cast declaring the target as
/// its own output dtype exactly as a well-formed trace does.
///
/// The trailing `Relu` keeps the cast off the graph output, where the
/// carve-out deliberately does not apply (ny's verifier API declares FLOAT32
/// outputs, and ny-onnx refuses an integer `Cast` on an authored output).
/// `Relu` is also convenient for reading the cast's own bounds back out: it is
/// the identity on the non-negative half and clamps the negative half to 0, so
/// the tests inspect the `Trunc` layer's bounds directly where sign matters.
fn cast_graph(target: DType, shape: &[usize]) -> ComputationGraph {
    let mut cast = node(
        1,
        "cast",
        TraceOp::ToDtype {
            target_dtype: target,
        },
        &[0],
        shape,
    );
    cast.output_dtype = target;
    ComputationGraph::from_nodes(vec![
        node(0, "x", TraceOp::Input, &[], shape),
        cast,
        node(2, "use", TraceOp::Relu, &[1], shape),
    ])
}

/// The same graph shape with a deliberately opaque node standing in for the
/// cast: the ONLY other sound treatment, and the one the tightness comparison
/// is made against.
fn opaque_graph(shape: &[usize]) -> ComputationGraph {
    ComputationGraph::from_nodes(vec![
        node(0, "x", TraceOp::Input, &[], shape),
        node(
            1,
            "cast",
            TraceOp::Custom {
                name: "opaque_cast".into(),
            },
            &[0],
            shape,
        ),
        node(2, "use", TraceOp::Relu, &[1], shape),
    ])
}

fn box_of(lower: &[f32], upper: &[f32]) -> BoundedTensor {
    let shape = IxDyn(&[lower.len()]);
    BoundedTensor::new(
        ArrayD::from_shape_vec(shape.clone(), lower.to_vec()).expect("lower shape"),
        ArrayD::from_shape_vec(shape, upper.to_vec()).expect("upper shape"),
    )
    .expect("ordered input box")
}

/// Bounds at the graph's OUTPUT after propagating `input`.
fn propagate(graph: &ComputationGraph, input: &BoundedTensor) -> ny_core::Result<BoundedTensor> {
    let model = translate(graph).expect("graph translates");
    let network = model
        .build_graph_network(GraphNetworkOptions::default())
        .expect("GraphModel builds a graph network");
    network.propagate_ibp(input)
}

/// nn's `f32_to_i64` semantics, mirrored: round toward zero, then back to the
/// f32 the emitted graph carries. This is the concrete evaluation the emitted
/// interval must enclose.
fn concrete_cast(x: f32) -> f32 {
    (x as i64) as f32
}

/// (1) + (2): the emitted `Trunc` bounds ENCLOSE every sampled concrete
/// evaluation, and are strictly tighter than the vacuous carrier the bridge
/// would otherwise be limited to.
#[test]
fn guarded_trunc_encloses_sampled_evaluations_and_beats_the_vacuous_alternative() {
    for target in [DType::I32, DType::I64] {
        // Deliberately sign-asymmetric: on [-2.5, -1.5] `trunc` gives
        // [-2, -1] while `floor` would give [-3, -2] — a lowering that
        // confused the two would fail this box.
        let lower = [-2.5_f32, -0.75, 0.0, 1.25];
        let upper = [-1.5_f32, 0.75, 3.5, 1.75];
        let input = box_of(&lower, &upper);

        let graph = cast_graph(target, &[4]);
        let model = translate(&graph).unwrap_or_else(|e| panic!("{target:?} translates: {e:?}"));
        assert_eq!(
            model
                .network
                .layers
                .iter()
                .filter(|l| l.layer_type == LayerType::Trunc)
                .count(),
            1,
            "{target:?}: the cast lowers to exactly one Trunc"
        );

        let network = model
            .build_graph_network(GraphNetworkOptions::default())
            .unwrap_or_else(|e| panic!("{target:?} builds: {e:?}"));
        let out = network
            .propagate_ibp(&input)
            .unwrap_or_else(|e| panic!("{target:?} propagates: {e:?}"));

        // The graph output is Relu(trunc(x)). Sample the box densely and
        // require the certified interval to contain every concrete answer.
        for (index, (&low, &high)) in lower.iter().zip(upper.iter()).enumerate() {
            for step in 0..=64 {
                let x = low + (high - low) * (step as f32 / 64.0);
                let truth = concrete_cast(x).max(0.0);
                let (certified_low, certified_high) = (out.lower()[[index]], out.upper()[[index]]);
                assert!(
                    certified_low <= truth && truth <= certified_high,
                    "{target:?}: element {index}, x = {x}: concrete {truth} escapes certified \
                     [{certified_low}, {certified_high}]"
                );
            }
        }

        // Tightness: the vacuous carrier is the only other sound treatment,
        // and the guarded Trunc must not be as loose as it.
        let vacuous = propagate(&opaque_graph(&[4]), &input).expect("opaque carrier propagates");
        let mut strictly_tighter_somewhere = false;
        for index in 0..4 {
            assert!(
                out.lower()[[index]] >= vacuous.lower()[[index]]
                    && out.upper()[[index]] <= vacuous.upper()[[index]],
                "{target:?}: element {index} must be no LOOSER than the vacuous carrier"
            );
            if out.lower()[[index]] > vacuous.lower()[[index]]
                || out.upper()[[index]] < vacuous.upper()[[index]]
            {
                strictly_tighter_somewhere = true;
            }
        }
        assert!(
            strictly_tighter_somewhere,
            "{target:?}: the whole point of the arm is a finite enclosure where the \
             alternative is [-inf, +inf]"
        );
        assert!(
            out.lower().iter().all(|v| v.is_finite()) && out.upper().iter().all(|v| v.is_finite()),
            "{target:?}: bounds must be finite"
        );
    }
}

/// (1), asserted as an EXACT interval rather than a containment: on a
/// non-negative box the trailing `Relu` is the identity, so the graph output
/// is `trunc(.)` itself and `[trunc(l), trunc(u)]` can be pinned element by
/// element. (The round-toward-zero-versus-`floor` distinction on the negative
/// half is covered by the sampled-enclosure test above, whose box straddles
/// both signs.)
#[test]
fn guarded_trunc_is_exact_on_a_non_negative_box() {
    let input = box_of(&[0.0, 1.25, 2.75], &[0.5, 1.75, 3.5]);
    let out = propagate(&cast_graph(DType::I64, &[3]), &input).expect("propagates");
    // Relu(trunc(.)) on a non-negative box is trunc(.).
    let expected_lower = [0.0_f32, 1.0, 2.0];
    let expected_upper = [0.0_f32, 1.0, 3.0];
    for index in 0..3 {
        assert_eq!(
            out.lower()[[index]],
            expected_lower[index],
            "element {index} lower"
        );
        assert_eq!(
            out.upper()[[index]],
            expected_upper[index],
            "element {index} upper"
        );
    }
}

/// (3): the finite/in-range proof obligation is LIVE end to end. An input box
/// that leaves the destination integer domain must fail closed rather than
/// certify Rust's `trunc` where ONNX leaves the answer undefined.
#[test]
fn out_of_domain_input_fails_closed_at_propagation() {
    for (target, exponent) in [(DType::I32, 31_i32), (DType::I64, 63_i32)] {
        let limit = 2.0_f32.powi(exponent);
        // `limit` itself is OUTSIDE the domain (the positive endpoint is
        // exclusive: 2^31 is not an i32 even though `i32::MAX as f32` rounds
        // to it).
        let input = box_of(&[0.0], &[limit]);
        let err = propagate(&cast_graph(target, &[1]), &input)
            .expect_err("an out-of-domain box must be refused, not certified");
        assert!(
            matches!(err, NyError::SoundnessRefusal(ref m) if m.contains("Cast to")),
            "{target:?}: got {err:?}"
        );

        // ... and the largest representable in-range box is still accepted,
        // so the guard is a domain check and not a blanket refusal.
        let in_range = box_of(&[-limit], &[ny_tensor::next_down_f32(limit)]);
        propagate(&cast_graph(target, &[1]), &in_range)
            .unwrap_or_else(|e| panic!("{target:?}: the closed in-range box must pass: {e:?}"));
    }
}

/// (4): the INT64 domain endpoint the guarded layer enforces is exactly the
/// one nn's `f32_to_i64` accepts, so the lowering agrees pointwise with the
/// producer wherever the producer does not itself error.
///
/// nn rejects anything outside `[-9_223_372_000_000_000_000.0,
/// 9_223_371_500_000_000_000.0]`; as f32 those two literals ARE `-2^63` and
/// `next_down(2^63)`, the closed set the layer admits.
#[test]
fn the_int64_domain_matches_nns_accepted_range_exactly() {
    const NN_MIN_F32_FOR_I64: f32 = -9_223_372_000_000_000_000.0;
    const NN_MAX_F32_FOR_I64: f32 = 9_223_371_500_000_000_000.0;
    assert_eq!(NN_MIN_F32_FOR_I64, -(2.0_f32.powi(63)));
    assert_eq!(
        NN_MAX_F32_FOR_I64,
        ny_tensor::next_down_f32(2.0_f32.powi(63))
    );

    // The exact closed set nn accepts propagates; one ulp past either end
    // fails closed.
    let widest = box_of(&[NN_MIN_F32_FOR_I64], &[NN_MAX_F32_FOR_I64]);
    propagate(&cast_graph(DType::I64, &[1]), &widest)
        .expect("nn's whole accepted range must be certifiable");

    let over = box_of(&[0.0], &[2.0_f32.powi(63)]);
    assert!(
        matches!(
            propagate(&cast_graph(DType::I64, &[1]), &over),
            Err(NyError::SoundnessRefusal(_))
        ),
        "one ulp past nn's maximum must fail closed"
    );
}
