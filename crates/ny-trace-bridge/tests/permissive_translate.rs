// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Opt-in permissive translation: one unported op must not reject a whole
//! model, and a soundness refusal must not be laundered by asking nicely.
//!
//! `ny_build::graph::builder` already degrades a `LayerSpec` its converter
//! cannot lower into a conservative `[-inf, +inf]` `OpaqueSkip` node under
//! `GraphNetworkOptions::permissive()`. The bridge gained the same opt-in one
//! layer up (`TranslateOptions::permissive()`), reusing the carrier the
//! `TraceOp::Custom` arm already proved sound. These tests pin the contract:
//!
//! - a model with ONE unported op translates in permissive mode, records the
//!   skip, and propagates `[-inf, +inf]` AT that node — a genuine
//!   over-approximation, never an identity;
//! - the same model still HARD-FAILS in strict mode, which stays the default
//!   for every existing entry point;
//! - an op refused for SOUNDNESS (`Topk`'s ambiguous wire semantics,
//!   `MoeGating`/`IndexPut`/`GridSample`'s data-dependent routing, a lossy
//!   `ToDtype`) refuses in BOTH modes;
//! - the graph-wide dtype gate, the single-variable-input guard and every
//!   non-`UnsupportedOp` error also stay hard in both modes;
//! - a fully supported model is byte-identical under both policies and reports
//!   zero skips, so the mode cannot degrade something silently.

use ndarray::{ArrayD, IxDyn};
use ny_build::GraphNetworkOptions;
use ny_core::{LayerType, NyError};
use ny_tensor::BoundedTensor;
use ny_trace_bridge::coverage::{refusal_class, RefusalClass};
use ny_trace_bridge::schema::{
    ComputationGraph, DType, GridSamplePaddingMode, NodeId, TraceNode, TraceOp, WeightPayload,
};
use ny_trace_bridge::translate::{
    translate, translate_with_metadata, translate_with_options, TranslateOptions, UnportedOpPolicy,
};

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

/// `Input -> Relu -> <op> -> Relu`: a model that is entirely supported except
/// for the one node under test, which is what makes "one op rejects the whole
/// model" the observable failure.
fn model_with(op: TraceOp) -> ComputationGraph {
    ComputationGraph::from_nodes(vec![
        node(0, "x", TraceOp::Input, &[], &[4]),
        node(1, "pre", TraceOp::Relu, &[0], &[4]),
        node(2, "subject", op, &[1], &[4]),
        node(3, "post", TraceOp::Relu, &[2], &[4]),
    ])
}

/// A genuinely UNPORTED op: `coverage` classifies it non-`Unsupported` (a
/// sound lowering exists in principle) and the translator simply has no
/// implementation. Exactly the case permissive mode exists for.
///
/// This used to be `Conv3d`, which has since been ported — `LayerType::Conv3d`
/// and `Conv3dLayer` landed and the bridge emits them. The stand-in is a
/// `Topk` whose flags name a selection `TopkLayer` does not model: the wire is
/// unambiguous (nn serializes `largest`/`sorted`), so this is a missing layer
/// rather than an unreadable trace — degradable by construction.
fn unported_op() -> TraceOp {
    TraceOp::Topk {
        k: 2,
        dim: 0,
        largest: false,
        sorted: true,
    }
}

fn input_box() -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[4]), -1.0_f32),
        ArrayD::from_elem(IxDyn(&[4]), 1.0_f32),
    )
    .expect("valid input box")
}

/// The unported op is refused, and refused as an `UnsupportedOp` — the error
/// class permissive mode keys on. (If the translator ever ports this op, this
/// test fails first and names the replacement to pick — which is exactly what
/// happened to its `Conv3d` predecessor.)
#[test]
fn the_subject_op_is_unported_and_classified_degradable() {
    let err = translate(&model_with(unported_op()))
        .expect_err("the subject op is not ported; pick another unported op if it ever is");
    assert!(
        matches!(err, NyError::UnsupportedOp(_)),
        "permissive mode only degrades UnsupportedOp; got {err:?}"
    );
    assert_eq!(
        refusal_class(&unported_op()),
        RefusalClass::Unported,
        "a sound lowering exists in principle, so the refusal is a missing port"
    );
    assert!(RefusalClass::Unported.is_degradable());
    assert!(!RefusalClass::Soundness.is_degradable());
}

/// STRICT REMAINS THE DEFAULT: the same model still hard-fails through every
/// pre-existing entry point and through an explicitly strict option set.
#[test]
fn strict_entry_points_all_refuse_the_unported_op() {
    let graph = model_with(unported_op());
    assert!(translate(&graph).is_err(), "translate");
    assert!(
        translate_with_metadata(&graph).is_err(),
        "translate_with_metadata"
    );
    assert!(
        translate_with_options(&graph, TranslateOptions::strict()).is_err(),
        "explicit strict"
    );
    assert!(
        translate_with_options(&graph, TranslateOptions::default()).is_err(),
        "TranslateOptions::default() must be strict"
    );
    assert_eq!(
        TranslateOptions::default(),
        TranslateOptions::strict(),
        "the default policy is Refuse"
    );
    assert_eq!(
        TranslateOptions::permissive().unported_op_policy,
        UnportedOpPolicy::DegradeToOpaqueSkip
    );
}

/// Permissive mode translates the model, records the skip with enough detail
/// to be impossible to overlook, and the degraded node propagates
/// `[-inf, +inf]` — a sound over-approximation of whatever the op computes,
/// never an identity passthrough.
#[test]
fn permissive_mode_degrades_one_unported_op_and_records_it() {
    let graph = model_with(unported_op());
    let translation = translate_with_options(&graph, TranslateOptions::permissive())
        .expect("permissive mode must not reject the whole model over one unported op");

    // The skip is reported, counted, and attributable.
    assert_eq!(translation.skipped_count(), 1);
    let skipped = &translation.skipped_nodes[0];
    assert_eq!(skipped.node_id, 2);
    assert_eq!(skipped.trace_name, "subject");
    assert_eq!(skipped.op, "Topk");
    assert!(
        skipped.reason.contains("largest=false"),
        "the original refusal must survive into the record; got {:?}",
        skipped.reason
    );

    // The carrier is the same `LayerType::Unknown` the `Custom` arm emits,
    // which the graph builder substitutes with `Layer::OpaqueSkip`.
    let carrier = translation
        .model
        .network
        .layers
        .iter()
        .find(|l| l.layer_type == LayerType::Unknown)
        .expect("the degraded node emits the Unknown/OpaqueSkip carrier");
    assert_eq!(
        carrier.name, skipped.layer_name,
        "the record must name the layer that actually appears in the model"
    );

    // ... and everything else still lowered: two Relus plus the input identity.
    assert_eq!(
        translation
            .model
            .network
            .layers
            .iter()
            .filter(|l| l.layer_type == LayerType::ReLU)
            .count(),
        2,
        "the rest of the model must still be translated"
    );

    // Vacuous, not identity: IBP through the degraded node widens to +-inf.
    // (The trailing Relu clamps the lower bound to 0, so read the upper.)
    let network = translation
        .model
        .build_graph_network(GraphNetworkOptions::default())
        .expect("the degraded GraphModel builds");
    let out = network
        .propagate_ibp(&input_box())
        .expect("IBP over the OpaqueSkip carrier");
    assert!(
        out.upper().iter().all(|v| v.is_infinite() && *v > 0.0),
        "a degraded node must produce vacuous bounds, not a passthrough: got {:?}",
        out.upper()
    );
}

/// Soundness-class refusals stay HARD in BOTH modes. These are not missing
/// ports: the trace itself cannot be read unambiguously, and a permissive flag
/// must not launder that.
#[test]
fn soundness_refusals_refuse_in_both_modes() {
    // `Topk` used to head this list for its ambiguous wire semantics. It is
    // deliberately gone: nn serializes `largest`/`sorted`, so the trace names
    // exactly one function and the combinations `TopkLayer` cannot model are
    // MISSING-LAYER refusals — degradable, and pinned as such in
    // `unported_topk_combination_degrades` below.
    let soundness_ops = [
        // Data-dependent write routing / gating.
        TraceOp::IndexPut { dim: 0 },
        TraceOp::MoeGating {
            num_experts: 2,
            top_k: 1,
        },
        // Data-dependent source-coordinate mapping.
        TraceOp::GridSample {
            padding_mode: GridSamplePaddingMode::Zeros,
            align_corners: false,
        },
        // Unmodeled cast rounding.
        TraceOp::ToDtype {
            target_dtype: DType::F16,
        },
    ];
    for op in soundness_ops {
        assert_eq!(
            refusal_class(&op),
            RefusalClass::Soundness,
            "{op:?} must classify as a soundness refusal"
        );
        let graph = model_with(op.clone());
        assert!(
            translate_with_options(&graph, TranslateOptions::strict()).is_err(),
            "{op:?}: strict mode must refuse"
        );
        let permissive = translate_with_options(&graph, TranslateOptions::permissive());
        assert!(
            permissive.is_err(),
            "{op:?}: a soundness refusal must NOT be degraded, even in permissive mode"
        );
    }
}

/// The graph-wide dtype gate runs before any per-node policy and is unaffected
/// by it: a non-F32 carrier fails closed in permissive mode too.
#[test]
fn the_dtype_gate_stays_hard_in_permissive_mode() {
    let mut input = node(0, "x", TraceOp::Input, &[], &[4]);
    input.output_dtype = DType::F64;
    let graph =
        ComputationGraph::from_nodes(vec![input, node(1, "act", TraceOp::Relu, &[0], &[4])]);
    for (label, options) in [
        ("strict", TranslateOptions::strict()),
        ("permissive", TranslateOptions::permissive()),
    ] {
        let err = translate_with_options(&graph, options)
            .expect_err("a non-F32 node must fail closed in every mode");
        assert!(
            matches!(err, NyError::UnsupportedOp(ref m)
                if m.contains("output dtype") && m.contains("only soundly supports F32")),
            "{label}: got {err:?}"
        );
    }
}

/// An input-less refusal is never degraded — `OpaqueSkip` derives its shape
/// from a first input edge and ny-propagate rejects a zero-input one, so
/// degrading would build an INVALID graph rather than a conservative one.
/// `SegmentBoundary` is also soundness-class, so both legs cover it.
#[test]
fn a_refused_input_less_node_is_never_degraded() {
    let mut graph = ComputationGraph::from_nodes(vec![
        node(0, "x", TraceOp::Input, &[], &[4]),
        node(
            1,
            "marker",
            TraceOp::SegmentBoundary {
                reason: "test".to_string(),
                input_bounds: None,
            },
            &[],
            &[4],
        ),
    ]);
    // Reachability seeds from the marked outputs; make the marker the output
    // so it is actually reached and translated.
    graph.output_nodes = vec![NodeId(1)];
    assert!(
        translate_with_options(&graph, TranslateOptions::permissive()).is_err(),
        "an input-less refusal must not be degraded into an invalid carrier"
    );
}

/// Only `UnsupportedOp` degrades. A NUMERICAL refusal on a perfectly
/// degradable-class op (`LayerNorm` classifies `SoundButLoose`) stays hard in
/// permissive mode: an invalid epsilon is a defect in the trace, not a missing
/// port, and vacuous-ing it would hide a real problem behind a warning.
#[test]
fn non_unsupported_errors_are_never_degraded() {
    let graph = model_with(TraceOp::LayerNorm {
        // Below `NORMALIZATION_MIN_EPS`: this family divides by
        // sqrt(var + eps) with a DATA-dependent var.
        eps: 0.0,
        weight: WeightPayload::f32(vec![1.0; 4], vec![4]),
        bias: WeightPayload::f32(vec![0.0; 4], vec![4]),
    });
    for (label, options) in [
        ("strict", TranslateOptions::strict()),
        ("permissive", TranslateOptions::permissive()),
    ] {
        let err = translate_with_options(&graph, options)
            .expect_err("an invalid epsilon must fail closed in every mode");
        assert!(
            matches!(err, NyError::NumericalInstability(_)),
            "{label}: got {err:?}"
        );
    }
}

/// A fully supported model is unchanged by the policy: same layer sequence,
/// zero skips, identical bounds. The mode must be inert when nothing needs it.
#[test]
fn a_fully_supported_model_is_identical_under_both_policies() {
    let graph = ComputationGraph::from_nodes(vec![
        node(0, "x", TraceOp::Input, &[], &[4]),
        node(1, "a", TraceOp::Relu, &[0], &[4]),
        node(2, "b", TraceOp::Tanh, &[1], &[4]),
    ]);
    let strict = translate_with_options(&graph, TranslateOptions::strict()).expect("strict");
    let permissive =
        translate_with_options(&graph, TranslateOptions::permissive()).expect("permissive");

    assert_eq!(strict.skipped_count(), 0);
    assert_eq!(
        permissive.skipped_count(),
        0,
        "permissive mode must not degrade anything it does not have to"
    );
    let layer_types = |t: &ny_trace_bridge::translate::Translation| -> Vec<LayerType> {
        t.model
            .network
            .layers
            .iter()
            .map(|l| l.layer_type.clone())
            .collect()
    };
    assert_eq!(layer_types(&strict), layer_types(&permissive));

    let bounds = |t: &ny_trace_bridge::translate::Translation| {
        t.model
            .build_graph_network(GraphNetworkOptions::default())
            .expect("builds")
            .propagate_ibp(&input_box())
            .expect("propagates")
    };
    let (a, b) = (bounds(&strict), bounds(&permissive));
    assert_eq!(a.lower(), b.lower());
    assert_eq!(a.upper(), b.upper());
}
