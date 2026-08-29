// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Runtime cross-checks that pin [`super`]'s declared matrix to the real
//! dispatchers, plus the historical-regression tests.
//!
//! # Cross-check inventory
//!
//! The compile-time half of the guard (the four exhaustive matches in
//! [`super::classify`]) proves that every variant is *classified*. It cannot
//! prove that a classification is *true*. These do:
//!
//! | test | pins |
//! | --- | --- |
//! | [`batched_admission_gate_matches_the_per_layer_arm`] | `supports_batched_crown` ⟺ a batched arm exists — the CumSum regression exactly |
//! | [`batched_matrix_matches_the_real_dispatchers`] | declared `Supported` ⟺ a real batched arm ran |
//! | [`dense_matrix_matches_the_real_dispatcher`] | declared `FallsBack` ⟺ the dense dispatcher really returns `Unsupported` |
//! | [`block_wise_inherits_only_variants_without_a_graph_level_batched_arm`] | the block driver never inherits a verdict it cannot reach |
//! | [`mul_binary_alphas_are_consumed_by_the_backward`] | `MulBinary` really is alpha-bearing, so the router must register it |
//! | [`dense_only_census_is_pinned`] | the bug-class census — dense-supported / batched-fallback — cannot grow silently |
//! | [`concat_sub_where_cumsum_graph_keeps_every_batched_arm`] | the four historical regressions, end to end on one graph |

use std::collections::HashMap;

use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;

use super::catalogue::{canonical_layers, CATALOGUE_WIDTH, UNCONSTRUCTIBLE_VARIANTS};
use super::{classify, DispatchVerdict, Dispatcher, LAYER_VARIANT_COUNT};
use crate::bounds::{BatchedLinearBounds, LinearBounds};
use crate::layers::{
    CompareLayer, CompareOp, ConcatLayer, CumsumLayer, Layer, LinearLayer, MulBinaryLayer,
    SubLayer, WhereLayer,
};
use crate::network::backward_dispatch::{
    dispatch_backward_layer, BackwardDispatchResult, DispatchContext,
};
use crate::network::{GraphNetwork, GraphNode};
use crate::MulBinaryRelaxationMode;

/// The sentinel the batched per-layer dispatcher emits from its explicit
/// unsupported arm.
///
/// Matching the message is deliberate: it is the ONLY signal that separates
/// "this variant has no arm at all" from "this variant's arm ran and declined
/// for this node's shape". Conflating the two is what let three regressions
/// hide — a missing arm and a declining arm both surface as `UnsupportedOp`.
const BATCHED_NO_ARM: &str = "batched CROWN backward not implemented";

fn interval(n: usize, lo: f32, hi: f32) -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[n]), lo),
        ArrayD::from_elem(IxDyn(&[n]), hi),
    )
    .expect("probe interval")
}

// ===================================================================
// Probes: what the REAL dispatchers do with a catalogue sample
// ===================================================================

/// What the batched per-layer dispatcher did with a sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BatchedProbe {
    /// The explicit unsupported arm fired: this variant has NO batched arm.
    NoArm,
    /// An arm ran (whether or not it then declined for the sample's shape).
    ArmExists,
}

fn probe_batched(layer: &Layer) -> BatchedProbe {
    let n = CATALOGUE_WIDTH;
    let bounds = BatchedLinearBounds::identity(&[n]).expect("identity batched bounds");
    let pre = interval(n, -1.0, 1.0);
    match layer.propagate_crown_backward_batched(&bounds, Some(&pre), None) {
        Err(err) if err.to_string().contains(BATCHED_NO_ARM) => BatchedProbe::NoArm,
        _ => BatchedProbe::ArmExists,
    }
}

/// What the dense dispatcher did with a sample, ordered by how strongly it
/// evidences a real arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum DenseProbe {
    /// `BackwardDispatchResult::Unsupported` — the caller IBP-falls-back. The
    /// weakest outcome, and the one a missing arm produces.
    Unsupported,
    /// The arm ran and errored on the probe's shape. Evidence that an arm
    /// exists; not evidence that it produces bounds.
    Errored,
    /// Real bounds came out (`Single` / `Binary` / `Nary` / `PassThrough`).
    Handled,
}

fn probe_dense_over_box(layer: &Layer, lo: f32, hi: f32) -> DenseProbe {
    let n = CATALOGUE_WIDTH;
    let arity = if layer.is_ternary() {
        3
    } else {
        layer.min_inputs().max(1)
    };
    let inputs: Vec<String> = (0..arity).map(|i| format!("in{i}")).collect();
    let bounds = interval(n, lo, hi);
    let node_bounds: HashMap<String, BoundedTensor> = inputs
        .iter()
        .map(|name| (name.clone(), bounds.clone()))
        .collect();
    let ctx = DispatchContext {
        node_name: "probe",
        layer,
        inputs: &inputs,
        pre_activation: &bounds,
        network_input: &bounds,
        node_bounds: (&node_bounds).into(),
        engine: None,
        deadline: None,
        bilinear_alphas: None,
        mul_binary_relaxation: MulBinaryRelaxationMode::default(),
        mul_binary_alphas: None,
        norm_inv_rms_override: None,
    };
    match dispatch_backward_layer(&ctx, &LinearBounds::identity(n)) {
        Ok(BackwardDispatchResult::Unsupported(_)) => DenseProbe::Unsupported,
        Ok(_) => DenseProbe::Handled,
        Err(_) => DenseProbe::Errored,
    }
}

/// Best evidence over two boxes.
///
/// A sign-straddling box and a strictly positive one, because several real arms
/// (`Log`, `Sqrt`, `Reciprocal`, `Atan2`) legitimately decline on a box that
/// crosses their domain boundary. Probing only the straddling box would report
/// those arms as missing — a false accusation, which in a guard is as bad as a
/// missed one.
fn probe_dense(layer: &Layer) -> DenseProbe {
    probe_dense_over_box(layer, -1.0, 1.0).max(probe_dense_over_box(layer, 0.5, 1.5))
}

/// Diagnostic, not a gate: prints what every dispatcher actually does, next to
/// what the matrix declares.
///
/// This is the tool for re-seeding after a real coverage change — run it, read
/// the observed columns, and update the classifiers and the pinned censuses to
/// match. Seeding the matrix by reading the dispatchers instead is how a guard
/// ends up describing an aspiration.
///
/// ```sh
/// cargo test -p ny-propagate --features diagnostics --lib dispatch_matrix -- --nocapture
/// ```
///
/// Behind the off-by-default `diagnostics` feature rather than `#[ignore]`: it
/// prints observed behaviour and asserts nothing, so it is a tool, not disabled
/// coverage. `source_policy`'s `rust_tests_cannot_be_ignored` forbids the latter.
#[cfg(feature = "diagnostics")]
#[test]
fn dump_observed_dispatcher_behaviour() {
    println!("layer\tdense_observed\tbatched_observed\tgate\tdense_declared\tbatched_declared");
    for (name, layer) in canonical_layers() {
        println!(
            "{name}\t{:?}\t{:?}\t{}\t{}\t{}",
            probe_dense(&layer),
            probe_batched(&layer),
            layer.supports_batched_crown(),
            super::verdict(Dispatcher::DenseCrownBackward, &layer).as_str(),
            super::verdict(Dispatcher::BatchedCrownBackward, &layer).as_str(),
        );
    }
    for (name, reason) in UNCONSTRUCTIBLE_VARIANTS {
        println!("{name}\t(no sample)\t(no sample)\t-\t-\t-\t{reason}");
    }
}

// ===================================================================
// Catalogue integrity
// ===================================================================

#[test]
fn catalogue_covers_every_layer_variant_exactly_once() {
    let layers = canonical_layers();
    let mut names: Vec<&str> = layers
        .iter()
        .map(|(name, _)| *name)
        .chain(UNCONSTRUCTIBLE_VARIANTS.iter().map(|(name, _)| *name))
        .collect();
    names.sort_unstable();
    let unique = names.len();
    names.dedup();
    assert_eq!(unique, names.len(), "duplicate entry in the catalogue");
    assert_eq!(
        names.len(),
        LAYER_VARIANT_COUNT,
        "catalogue drifted from `Layer`. The four exhaustive matches in `classify` already \
         forced you to CLASSIFY the new variant; now give it a sample here (or, if it cannot \
         be constructed, an entry in UNCONSTRUCTIBLE_VARIANTS with the reason) and bump \
         LAYER_VARIANT_COUNT."
    );

    // The sample really is the variant the name claims. Catalogue keys are the
    // ENUM variant names (what a reader greps for); a handful of layers report
    // a different `layer_type()` string, so those are aliased explicitly rather
    // than silently skipped.
    const LAYER_TYPE_ALIASES: &[(&str, &str)] = &[
        ("IntervalConst", "IntervalLinear"),
        ("ScatterNd", "ScatterND"),
    ];
    for (name, layer) in &layers {
        let expected = LAYER_TYPE_ALIASES
            .iter()
            .find(|(variant, _)| variant == name)
            .map_or(*name, |(_, layer_type)| *layer_type);
        assert_eq!(
            layer.layer_type(),
            expected,
            "catalogue sample for {name} is a {} — the runtime probes would test the wrong \
             dispatcher arm",
            layer.layer_type()
        );
    }
}

#[test]
fn every_non_supported_verdict_carries_a_reason() {
    for (name, layer) in canonical_layers() {
        for dispatcher in Dispatcher::ALL {
            let verdict = super::verdict(dispatcher, &layer);
            if verdict.is_supported() {
                assert_eq!(verdict.reason(), None);
                continue;
            }
            let reason = verdict.reason().unwrap_or("");
            assert!(
                reason.len() > 20,
                "{name} on {} is {} with no substantive reason. A gap without a stated reason \
                 is indistinguishable from an oversight — which is the whole failure mode this \
                 matrix exists to prevent.",
                dispatcher.as_str(),
                verdict.as_str()
            );
        }
    }
}

// ===================================================================
// Batched dispatcher cross-checks
// ===================================================================

/// The network-level batched admission gate must admit **exactly** the variants
/// whose per-layer batched arm exists.
///
/// This is the CumSum regression in one line. `CumsumLayer::propagate_linear_batched`
/// was implemented and dispatched, but `supports_batched_crown` did not list
/// `CumSum` — so `Network::propagate_crown_batched` and the streaming verifier,
/// which gate on `.all(|l| l.supports_batched_crown())`, refused the batched
/// path for any network containing a single CumSum node.
///
/// The gate is consulted only on SEQUENTIAL networks, whose layers are all
/// unary, so per-layer arm existence is exactly the right ground truth for it.
#[test]
fn batched_admission_gate_matches_the_per_layer_arm() {
    let mut disagreements = Vec::new();
    for (name, layer) in canonical_layers() {
        let gated_in = layer.supports_batched_crown();
        let has_arm = probe_batched(&layer) == BatchedProbe::ArmExists;
        if gated_in != has_arm {
            disagreements.push(format!(
                "{name}: supports_batched_crown()={gated_in} but a batched arm {} exist",
                if has_arm { "DOES" } else { "does NOT" }
            ));
        }
    }
    assert!(
        disagreements.is_empty(),
        "batched admission gate disagrees with the batched dispatcher:\n  {}\n\n\
         `gate=false, arm=true` is the CumSum shape: a whole network is refused the batched \
         path over a layer that would have propagated fine. `gate=true, arm=false` is worse: \
         the gate promises a path the dispatcher cannot deliver, so every such node takes the \
         partial-CROWN fallback after the network was already admitted.",
        disagreements.join("\n  ")
    );
}

/// A variant marked `Supported` on the batched path must actually reach an arm.
///
/// Split by WHERE the arm lives, because the two dispatchers report differently:
/// a graph-level arm (`dispatch_special_batched_operator`) is only reachable
/// through the graph driver, and the per-layer dispatcher lists those variants
/// in its explicit unsupported set on purpose. Asserting that split — rather
/// than just "somewhere there is an arm" — is what stops a multi-input variant
/// from being quietly moved onto the unary path, where its single-input
/// backward would be the wrong function entirely.
#[test]
fn batched_matrix_matches_the_real_dispatchers() {
    let mut mismatches = Vec::new();
    for (name, layer) in canonical_layers() {
        let declared = super::verdict(Dispatcher::BatchedCrownBackward, &layer);
        let observed = probe_batched(&layer);
        let graph_level = classify::batched_arm_is_graph_level(&layer);

        if graph_level {
            if observed != BatchedProbe::NoArm {
                mismatches.push(format!(
                    "{name}: declared graph-level, but the PER-LAYER batched dispatcher has an \
                     arm for it too — one of the two is now dead code"
                ));
            }
            if !declared.is_supported() {
                mismatches.push(format!(
                    "{name}: routed to the graph-level batched dispatcher but declared {}",
                    declared.as_str()
                ));
            }
            continue;
        }

        match (declared.is_supported(), observed) {
            (true, BatchedProbe::NoArm) => mismatches.push(format!(
                "{name}: declared SUPPORTED on the batched path but the dispatcher's explicit \
                 unsupported arm fired — every node of this kind silently takes the looser \
                 generic fallback"
            )),
            (false, BatchedProbe::ArmExists) => mismatches.push(format!(
                "{name}: declared {} on the batched path but a real arm exists — the matrix is \
                 under-reporting coverage, and callers that consult it will keep paying for a \
                 fallback they no longer need",
                declared.as_str()
            )),
            _ => {}
        }
    }
    assert!(
        mismatches.is_empty(),
        "batched dispatch matrix is out of date:\n  {}",
        mismatches.join("\n  ")
    );
}

/// Config-dependent rows: `AdaIN1d` splits on the style mode, and the split
/// must be the same one the admission gate makes.
#[test]
fn variable_style_adain_is_declared_and_gated_out_of_the_batched_path() {
    let variable = super::catalogue::variable_style_adain();
    assert!(
        !variable.supports_batched_crown(),
        "variable-style AdaIN needs live style operands the batched carrier cannot supply"
    );
    assert!(
        super::verdict(Dispatcher::BatchedCrownBackward, &variable).falls_back(),
        "the matrix must record the variable-style AdaIN gap, not inherit the fixed-style row"
    );

    let (_, fixed) = canonical_layers()
        .into_iter()
        .find(|(name, _)| *name == "AdaIN1d")
        .expect("fixed-style AdaIN sample");
    assert!(fixed.supports_batched_crown());
    assert!(super::verdict(Dispatcher::BatchedCrownBackward, &fixed).is_supported());
}

// ===================================================================
// Dense dispatcher cross-check
// ===================================================================

/// Variants whose dense arm exists but declines for the probe's *sample*,
/// not for the variant.
///
/// Pinned so the list can neither grow (hiding a newly-missing arm behind
/// "the probe is inconclusive") nor shrink unnoticed. Each entry needs a
/// reason that is about the SAMPLE, never about the layer.
const DENSE_PROBE_SAMPLE_DECLINES: &[(&str, &str)] = &[
    (
        "SelfAttention",
        "the probe hands three identical 1-D boxes, not Q/K/V matrices",
    ),
    (
        "ScatterAdd",
        "the probe's sample embeds no indices, so the scatter map is unknown",
    ),
    (
        "IndexAdd",
        "the probe's sample embeds no indices, so the index map is unknown",
    ),
    (
        "ScatterNd",
        "the probe's sample embeds no indices, so the scatter map is unknown",
    ),
];

#[test]
fn dense_matrix_matches_the_real_dispatcher() {
    let mut mismatches = Vec::new();
    for (name, layer) in canonical_layers() {
        let declared = super::verdict(Dispatcher::DenseCrownBackward, &layer);
        let observed = probe_dense(&layer);
        let sample_declines = DENSE_PROBE_SAMPLE_DECLINES
            .iter()
            .any(|(pinned, _)| *pinned == name);

        match (declared.is_supported(), observed, sample_declines) {
            // Declared supported, arm ran: agreement.
            (true, DenseProbe::Handled | DenseProbe::Errored, false) => {}
            // Declared supported, probe declined, and that decline is pinned as
            // a property of the sample: agreement, and the pin is still earning
            // its place.
            (true, DenseProbe::Unsupported, true) => {}
            (true, DenseProbe::Unsupported, false) => mismatches.push(format!(
                "{name}: declared SUPPORTED on the dense path but `dispatch_backward_layer` \
                 returned Unsupported for BOTH probe boxes — the caller IBP-concretizes every \
                 node of this kind"
            )),
            (true, _, true) => mismatches.push(format!(
                "{name}: pinned in DENSE_PROBE_SAMPLE_DECLINES but the probe now reaches its \
                 arm — remove the pin so a future regression is not masked by it"
            )),
            // Declared as a fallback: the dispatcher must really say so.
            (false, DenseProbe::Unsupported, _) => {}
            (false, other, _) => mismatches.push(format!(
                "{name}: declared {} on the dense path but the dispatcher produced {other:?} — \
                 the matrix under-reports an arm that exists",
                declared.as_str()
            )),
        }
    }
    assert!(
        mismatches.is_empty(),
        "dense dispatch matrix is out of date:\n  {}",
        mismatches.join("\n  ")
    );
}

// ===================================================================
// Block-wise cross-check
// ===================================================================

/// The per-block driver's unary catch-all forwards to
/// `Layer::propagate_crown_backward_batched`. It never calls
/// `dispatch_special_batched_operator`, so a variant whose batched arm lives at
/// the GRAPH level must never be left in the inherit set: the matrix would
/// claim the block path gets that arm when it provably cannot.
///
/// This is the same bug shape as the three regressions, one layer up — a
/// dispatcher inheriting a capability it does not actually reach.
#[test]
fn block_wise_inherits_only_variants_without_a_graph_level_batched_arm() {
    let mut leaks = Vec::new();
    for (name, layer) in canonical_layers() {
        if classify::block_wise_inherits_batched(&layer)
            && classify::batched_arm_is_graph_level(&layer)
        {
            leaks.push(name);
        }
    }
    assert!(
        leaks.is_empty(),
        "block-wise inherits the batched verdict for {leaks:?}, whose batched arm lives in the \
         GRAPH dispatcher the per-block driver never calls. Give each an explicit override arm \
         in `block_wise_arm` stating what the block path really does."
    );
}

/// The one deliberate divergence in the matrix, stated as a test so it stays
/// deliberate: `Sub` / `Concat` / `Where` are `Supported` on the batched graph
/// path and `FallsBack` on the block path.
#[test]
fn block_wise_records_the_multi_input_divergence_from_the_batched_path() {
    for name in ["Sub", "Concat", "Where"] {
        let (_, layer) = canonical_layers()
            .into_iter()
            .find(|(n, _)| *n == name)
            .expect("catalogue sample");
        assert!(
            super::verdict(Dispatcher::BatchedCrownBackward, &layer).is_supported(),
            "{name} must be supported on the batched graph path"
        );
        assert!(
            super::verdict(Dispatcher::BlockWise, &layer).falls_back(),
            "{name} has no block-wise arm: the per-block driver routes it through the UNARY \
             batched backward. Recording it as supported here would be a false claim."
        );
    }
}

// ===================================================================
// Alpha entry cross-check
// ===================================================================

/// The variants the DAG α-CROWN router registers optimizable alphas for.
///
/// Pinned because the router's own predicates are inline `matches!` blocks with
/// no callable surface. Growing this list means wiring a new alpha family into
/// `init_dag_alpha_state`; shrinking it means a family lost its entry point.
/// `MulBinary` is the most recent addition (#mulbinary-alpha-router) and the
/// reason this pin exists at all.
const ALPHA_BEARING: &[&str] = &[
    "ReLU",
    "Sigmoid",
    "Tanh",
    "Sqrt",
    "Reciprocal",
    "BilinearCrown",
    "MulBinary",
];

#[test]
fn alpha_entry_census_is_pinned() {
    let mut declared: Vec<&str> = canonical_layers()
        .into_iter()
        .filter(|(_, layer)| super::verdict(Dispatcher::AlphaEntry, layer).is_supported())
        .map(|(name, _)| name)
        .collect();
    declared.sort_unstable();
    let mut expected: Vec<&str> = ALPHA_BEARING.to_vec();
    expected.sort_unstable();
    assert_eq!(
        declared, expected,
        "the alpha-bearing set changed. Every variant here must have BOTH a backward that \
         consumes its alphas AND an entry point in `init_dag_alpha_state` that registers them \
         and holds the graph in the optimizer — the MulBinary regression was exactly the case \
         where the first half existed and the second did not."
    );
}

/// Proof that the `MulBinary` alpha-entry row is not aspirational: the dense
/// backward really does consume a `mul_binary_alphas` entry, so a router that
/// does not register one leaves a real degree of freedom unused.
///
/// Uses a deliberately off-centre alpha (`0.9`, against the `0.5` Middle-mode
/// default) so an ignored parameter is observable as identical bounds.
#[test]
fn mul_binary_alphas_are_consumed_by_the_backward() {
    let n = CATALOGUE_WIDTH;
    let layer = Layer::MulBinary(MulBinaryLayer);
    let inputs = vec!["a".to_string(), "b".to_string()];
    let bounds = interval(n, -1.0, 1.0);
    let node_bounds: HashMap<String, BoundedTensor> = inputs
        .iter()
        .map(|name| (name.clone(), bounds.clone()))
        .collect();
    let mut alphas: HashMap<String, Array2<f32>> = HashMap::new();
    alphas.insert("probe".to_string(), Array2::from_elem((2, n), 0.9));

    let dispatch = |alpha: Option<&HashMap<String, Array2<f32>>>| {
        let ctx = DispatchContext {
            node_name: "probe",
            layer: &layer,
            inputs: &inputs,
            pre_activation: &bounds,
            network_input: &bounds,
            node_bounds: (&node_bounds).into(),
            engine: None,
            deadline: None,
            bilinear_alphas: None,
            mul_binary_relaxation: MulBinaryRelaxationMode::default(),
            mul_binary_alphas: alpha,
            norm_inv_rms_override: None,
        };
        match dispatch_backward_layer(&ctx, &LinearBounds::identity(n)) {
            Ok(BackwardDispatchResult::Binary { bounds_a, .. }) => bounds_a.lower_a().clone(),
            other => panic!("MulBinary must dispatch to Binary, got {other:?}"),
        }
    };

    let without = dispatch(None);
    let with = dispatch(Some(&alphas));
    assert_ne!(
        without, with,
        "the dense backward ignored `mul_binary_alphas`. Either MulBinary is not really \
         alpha-bearing (fix ALPHA_BEARING and the alpha_entry classifier) or the alpha plumbing \
         regressed (fix that) — but the matrix must not keep claiming an optimizable degree of \
         freedom that no code reads."
    );
}

// ===================================================================
// The bug-class census
// ===================================================================

/// Variants that reach a real DENSE arm and take the batched path's generic
/// fallback — the precise shape of all three regressions.
///
/// Pinned as a RATCHET, not asserted empty: several entries are legitimate,
/// audited gaps (a dense envelope written against a flat `[out, in]` operand
/// has no sound batched port until someone writes one). A variant appearing
/// here that is not on the list is a new silent degradation; a variant
/// disappearing means a batched arm landed and the list should shrink.
const DENSE_ONLY_CENSUS: &[&str] = &[
    "IntervalConst",
    "AveragePool",
    "MaxPool2d",
    "SelfAttention",
    "Atan2",
    "MinBinary",
    "MaxBinary",
    "Gather",
    "ScatterAdd",
    "IndexAdd",
    "ScatterNd",
    "SkipMerge",
    "OpaqueSkip",
];

#[test]
fn dense_only_census_is_pinned() {
    let mut observed = super::dense_only_variants();
    observed.sort_unstable();
    let mut expected: Vec<&str> = DENSE_ONLY_CENSUS.to_vec();
    expected.sort_unstable();
    assert_eq!(
        observed, expected,
        "the dense-supported / batched-fallback census moved.\n\
         A variant ADDED here is a new silent degradation of exactly the kind that shipped \
         three times this week: nodes of that kind keep verifying, just with a much looser \
         bound and no error anywhere. Either add the batched arm or add the variant to \
         DENSE_ONLY_CENSUS with the audited reason already recorded in `batched_crown_backward`.\n\
         A variant REMOVED here means a batched arm landed — shrink the census."
    );
}

/// The seed's honest state: nothing on these four paths is currently REFUSED.
///
/// Every gap degrades to a sound, looser bound rather than failing closed, so
/// the `Refused` bucket is empty today. It is kept in the taxonomy because a
/// future variant may genuinely have no sound lowering on some path, and
/// because a `Refused` row must never be quietly downgraded to `FallsBack` —
/// that would turn a fail-closed refusal into a silently-published bound.
#[test]
fn no_variant_is_currently_refused_on_any_dispatcher() {
    let mut refusals = Vec::new();
    for (name, layer) in canonical_layers() {
        for dispatcher in Dispatcher::ALL {
            if matches!(
                super::verdict(dispatcher, &layer),
                DispatchVerdict::Refused(_)
            ) {
                refusals.push(format!("{name} on {}", dispatcher.as_str()));
            }
        }
    }
    assert!(
        refusals.is_empty(),
        "new REFUSED classification(s): {refusals:?}. A refusal means the dispatcher fails \
         CLOSED for this variant — no bound at all, not a looser one. If that is really the \
         behaviour, update this test; if the dispatcher actually degrades to IBP, the verdict \
         is FallsBack."
    );
}

// ===================================================================
// Historical regressions: the four arms, end to end, on one graph
// ===================================================================

fn uniform(shape: &[usize], lo: f32, hi: f32) -> BoundedTensor {
    BoundedTensor::new(ArrayD::from_elem(shape, lo), ArrayD::from_elem(shape, hi))
        .expect("uniform interval")
}

fn linear_layer(rows: usize, cols: usize, weights: &[f32], bias: &[f32]) -> Layer {
    Layer::Linear(
        LinearLayer::new(
            Array2::from_shape_vec((rows, cols), weights.to_vec()).expect("weight shape"),
            Some(Array1::from_vec(bias.to_vec())),
        )
        .expect("linear layer"),
    )
}

/// A 2-wide condition node whose IBP bounds are an EXACTLY degenerate `[1, 0]`
/// mask.
///
/// A zero-weight `Linear` will not do: its sound IBP widens by `in_features + 2`
/// ULPs, so the "constant" is `[1-ulp, 1+ulp]` and both the dense and batched
/// constant-mask tiers correctly refuse it. `Compare` emits exactly `(0,0)` /
/// `(1,1)` when the input interval lies wholly on one side of the threshold,
/// which is what a real lowered attention mask looks like.
fn add_exact_mask_condition(graph: &mut GraphNetwork) {
    // shift = x + [+100, -100] => position 0 in [99, 101], position 1 in [-101, -99].
    graph.add_node(GraphNode::from_input(
        "mask_shift",
        linear_layer(2, 2, &[1.0, 0.0, 0.0, 1.0], &[100.0, -100.0]),
    ));
    graph.add_node(GraphNode::new(
        "cond",
        Layer::Compare(CompareLayer::new(0.0, CompareOp::Gt)),
        vec!["mask_shift".to_string()],
    ));
}

/// All four historical regressions on ONE graph, whose exact output is
/// identically zero.
///
/// # Why the shape of this fixture matters more than the arithmetic
///
/// The obvious fixture — put the node under test at the OUTPUT and check the
/// bound — is **vacuous**, and was verified to be so by disabling the batched
/// `Sub` arm and watching it still pass. The reason is worth recording, because
/// it is the same blindness the dispatch matrix exists to remove:
///
/// * the batched driver collects `node_bounds` with a **dense** CROWN-IBP pass,
///   which HAS every arm;
/// * when a node's batched dispatch reports `Unsupported` and nothing else is
///   pending, the driver concretizes that node through its `node_bounds` entry;
/// * at the OUTPUT node that entry is the dense CROWN bound — i.e. exactly the
///   answer the missing arm would have produced.
///
/// So a missing arm at the output is invisible. Every node under test here is
/// therefore INTERIOR, with a live sibling path still pending when it is
/// processed. In that position a missing arm cannot silently degrade: the
/// driver refuses to concretize one path while another is outstanding (it would
/// be unsound to do so) and propagates the error instead. The test asserts both
/// halves — the pass must SUCCEED, and the bound must be exactly zero — so a
/// missing arm fails one or the other:
///
/// ```text
///   lin   = I·x                                  x ∈ [-1, 1]^2
///   cs    = CumSum(lin)                          = [x0, x0 + x1]
///   cond  = Compare(x + [100, -100])             = [1, 0]   (exact mask)
///   pick  = Where(cond, cs, lin)                 = [x0, x1]
///   cat   = Concat(pick, cs)                     = [x0, x1, x0, x0 + x1]
///   pa    = [[0 0 0 1];[0 0 0 1]]·cat            = [x0 + x1, x0 + x1]
///   pb    = [[1 1 0 0];[1 1 0 0]]·cat            = [x0 + x1, x0 + x1]
///   diff  = Sub(pa, pb)                          ≡ [0, 0]
///   cat2  = Concat(diff, lin)                    = [0, 0, x0, x1]
///   out   = [[1 0 0 0];[0 1 0 0]]·cat2           ≡ [0, 0]
/// ```
///
/// The width is 2 throughout, not 1, because `CumsumLayer::propagate_linear_batched`
/// requires a SQUARE grouped carrier (`input_shape == output_shape`, coefficients
/// `[axis_len, axis_len]`). A 1-wide graph output makes that guard refuse, and
/// the CumSum arm then never fires no matter how the rest of the graph is built.
///
/// Plain IBP over the same graph gives `out ∈ [-4, 4]`, so the assertion is not
/// satisfied by accident.
#[test]
fn concat_sub_where_cumsum_graph_keeps_every_batched_arm() {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "lin",
        linear_layer(2, 2, &[1.0, 0.0, 0.0, 1.0], &[0.0, 0.0]),
    ));
    graph.add_node(GraphNode::new(
        "cs",
        Layer::CumSum(CumsumLayer::new(-1, false, false)),
        vec!["lin".to_string()],
    ));
    add_exact_mask_condition(&mut graph);
    graph.add_node(GraphNode::new(
        "pick",
        Layer::Where(WhereLayer::new()),
        vec!["cond".to_string(), "cs".to_string(), "lin".to_string()],
    ));
    graph.add_node(GraphNode::binary(
        "cat",
        Layer::Concat(ConcatLayer::new(-1)),
        "pick",
        "cs",
    ));
    // pa = [cs1, cs1]; pb = [pick0 + pick1, pick0 + pick1]. Both are x0 + x1,
    // by two different routes through the join.
    graph.add_node(GraphNode::new(
        "pa",
        linear_layer(2, 4, &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], &[0.0, 0.0]),
        vec!["cat".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "pb",
        linear_layer(2, 4, &[1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0], &[0.0, 0.0]),
        vec!["cat".to_string()],
    ));
    graph.add_node(GraphNode::binary("diff", Layer::Sub(SubLayer), "pa", "pb"));
    // The `lin` operand is what keeps `diff` INTERIOR: it leaves a sibling path
    // pending when `diff` is processed, so a missing Sub arm cannot quietly
    // concretize through `diff`'s (already exact) dense CROWN-IBP box.
    graph.add_node(GraphNode::binary(
        "cat2",
        Layer::Concat(ConcatLayer::new(-1)),
        "diff",
        "lin",
    ));
    graph.add_node(GraphNode::new(
        "out",
        linear_layer(2, 4, &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], &[0.0, 0.0]),
        vec!["cat2".to_string()],
    ));
    graph.set_output("out");

    let input = uniform(&[2], -1.0, 1.0);
    let batched = match graph.propagate_crown_batched(&input) {
        Ok(bounds) => bounds,
        Err(err) => panic!(
            "batched CROWN aborted on the four-arm graph: {err}\n\
             An interior node whose batched arm is missing cannot degrade quietly — the driver \
             refuses to concretize one path while a sibling path is still pending — so it \
             propagates the error instead. Check which of CumSum / Where / Concat / Sub lost \
             its batched arm."
        ),
    };
    let dense = graph
        .propagate_crown_fixed_slope(&input)
        .expect("dense CROWN should succeed on the four-arm graph");
    let ibp = graph.propagate_ibp(&input).expect("IBP should succeed");

    const TOL: f32 = 1e-4;

    // Soundness first: the certified box must contain every sampled concrete
    // evaluation. A tightness assertion on an unsound bound would be worse than
    // no assertion at all.
    let steps = 9usize;
    for i in 0..steps {
        for j in 0..steps {
            let x = [
                -1.0 + 2.0 * (i as f32) / ((steps - 1) as f32),
                -1.0 + 2.0 * (j as f32) / ((steps - 1) as f32),
            ];
            let cs = [x[0], x[0] + x[1]];
            let pick = [cs[0], x[1]];
            let concrete = cs[1] - (pick[0] + pick[1]);
            assert!(
                concrete.abs() < TOL,
                "fixture is wrong: out = {concrete} at x = {x:?}, expected 0"
            );
            for slot in 0..2 {
                assert!(
                    concrete >= batched.lower()[[slot]] - TOL
                        && concrete <= batched.upper()[[slot]] + TOL,
                    "UNSOUND: concrete out[{slot}] = {concrete} at x = {x:?} escapes the \
                     batched bound [{}, {}]",
                    batched.lower()[[slot]],
                    batched.upper()[[slot]]
                );
            }
        }
    }

    // The fixture is only informative if the fallback it rules out is loose.
    assert!(
        ibp.upper()[[0]] - ibp.lower()[[0]] > 1.0,
        "fixture is degenerate: plain IBP already gives out ∈ [{}, {}], so a node that \
         concretized through its interval would be indistinguishable from one that did not",
        ibp.lower()[[0]],
        ibp.upper()[[0]]
    );

    // Tightness: the cancellation survives only if the backward walk keeps the
    // linear relation through all four arms.
    let (lo, hi) = (batched.lower()[[0]], batched.upper()[[0]]);
    assert!(
        lo.abs() < TOL && hi.abs() < TOL,
        "batched out = [{lo}, {hi}], expected the exact [0, 0]. Some node on this path lost \
         the linear relation — the candidates are CumSum, Where, Concat and Sub, and plain \
         IBP over the same graph gives [{}, {}].",
        ibp.lower()[[0]],
        ibp.upper()[[0]]
    );

    // And no looser than the dense reference on the identical graph.
    assert!(
        lo >= dense.lower()[[0]] - TOL && hi <= dense.upper()[[0]] + TOL,
        "batched out = [{lo}, {hi}] is LOOSER than dense [{}, {}]",
        dense.lower()[[0]],
        dense.upper()[[0]]
    );
}

/// Both `Where` configurations route to the GRAPH-level batched dispatcher, so
/// the matrix's single `Where` row is valid for each.
///
/// An embedded-constant `Where` is UNARY at the graph level (`min_inputs() == 1`:
/// only the condition is a live edge), which makes it exactly the shape that
/// could be mistaken for a per-layer dispatch. It is not: the per-layer batched
/// dispatcher has no `Where` arm in either configuration, and its tier-1
/// constant-select fold lives beside the other three tiers in the graph
/// dispatcher.
#[test]
fn both_where_configurations_route_to_the_graph_level_batched_dispatcher() {
    let plain = Layer::Where(WhereLayer::new());
    let embedded = super::catalogue::embedded_constant_where();
    let Layer::Where(inner) = &embedded else {
        panic!("embedded_constant_where must build a Where");
    };
    assert!(
        inner.has_embedded_constants(),
        "the embedded-constant fixture lost its constants"
    );
    assert_eq!(
        embedded.min_inputs(),
        1,
        "an embedded-constant Where has only the condition as a live edge — the case most \
         likely to be mistaken for a unary dispatch"
    );

    for (label, layer) in [("plain", &plain), ("embedded-constant", &embedded)] {
        assert_eq!(
            probe_batched(layer),
            BatchedProbe::NoArm,
            "{label} Where must NOT have a per-layer batched arm; all four condition tiers \
             live in the graph dispatcher, which is the only place the condition bounds exist"
        );
        assert!(
            classify::batched_arm_is_graph_level(layer),
            "{label} Where must be recorded as a graph-level batched arm"
        );
        assert!(
            super::verdict(Dispatcher::BatchedCrownBackward, layer).is_supported(),
            "{label} Where must be recorded as supported on the batched path"
        );
    }
}

/// Every propagate-side layer NAME must round-trip through
/// [`ny_core::LayerType::from_str`], except a short documented list.
///
/// # The bug this exists to catch
///
/// `Layer::layer_type()` returns ny-propagate's own display name;
/// `LayerType::from_str` maps ONNX/ny-build's spelling. Where the two spellings
/// differ, `from_str` silently yields `LayerType::Unknown` — and every
/// consumer that classifies a graph by layer NAME then reports a perfectly
/// well-relaxed layer as unsupported. It is silent in the worst way: nothing
/// fails, a verdict just quietly gets worse.
///
/// It has bitten twice. `PowConstant` was the first (hence the alias and its
/// comment in `layer_type.rs`). `BilinearCrown` was the second and much more
/// expensive: it is the McCormick relaxation ny-build emits for a MatMul with
/// BOTH operands bounded — i.e. `Q @ K^T` and `probs @ V` — so an unparseable
/// name condemned **every attention model** as a soundness hole in downstream
/// conformance gates. Neither was caught by a test, because the layers
/// themselves were fine; only their names were.
///
/// This walks the same catalogue the dispatch matrix walks, so a new `Layer`
/// variant is covered the moment it is added to `canonical_layers`.
#[test]
fn propagate_layer_names_parse_back_to_a_known_layer_type() {
    use std::str::FromStr as _;

    // Names with no `LayerType` counterpart, each for a stated reason. These
    // are the ONLY layers allowed to parse to `Unknown`.
    //
    // The four `*Constant` primitives and `ExpandLikeLastAxis` are ny-propagate
    // internal arithmetic carriers ny-build synthesizes; they never appear in a
    // ny-core layer inventory. `SkipMerge` is the identity ny-build emits for
    // ONNX's `noop_with_empty_axes` reduction, which likewise has no
    // ny-core spelling. `OpaqueSkip` and `QdqPerturbation` are DELIBERATE:
    // `OpaqueSkip` is the vacuous `[-inf, +inf]` carrier and `QdqPerturbation`
    // fails closed in its only constructor, so both SHOULD read as unsupported
    // to anything classifying a graph — parsing them would hide exactly the
    // thing a consumer needs to see.
    const NO_LAYER_TYPE_COUNTERPART: &[&str] = &[
        "AddConstant",
        "SubConstant",
        "MulConstant",
        "DivConstant",
        "ExpandLikeLastAxis",
        "SkipMerge",
        "OpaqueSkip",
        "QdqPerturbation",
        // The interval carriers must NOT be aliased to `Linear` / an offset:
        // their whole reason for being a separate `Layer` variant is that a
        // reader who sees "Linear" computes with the MIDPOINT and silently
        // drops the enclosure width. Parsing them to their point-valued
        // counterpart would recreate exactly the unsound fold the carrier
        // exists to prevent, so reading as `Unknown` is the correct answer
        // here. See the module doc on `layers::interval_const`.
        "IntervalLinear",
        "IntervalOffset",
    ];

    let mut unparseable = Vec::new();
    for (label, layer) in canonical_layers() {
        let name = layer.layer_type();
        let parsed = ny_core::LayerType::from_str(name);
        let is_unknown = matches!(parsed, Ok(ny_core::LayerType::Unknown) | Err(()));
        if is_unknown && !NO_LAYER_TYPE_COUNTERPART.contains(&name) {
            unparseable.push(format!("{label} -> layer_type() = {name:?}"));
        }
    }
    assert!(
        unparseable.is_empty(),
        "these propagate layer names do not parse back to a LayerType, so every \
         name-based classifier will call them Unknown/unsupported. Add an alias arm to \
         `ny_core::LayerType::from_str` (as \"Pow\" | \"PowConstant\" and \
         \"MatMul\" | \"BilinearCrown\" already do), or, if the layer genuinely has no \
         ny-core counterpart, add it to NO_LAYER_TYPE_COUNTERPART with a reason: {unparseable:#?}"
    );

    // The allow-list must not rot either: an entry that starts parsing (because
    // someone added the alias) should be removed from the list, not left to
    // silently weaken the check above.
    for name in NO_LAYER_TYPE_COUNTERPART {
        assert!(
            matches!(
                ny_core::LayerType::from_str(name),
                Ok(ny_core::LayerType::Unknown) | Err(())
            ),
            "{name} now parses to a real LayerType — drop it from \
             NO_LAYER_TYPE_COUNTERPART so the round-trip check covers it"
        );
    }
}
