// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #extract-skeleton increment 2 wiring tests: `prep_resnet_domain_with` must
//! be BIT-identical to the legacy `prep_resnet_domain` on every domain when the
//! skeleton folds (the increment-1 fold≡extract oracle, re-proven here through
//! the production prep seam), the `NY_EXTRACT_SKELETON=0` kill-switch must
//! revert to the legacy path wholesale, and a stale / mis-keyed skeleton must
//! route to the legacy extraction and still produce a correct prep (the
//! fail-closed spine).
//!
//! #extract-skeleton increment 3 tests (bottom of file): the verifier-level
//! `ResnetSkeletonCache` — hit/miss/stale semantics, cross-call bit-identity
//! through the shared cache, and the build-outside-lock concurrency contract.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use ndarray::{arr1, Array, ArrayD, IxDyn};
use ny_core::{GpuCrownLayer, GpuResnetSegment};

use super::{
    build_call_skeleton, prep_resnet_domain, prep_resnet_domain_ext, prep_resnet_domain_with,
    BetaCrownVerifier, ResnetDomainPrep,
};
use crate::beta_crown::branching::GraphNeuronConstraint;
use crate::beta_crown::engine::ResnetSkeletonCache;
use crate::beta_crown::state::AlphaNeuronState;
use crate::beta_crown::{BetaCrownConfig, GraphBabDomain};
use crate::layers::{AddConstantLayer, AddLayer, Conv2dLayer, MaxPool2dLayer, ReLULayer};
use crate::{BoundedTensor, GraphNetwork, GraphNode, Layer, NETWORK_INPUT};

// ---------- fixture (the increment-1 conv resnet, production domains) ----------

fn box_input(shape: &[usize], lo: f32, hi: f32) -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_elem(IxDyn(shape), lo),
        ArrayD::from_elem(IxDyn(shape), hi),
    )
    .expect("valid input box")
}

/// Deterministic LCG in [-1, 1) (the ny-gpu differential-oracle pattern).
fn lcg(seed: u64) -> impl FnMut() -> f32 {
    let mut state = seed;
    move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

#[allow(clippy::too_many_arguments)]
fn conv(
    rng: &mut impl FnMut() -> f32,
    name: &str,
    input: &str,
    (ic, oc): (usize, usize),
    k: usize,
    s: usize,
    p: usize,
    bias: bool,
) -> GraphNode {
    let kernel = Array::from_shape_vec(
        IxDyn(&[oc, ic, k, k]),
        (0..oc * ic * k * k).map(|_| rng() * 0.35).collect(),
    )
    .expect("kernel");
    let b = bias.then(|| arr1(&(0..oc).map(|_| rng() * 0.1).collect::<Vec<f32>>()));
    let layer = Layer::Conv2d(Conv2dLayer::new(kernel, b, (s, s), (p, p)).expect("conv"));
    if input == NETWORK_INPUT {
        GraphNode::from_input(name, layer)
    } else {
        GraphNode::new(name, layer, vec![input.to_string()])
    }
}

fn relu(name: &str, input: &str) -> GraphNode {
    GraphNode::new(name, Layer::ReLU(ReLULayer), vec![input.to_string()])
}

fn add(name: &str, a: &str, b: &str) -> GraphNode {
    GraphNode::new(
        name,
        Layer::Add(AddLayer),
        vec![a.to_string(), b.to_string()],
    )
}

/// The increment-1 real-shaped conv resnet (see `resnet_skeleton.rs` tests):
/// conv0 → relu0 → maxpool → identity block → AddConstant → projection block
/// → relu_out → conv_out. Exercises every static/dynamic slot kind.
fn conv_resnet_fixture() -> GraphNetwork {
    let mut rng = lcg(0xC4A1_57AC_71F3);
    let mut g = GraphNetwork::new();
    g.add_node(conv(
        &mut rng,
        "conv0",
        NETWORK_INPUT,
        (2, 4),
        3,
        1,
        1,
        true,
    ));
    g.add_node(relu("relu0", "conv0"));
    g.add_node(GraphNode::new(
        "maxpool",
        Layer::MaxPool2d(MaxPool2dLayer::new((2, 2), (2, 2), (0, 0))),
        vec!["relu0".to_string()],
    ));
    g.add_node(conv(&mut rng, "b1c1", "maxpool", (4, 4), 3, 1, 1, false));
    g.add_node(relu("b1r1", "b1c1"));
    g.add_node(conv(&mut rng, "b1c2", "b1r1", (4, 4), 3, 1, 1, true));
    g.add_node(add("add1", "b1c2", "maxpool"));
    g.add_node(GraphNode::new(
        "addc",
        Layer::AddConstant(AddConstantLayer::new(ArrayD::from_elem(IxDyn(&[1]), 0.1))),
        vec!["add1".to_string()],
    ));
    g.add_node(relu("b2r1", "addc"));
    g.add_node(conv(&mut rng, "b2c1", "b2r1", (4, 8), 3, 1, 1, true));
    g.add_node(conv(&mut rng, "p2c1", "addc", (4, 8), 1, 1, 0, false));
    g.add_node(add("add2", "b2c1", "p2c1"));
    g.add_node(relu("relu_out", "add2"));
    g.add_node(conv(
        &mut rng,
        "conv_out",
        "relu_out",
        (8, 2),
        1,
        1,
        0,
        true,
    ));
    g.set_output("conv_out");
    g
}

const FIXTURE_RELUS: [&str; 4] = ["relu0", "b1r1", "b2r1", "relu_out"];
const FIXTURE_START: &str = "conv_out";

fn mk_child(
    graph: &GraphNetwork,
    parent: &GraphBabDomain,
    rname: &str,
    j: usize,
    is_active: bool,
) -> GraphBabDomain {
    parent
        .with_constraint(
            graph,
            GraphNeuronConstraint {
                node_name: rname.to_string(),
                neuron_idx: j,
                is_active,
                score: 0.0,
            },
            false,
        )
        .expect("with_constraint")
        .expect("feasible child")
}

/// First unstable neuron of `rname` per the domain's pre-activation bounds.
fn unstable_neuron(graph: &GraphNetwork, d: &GraphBabDomain, rname: &str) -> usize {
    let pre = graph
        .nodes
        .get(rname)
        .and_then(|n| n.inputs.first())
        .expect("relu input")
        .clone();
    let bt = d.node_bounds.get(&pre).expect("pre bounds").flatten();
    (0..bt.len())
        .find(|&j| bt.lower()[[j]] < 0.0 && bt.upper()[[j]] > 0.0)
        .expect("fixture bug: no unstable neuron")
}

/// Production-shaped domain batch: root (no β), active/inactive split children
/// (β), a two-split deep child, and an α-state child — each with its OWN
/// `compute_constrained_forward_bounds` cache + constrained input (the exact
/// inputs both prep routes consume in production).
#[allow(clippy::type_complexity)]
fn fixture_domains() -> (
    GraphNetwork,
    Vec<GraphBabDomain>,
    Vec<HashMap<String, Arc<BoundedTensor>>>,
    Vec<BoundedTensor>,
) {
    let graph = conv_resnet_fixture();
    let input = box_input(&[2, 6, 6], -1.0, 1.0);
    let node_bounds = graph.collect_node_bounds(&input).expect("node bounds");
    let root = GraphBabDomain::root(node_bounds, -100.0, 100.0, &input, false).expect("root");

    let j0 = unstable_neuron(&graph, &root, "relu_out");
    let active = mk_child(&graph, &root, "relu_out", j0, true);
    let inactive = mk_child(&graph, &root, "relu_out", j0, false);
    let j1 = unstable_neuron(&graph, &active, "b1r1");
    let deep = mk_child(&graph, &active, "b1r1", j1, false);

    // α child: per-neuron α ∈ [0, 1] on every ReLU (both prep routes bridge the
    // SAME GraphDomainAlphaState through `build_alpha_bridge`).
    let mut alpha_child = active.clone();
    for rname in FIXTURE_RELUS {
        let pre = graph
            .nodes
            .get(rname)
            .and_then(|n| n.inputs.first())
            .expect("relu input")
            .clone();
        let bt = alpha_child
            .node_bounds
            .get(&pre)
            .expect("pre bounds")
            .flatten();
        let mk = || {
            (0..bt.len())
                .filter(|&j| bt.lower()[[j]] < 0.0 && bt.upper()[[j]] > 0.0)
                .take(4)
                .map(|j| (j, AlphaNeuronState::new(0.3)))
                .collect::<rustc_hash::FxHashMap<_, _>>()
        };
        alpha_child
            .alpha_state
            .neurons
            .insert(rname.to_string(), mk());
        alpha_child
            .alpha_state
            .upper_neurons
            .insert(rname.to_string(), mk());
    }

    let doms = vec![root, active, inactive, deep, alpha_child];
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let mut caches = Vec::with_capacity(doms.len());
    let mut cinputs = Vec::with_capacity(doms.len());
    for d in &doms {
        let (cache, cin) = verifier
            .compute_constrained_forward_bounds(
                &graph,
                &d.input_bounds,
                &d.history,
                Some(&d.node_bounds),
                None,
            )
            .expect("constrained forward");
        caches.push(cache);
        cinputs.push(cin);
    }
    (graph, doms, caches, cinputs)
}

// ---------- bit-canonicalization (u64 signature; exact `to_bits` on payloads) ----------

fn push_slice_bits(sig: &mut Vec<u64>, v: &[f32]) {
    sig.push(v.len() as u64);
    sig.extend(v.iter().map(|x| u64::from(x.to_bits())));
}

fn push_opt_bits(sig: &mut Vec<u64>, v: Option<&[f32]>) {
    match v {
        Some(v) => {
            sig.push(1);
            push_slice_bits(sig, v);
        }
        None => sig.push(0),
    }
}

fn push_layer_sig(sig: &mut Vec<u64>, layer: &GpuCrownLayer) {
    match layer {
        GpuCrownLayer::Linear {
            weight,
            bias,
            out_features,
            in_features,
            ..
        } => {
            sig.push(1);
            push_slice_bits(sig, weight);
            push_opt_bits(sig, bias.as_deref());
            sig.extend([*out_features as u64, *in_features as u64]);
        }
        GpuCrownLayer::Activation {
            lower_slope,
            upper_slope,
            lower_intercept,
            upper_intercept,
            num_neurons,
        } => {
            sig.push(2);
            for v in [lower_slope, upper_slope, lower_intercept, upper_intercept] {
                push_slice_bits(sig, v);
            }
            sig.push(*num_neurons as u64);
        }
        GpuCrownLayer::Conv2d {
            weight_col,
            bias_expanded,
            out_channels,
            in_channels,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            out_h,
            out_w,
            in_h,
            in_w,
            ..
        } => {
            sig.push(3);
            push_slice_bits(sig, weight_col);
            push_opt_bits(sig, bias_expanded.as_deref());
            sig.extend(
                [
                    *out_channels,
                    *in_channels,
                    *kernel_h,
                    *kernel_w,
                    *stride_h,
                    *stride_w,
                    *pad_h,
                    *pad_w,
                    *out_h,
                    *out_w,
                    *in_h,
                    *in_w,
                ]
                .map(|d| d as u64),
            );
        }
        GpuCrownLayer::ActivationReluDualAlpha {
            lower_pos_slope,
            cross_slope,
            upper_neg_slope,
            cross_intercept,
            num_neurons,
        } => {
            sig.push(4);
            for v in [
                lower_pos_slope,
                cross_slope,
                upper_neg_slope,
                cross_intercept,
            ] {
                push_slice_bits(sig, v);
            }
            sig.push(*num_neurons as u64);
        }
        GpuCrownLayer::MaxPool2d {
            routing,
            ibp_lower,
            ibp_upper,
            input_dim,
            output_dim,
        } => {
            sig.push(5);
            sig.push(routing.len() as u64);
            sig.extend(routing.iter().map(|&r| u64::from(r)));
            push_slice_bits(sig, ibp_lower);
            push_slice_bits(sig, ibp_upper);
            sig.extend([*input_dim as u64, *output_dim as u64]);
        }
    }
}

/// Exact bit signature of EVERY `ResnetDomainPrep` field except `relu_names`
/// (compared directly for a readable failure): segment structure + layer
/// payload bits, frontier_abs, node_abs, beta_signed, in_lo, in_hi.
fn prep_sig(p: &ResnetDomainPrep) -> Vec<u64> {
    let mut sig = Vec::new();
    sig.push(p.segments.len() as u64);
    for seg in &p.segments {
        match seg {
            GpuResnetSegment::Chain(v) => {
                sig.push(10);
                sig.push(v.len() as u64);
                v.iter().for_each(|l| push_layer_sig(&mut sig, l));
            }
            GpuResnetSegment::Residual(v) => {
                sig.push(11);
                sig.push(v.len() as u64);
                v.iter().for_each(|l| push_layer_sig(&mut sig, l));
            }
            GpuResnetSegment::ResidualProj(f, pr) => {
                sig.push(12);
                sig.push(f.len() as u64);
                f.iter().for_each(|l| push_layer_sig(&mut sig, l));
                sig.push(pr.len() as u64);
                pr.iter().for_each(|l| push_layer_sig(&mut sig, l));
            }
        }
    }
    for table in [&p.frontier_abs, &p.node_abs, &p.beta_signed] {
        sig.push(table.len() as u64);
        for row in table.iter() {
            push_slice_bits(&mut sig, row);
        }
    }
    push_slice_bits(&mut sig, &p.in_lo);
    push_slice_bits(&mut sig, &p.in_hi);
    sig
}

fn assert_preps_bit_identical(a: &ResnetDomainPrep, b: &ResnetDomainPrep, ctx: &str) {
    assert_eq!(a.relu_names, b.relu_names, "{ctx}: relu_names");
    // #extract-skeleton x #image-node-crown: both routes here are flags-false,
    // so the stop node must be None on BOTH (and therefore equal).
    assert_eq!(a.stop_node, b.stop_node, "{ctx}: stop_node");
    assert!(
        a.stop_node.is_none(),
        "{ctx}: flags-false preps can never carry a stop node"
    );
    assert!(
        prep_sig(a) == prep_sig(b),
        "{ctx}: ResnetDomainPrep must be bit-identical \
         (segments/frontier_abs/node_abs/beta_signed/in_lo/in_hi)"
    );
}

/// Data pointers of every static Arc payload — equal across two preps iff they
/// share the SAME skeleton's payloads (i.e. the fold path actually ran; a
/// legacy extraction materializes fresh Arcs every call).
fn static_arc_ptrs(segments: &[GpuResnetSegment]) -> Vec<*const f32> {
    let mut out = Vec::new();
    let mut visit = |layers: &[GpuCrownLayer]| {
        for l in layers {
            match l {
                GpuCrownLayer::Linear { weight, bias, .. } => {
                    out.push(weight.as_ptr());
                    if let Some(b) = bias.as_deref() {
                        out.push(b.as_ptr());
                    }
                }
                GpuCrownLayer::Conv2d {
                    weight_col,
                    bias_expanded,
                    ..
                } => {
                    out.push(weight_col.as_ptr());
                    if let Some(b) = bias_expanded.as_deref() {
                        out.push(b.as_ptr());
                    }
                }
                _ => {}
            }
        }
    };
    for seg in segments {
        match seg {
            GpuResnetSegment::Chain(v) | GpuResnetSegment::Residual(v) => visit(v),
            GpuResnetSegment::ResidualProj(f, p) => {
                visit(f);
                visit(p);
            }
        }
    }
    out
}

// ---------- tests ----------

/// Increment-2 wiring oracle: on every production-shaped domain (root, split
/// children, deep child, α child), `prep_resnet_domain_with(Some(skeleton))`
/// is BIT-identical to `prep_resnet_domain` — all seven `ResnetDomainPrep`
/// fields — AND the fold path demonstrably ran (static Arc payloads shared
/// across the domains' preps, which a legacy extraction cannot produce).
#[test]
fn skeleton_prep_wiring_bit_identical_to_legacy_prep() {
    crate::tests::with_serialized_env_vars_removed(&["NY_EXTRACT_SKELETON"], || {
        let (graph, doms, caches, cinputs) = fixture_domains();
        let skel = build_call_skeleton(
            &graph,
            FIXTURE_START,
            &caches[0],
            &cinputs[0],
            Some(&doms[0].alpha_state),
            false,
        )
        .expect("skeleton builds from the exemplar domain");

        let mut fold_ptrs: Vec<Vec<*const f32>> = Vec::new();
        for (i, d) in doms.iter().enumerate() {
            let legacy = prep_resnet_domain(
                &graph,
                FIXTURE_START,
                &caches[i],
                &cinputs[i],
                Some(&d.beta_state),
                Some(&d.alpha_state),
                false,
            )
            .expect("legacy prep succeeds");
            let with = prep_resnet_domain_with(
                Some(&skel),
                &graph,
                FIXTURE_START,
                &caches[i],
                &cinputs[i],
                Some(&d.beta_state),
                Some(&d.alpha_state),
                false,
                false,
                false,
            )
            .expect("skeleton prep succeeds where legacy succeeds");
            assert_preps_bit_identical(&with, &legacy, &format!("domain {i}"));
            fold_ptrs.push(static_arc_ptrs(&with.segments));
        }

        // Split children must actually carry β into beta_signed (the fixture is
        // not vacuous) — domain 1 is the active child of a relu_out split.
        assert!(
            !doms[1].beta_state.is_empty(),
            "fixture: split child carries β entries"
        );

        // Prove the fold path ran on EVERY domain: all preps share the
        // skeleton's static Arc payloads (a legacy fallback would have fresh
        // allocations and break pointer equality).
        assert!(!fold_ptrs[0].is_empty(), "fixture has static Arc payloads");
        for (i, ptrs) in fold_ptrs.iter().enumerate().skip(1) {
            assert_eq!(
                ptrs, &fold_ptrs[0],
                "domain {i}: static Arc payloads must be shared across folds"
            );
        }
    });
}

/// `NY_EXTRACT_SKELETON=0` kill-switch: the per-call skeleton build refuses
/// wholesale (every caller then passes `skeleton = None`), and the `None`
/// route is the legacy prep verbatim. Default (unset) and `=1` are ON.
#[test]
fn skeleton_gate_kill_switch_reverts_to_legacy_wholesale() {
    let (graph, doms, caches, cinputs) = fixture_domains();
    crate::tests::with_serialized_env_vars(&[("NY_EXTRACT_SKELETON", "0")], || {
        assert!(
            !crate::network::extract_skeleton_enabled(),
            "NY_EXTRACT_SKELETON=0 must disable the skeleton lane"
        );
        assert!(
            build_call_skeleton(&graph, FIXTURE_START, &caches[0], &cinputs[0], None, false)
                .is_none(),
            "kill-switch: the per-call skeleton build must refuse wholesale"
        );
        // The gated route (`skeleton = None`) is the legacy prep by delegation.
        let legacy = prep_resnet_domain(
            &graph,
            FIXTURE_START,
            &caches[0],
            &cinputs[0],
            Some(&doms[0].beta_state),
            Some(&doms[0].alpha_state),
            false,
        )
        .expect("legacy prep succeeds");
        let gated = prep_resnet_domain_with(
            None,
            &graph,
            FIXTURE_START,
            &caches[0],
            &cinputs[0],
            Some(&doms[0].beta_state),
            Some(&doms[0].alpha_state),
            false,
            false,
            false,
        )
        .expect("gated prep succeeds");
        assert_preps_bit_identical(&gated, &legacy, "kill-switch route");
    });
    crate::tests::with_serialized_env_vars_removed(&["NY_EXTRACT_SKELETON"], || {
        assert!(
            crate::network::extract_skeleton_enabled(),
            "gate must default ON"
        );
        assert!(
            build_call_skeleton(&graph, FIXTURE_START, &caches[0], &cinputs[0], None, false)
                .is_some(),
            "default ON must build the skeleton"
        );
    });
    crate::tests::with_serialized_env_vars(&[("NY_EXTRACT_SKELETON", "1")], || {
        assert!(
            crate::network::extract_skeleton_enabled(),
            "explicit =1 keeps the gate ON"
        );
    });
}

/// Fail-closed spine: a STALE skeleton (built from a graph whose node sequence
/// differs — `matches_graph` fails) and a MIS-KEYED skeleton (built for a
/// different start node — `cache_key` fails) must both route to the legacy
/// extraction and still produce a prep bit-identical to `prep_resnet_domain`;
/// and a domain BOTH routes refuse (missing bounds) yields `None` on both
/// (None-agreement is behavior).
#[test]
fn stale_or_miskeyed_skeleton_routes_to_legacy_prep() {
    crate::tests::with_serialized_env_vars_removed(&["NY_EXTRACT_SKELETON"], || {
        let (graph, doms, caches, cinputs) = fixture_domains();

        // STALE: same walk, but the source graph has an extra (unvisited) node,
        // so the node-name sequence differs and `matches_graph` refuses. The
        // backward walk from FIXTURE_START never touches the extra node, so the
        // base caches drive the build unchanged.
        let mut extended = conv_resnet_fixture();
        extended.add_node(relu("extra_tail", "conv_out"));
        let stale = build_call_skeleton(
            &extended,
            FIXTURE_START,
            &caches[0],
            &cinputs[0],
            None,
            false,
        )
        .expect("skeleton builds on the extended graph");
        assert!(
            !stale.matches_graph(&graph),
            "fixture: the extended-graph skeleton must be stale for the base graph"
        );
        for (i, d) in doms.iter().enumerate() {
            let legacy = prep_resnet_domain(
                &graph,
                FIXTURE_START,
                &caches[i],
                &cinputs[i],
                Some(&d.beta_state),
                Some(&d.alpha_state),
                false,
            )
            .expect("legacy prep succeeds");
            let with_stale = prep_resnet_domain_with(
                Some(&stale),
                &graph,
                FIXTURE_START,
                &caches[i],
                &cinputs[i],
                Some(&d.beta_state),
                Some(&d.alpha_state),
                false,
                false,
                false,
            )
            .expect("stale skeleton must fall back to a correct legacy prep");
            assert_preps_bit_identical(&with_stale, &legacy, &format!("stale, domain {i}"));
        }

        // MIS-KEYED: a valid skeleton for FIXTURE_START used at a TRUNCATED
        // start node (the interm-refine seed shape) — `cache_key` refuses, the
        // legacy extraction from the truncated start runs instead.
        let skel = build_call_skeleton(&graph, FIXTURE_START, &caches[0], &cinputs[0], None, false)
            .expect("skeleton builds");
        let legacy_trunc = prep_resnet_domain(
            &graph,
            "add2",
            &caches[0],
            &cinputs[0],
            Some(&doms[0].beta_state),
            Some(&doms[0].alpha_state),
            false,
        )
        .expect("legacy truncated-start prep succeeds");
        let with_trunc = prep_resnet_domain_with(
            Some(&skel),
            &graph,
            "add2",
            &caches[0],
            &cinputs[0],
            Some(&doms[0].beta_state),
            Some(&doms[0].alpha_state),
            false,
            false,
            false,
        )
        .expect("mis-keyed skeleton must fall back to a correct legacy prep");
        assert_preps_bit_identical(&with_trunc, &legacy_trunc, "mis-keyed start node");

        // None-agreement: a domain the legacy route refuses (missing bounds
        // entry) must be refused by the skeleton route too — never a
        // fold-produced prep where legacy would have handed the domain to the
        // CPU dense path.
        let mut broken = caches[0].clone();
        assert!(broken.remove("b1r1").is_some(), "fixture node b1r1");
        assert!(
            prep_resnet_domain(
                &graph,
                FIXTURE_START,
                &broken,
                &cinputs[0],
                None,
                None,
                false
            )
            .is_none(),
            "legacy refuses the broken domain"
        );
        assert!(
            prep_resnet_domain_with(
                Some(&skel),
                &graph,
                FIXTURE_START,
                &broken,
                &cinputs[0],
                None,
                None,
                false,
                false,
                false,
            )
            .is_none(),
            "skeleton route must refuse the broken domain too (None-agreement)"
        );
    });
}

/// #extract-skeleton x #image-node-crown reconciliation: `stop_at_bounded=true`
/// must DECLINE the skeleton fold outright — the prep routes to the legacy
/// extraction (observable: no cross-call static-Arc sharing, which only the
/// fold can produce) — and when the walk stops at a frozen-bounded interior
/// node M the prep must honor box(M): `stop_node == Some(M)` and
/// `in_lo`/`in_hi` bit-identical to the frozen map's entry for M.
#[test]
fn stop_at_bounded_declines_skeleton_and_honors_stop_box() {
    crate::tests::with_serialized_env_vars_removed(&["NY_EXTRACT_SKELETON"], || {
        let (graph, doms, caches, cinputs) = fixture_domains();
        let skel = build_call_skeleton(
            &graph,
            FIXTURE_START,
            &caches[0],
            &cinputs[0],
            Some(&doms[0].alpha_state),
            false,
        )
        .expect("skeleton builds from the exemplar domain");

        // (a) Decline oracle on the UNBROKEN domain: with `stop_at_bounded=true`
        // the walk still reaches NETWORK_INPUT (stop_node None; every field
        // bit-identical to the flags-false legacy prep), but the fold must NOT
        // have run: two preps through `Some(skeleton)` share no static Arc
        // payloads (folds would share ALL of them with the skeleton).
        let legacy = prep_resnet_domain(
            &graph,
            FIXTURE_START,
            &caches[0],
            &cinputs[0],
            None,
            None,
            false,
        )
        .expect("legacy prep succeeds");
        let a = prep_resnet_domain_with(
            Some(&skel),
            &graph,
            FIXTURE_START,
            &caches[0],
            &cinputs[0],
            None,
            None,
            false,
            false,
            true,
        )
        .expect("stop_at_bounded prep succeeds on the unbroken domain");
        let b = prep_resnet_domain_with(
            Some(&skel),
            &graph,
            FIXTURE_START,
            &caches[0],
            &cinputs[0],
            None,
            None,
            false,
            false,
            true,
        )
        .expect("second stop_at_bounded prep succeeds");
        assert!(
            a.stop_node.is_none(),
            "unbroken walk reaches NETWORK_INPUT — no stop"
        );
        assert_eq!(a.relu_names, legacy.relu_names, "unbroken: relu_names");
        assert!(
            prep_sig(&a) == prep_sig(&legacy),
            "unbroken stop_at_bounded prep must be bit-identical to the legacy prep"
        );
        let pa = static_arc_ptrs(&a.segments);
        let pb = static_arc_ptrs(&b.segments);
        assert!(!pa.is_empty(), "fixture has static Arc payloads");
        assert!(
            pa.iter().all(|p| !pb.contains(p)),
            "skeleton must be DECLINED under stop_at_bounded \
             (no cross-call static-Arc sharing — only the fold can produce it)"
        );

        // (b) Stop-box oracle: sever the walk below `relu0` by removing conv0's
        // frozen entry. Flags-false must keep the historical refusal (on BOTH
        // routes); `stop_at_bounded=true` must stop AT relu0 and carry
        // box(relu0) VERBATIM as the concretization box.
        let mut broken = caches[0].clone();
        assert!(broken.remove("conv0").is_some(), "fixture node conv0");
        assert!(
            prep_resnet_domain_with(
                Some(&skel),
                &graph,
                FIXTURE_START,
                &broken,
                &cinputs[0],
                None,
                None,
                false,
                false,
                false,
            )
            .is_none(),
            "flags-false must keep the historical refusal without conv0"
        );
        let stopped = prep_resnet_domain_with(
            Some(&skel),
            &graph,
            FIXTURE_START,
            &broken,
            &cinputs[0],
            None,
            None,
            false,
            false,
            true,
        )
        .expect("stop_at_bounded prep succeeds via the frozen stop");
        assert_eq!(
            stopped.stop_node.as_deref(),
            Some("relu0"),
            "walk stops at the deepest frozen-bounded node"
        );
        let m = broken.get("relu0").expect("relu0 frozen entry");
        assert_eq!(stopped.in_lo.len(), m.lower().len(), "stop box dim");
        assert!(
            m.lower()
                .iter()
                .zip(stopped.in_lo.iter())
                .all(|(&x, &y)| x.to_bits() == y.to_bits())
                && m.upper()
                    .iter()
                    .zip(stopped.in_hi.iter())
                    .all(|(&x, &y)| x.to_bits() == y.to_bits()),
            "in_lo/in_hi must be box(relu0)'s f32 endpoints VERBATIM"
        );
        assert_eq!(
            stopped.relu_names,
            vec!["relu_out", "b2r1", "b1r1"],
            "relu0 itself is NOT extracted (the stack is graph(relu0 -> conv_out])"
        );

        // Routes-to-legacy: the Some(skeleton) route must be bit-identical to
        // the no-skeleton `prep_resnet_domain_ext` route (same legacy walk).
        let ext = prep_resnet_domain_ext(
            &graph,
            FIXTURE_START,
            &broken,
            &cinputs[0],
            None,
            None,
            false,
            false,
            true,
        )
        .expect("ext prep succeeds via the frozen stop");
        assert_eq!(stopped.stop_node, ext.stop_node, "stop_node agrees");
        assert_eq!(stopped.relu_names, ext.relu_names, "relu_names agree");
        assert!(
            prep_sig(&stopped) == prep_sig(&ext),
            "Some(skeleton) route must be bit-identical to the no-skeleton ext route"
        );
    });
}

// ---------- #extract-skeleton increment 3: verifier-level ResnetSkeletonCache ----------

/// Increment-3 cache semantics: same `(graph, key)` twice ⇒ ONE build (the
/// second call is a `matches_graph`-validated hit sharing the same `Arc`); a
/// STALE entry (different graph node sequence, same key) ⇒ rebuild AND
/// replace; a distinct start node ⇒ a distinct entry (no eviction of the
/// other key); a failed build is NOT cached (no negative caching); and the
/// `NY_EXTRACT_SKELETON=0` kill-switch refuses even a WARM cache — without
/// invoking the build closure — then serves again once the gate is back on.
#[test]
fn skeleton_cache_hit_miss_stale_semantics() {
    crate::tests::with_env_edits(|env| {
        env.remove("NY_EXTRACT_SKELETON");
        let (graph, _doms, caches, cinputs) = fixture_domains();
        let cache = ResnetSkeletonCache::default();
        let builds = AtomicUsize::new(0);
        let build_at = |g: &GraphNetwork, start: &str| {
            builds.fetch_add(1, Ordering::SeqCst);
            build_call_skeleton(g, start, &caches[0], &cinputs[0], None, false)
        };

        // MISS ⇒ one build; HIT ⇒ no rebuild, the SAME Arc.
        let first = cache
            .get_or_build(&graph, FIXTURE_START, false, || {
                build_at(&graph, FIXTURE_START)
            })
            .expect("miss builds");
        assert_eq!(builds.load(Ordering::SeqCst), 1, "miss runs the build once");
        let second = cache
            .get_or_build(&graph, FIXTURE_START, false, || {
                build_at(&graph, FIXTURE_START)
            })
            .expect("hit serves");
        assert_eq!(builds.load(Ordering::SeqCst), 1, "hit must not rebuild");
        assert!(Arc::ptr_eq(&first, &second), "hit shares the cached Arc");

        // Distinct start node ⇒ distinct entry (one more build); the original
        // entry still hits afterwards (no cross-key eviction).
        let trunc = cache
            .get_or_build(&graph, "add2", false, || build_at(&graph, "add2"))
            .expect("truncated-start skeleton builds");
        assert_eq!(builds.load(Ordering::SeqCst), 2, "distinct key builds");
        assert_eq!(trunc.cache_key(), ("add2", false));
        let again = cache
            .get_or_build(&graph, FIXTURE_START, false, || {
                build_at(&graph, FIXTURE_START)
            })
            .expect("original entry still cached");
        assert_eq!(
            builds.load(Ordering::SeqCst),
            2,
            "distinct keys must not evict each other"
        );
        assert!(Arc::ptr_eq(&first, &again));

        // STALE hit (same key, different node sequence — the increment-2
        // extended-graph shape; the backward walk never touches the extra
        // node, so the base caches drive the build unchanged) ⇒ rebuild AND
        // replace: the replacement then serves the NEW graph.
        let mut extended = conv_resnet_fixture();
        extended.add_node(relu("extra_tail", "conv_out"));
        let rebuilt = cache
            .get_or_build(&extended, FIXTURE_START, false, || {
                build_at(&extended, FIXTURE_START)
            })
            .expect("stale hit rebuilds");
        assert_eq!(builds.load(Ordering::SeqCst), 3, "stale hit must rebuild");
        assert!(
            !Arc::ptr_eq(&first, &rebuilt),
            "a stale entry must never be served"
        );
        assert!(
            rebuilt.matches_graph(&extended) && !rebuilt.matches_graph(&graph),
            "the rebuild is keyed to the NEW graph"
        );
        let rehit = cache
            .get_or_build(&extended, FIXTURE_START, false, || {
                build_at(&extended, FIXTURE_START)
            })
            .expect("replacement is cached");
        assert_eq!(builds.load(Ordering::SeqCst), 3, "replacement serves hits");
        assert!(Arc::ptr_eq(&rebuilt, &rehit));

        // A failed build is NOT cached: `None` result, and the NEXT call for
        // the same key runs the build again (never a negative-cache entry).
        for expected in [4usize, 5] {
            assert!(
                cache
                    .get_or_build(&graph, "no_such_node", false, || {
                        builds.fetch_add(1, Ordering::SeqCst);
                        None
                    })
                    .is_none(),
                "a refused build yields None"
            );
            assert_eq!(builds.load(Ordering::SeqCst), expected);
        }

        // Kill-switch on a WARM cache: refuse without building, then serve the
        // still-cached entry once the gate is back on (no rebuild).
        env.set("NY_EXTRACT_SKELETON", "0");
        assert!(
            cache
                .get_or_build(&extended, FIXTURE_START, false, || {
                    build_at(&extended, FIXTURE_START)
                })
                .is_none(),
            "kill-switch must refuse even a warm cache (wholesale revert)"
        );
        assert_eq!(
            builds.load(Ordering::SeqCst),
            5,
            "gated call must not invoke the build"
        );
        env.remove("NY_EXTRACT_SKELETON");
        let served = cache
            .get_or_build(&extended, FIXTURE_START, false, || {
                build_at(&extended, FIXTURE_START)
            })
            .expect("gate back on: cached entry serves");
        assert_eq!(builds.load(Ordering::SeqCst), 5, "no rebuild after regate");
        assert!(Arc::ptr_eq(&rebuilt, &served));
    });
}

/// Increment-3 cross-call bit-identity: two separate prep passes over
/// DIFFERENT domain sets, each consulting the shared verifier-level cache
/// with its own exemplar (exactly what two batched calls sharing the verifier
/// do), produce preps BIT-identical to the legacy `prep_resnet_domain` on
/// every domain — with ONE build total and the skeleton's static Arc payloads
/// shared across both passes (which the legacy route cannot produce).
#[test]
fn verifier_cache_cross_call_preps_bit_identical_to_legacy() {
    crate::tests::with_serialized_env_vars_removed(&["NY_EXTRACT_SKELETON"], || {
        let (graph, doms, caches, cinputs) = fixture_domains();
        let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
        let builds = AtomicUsize::new(0);
        let mut pass_skels = Vec::new();
        let mut fold_ptrs: Vec<Vec<*const f32>> = Vec::new();
        // Pass 1 = {root, active, inactive}; pass 2 = {deep, α child}. Each
        // pass offers ITS OWN exemplar (pass 2's build closure must never run:
        // the cache serves pass 1's skeleton — static content is exemplar-
        // independent, and the per-domain fold re-bakes all dynamic state).
        for pass in [&[0usize, 1, 2][..], &[3usize, 4][..]] {
            let ex = pass[0];
            let skel = verifier
                .skeleton_cache
                .get_or_build(&graph, FIXTURE_START, false, || {
                    builds.fetch_add(1, Ordering::SeqCst);
                    build_call_skeleton(
                        &graph,
                        FIXTURE_START,
                        &caches[ex],
                        &cinputs[ex],
                        Some(&doms[ex].alpha_state),
                        false,
                    )
                })
                .expect("verifier cache serves a skeleton");
            for &i in pass {
                let legacy = prep_resnet_domain(
                    &graph,
                    FIXTURE_START,
                    &caches[i],
                    &cinputs[i],
                    Some(&doms[i].beta_state),
                    Some(&doms[i].alpha_state),
                    false,
                )
                .expect("legacy prep succeeds");
                let with = prep_resnet_domain_with(
                    Some(&*skel),
                    &graph,
                    FIXTURE_START,
                    &caches[i],
                    &cinputs[i],
                    Some(&doms[i].beta_state),
                    Some(&doms[i].alpha_state),
                    false,
                    false,
                    false,
                )
                .expect("cached-skeleton prep succeeds where legacy succeeds");
                assert_preps_bit_identical(&with, &legacy, &format!("cross-call, domain {i}"));
                fold_ptrs.push(static_arc_ptrs(&with.segments));
            }
            pass_skels.push(skel);
        }
        assert_eq!(
            builds.load(Ordering::SeqCst),
            1,
            "second pass must reuse the first pass's skeleton (one build total)"
        );
        assert!(
            Arc::ptr_eq(&pass_skels[0], &pass_skels[1]),
            "both passes share the SAME cached skeleton Arc"
        );
        assert!(!fold_ptrs[0].is_empty(), "fixture has static Arc payloads");
        for (i, ptrs) in fold_ptrs.iter().enumerate().skip(1) {
            assert_eq!(
                ptrs, &fold_ptrs[0],
                "prep {i}: static Arc payloads shared across BOTH passes"
            );
        }
    });
}

/// Increment-3 concurrency smoke: `get_or_build` under a rayon fan-out. The
/// map lock is held for map ops only — the build runs OUTSIDE the lock with a
/// double-checked, last-write-wins insert — so concurrent callers never
/// deadlock behind a build. The fan-out runs against a WARMED cache because
/// that is what makes "exactly one build" deterministic: cold racers are
/// ALLOWED to build redundantly under the last-write-wins contract (equivalent
/// skeletons, cost-only), so the assertion pins the warmed-hit path all
/// production fan-outs after the first batch take.
#[test]
fn skeleton_cache_concurrent_get_or_build_smoke() {
    crate::tests::with_serialized_env_vars_removed(&["NY_EXTRACT_SKELETON"], || {
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        let (graph, _doms, caches, cinputs) = fixture_domains();
        let cache = ResnetSkeletonCache::default();
        let builds = AtomicUsize::new(0);
        let build = || {
            builds.fetch_add(1, Ordering::SeqCst);
            build_call_skeleton(&graph, FIXTURE_START, &caches[0], &cinputs[0], None, false)
        };
        let warm = cache
            .get_or_build(&graph, FIXTURE_START, false, build)
            .expect("warm build");
        assert_eq!(builds.load(Ordering::SeqCst), 1);
        let arcs: Vec<_> = (0..8usize)
            .into_par_iter()
            .map(|_| {
                cache
                    .get_or_build(&graph, FIXTURE_START, false, build)
                    .expect("concurrent hit")
            })
            .collect();
        assert_eq!(
            builds.load(Ordering::SeqCst),
            1,
            "warmed fan-out: one build total across all 8 workers"
        );
        for (i, a) in arcs.iter().enumerate() {
            assert!(
                Arc::ptr_eq(&warm, a),
                "worker {i} must share the cached Arc"
            );
        }
    });
}
