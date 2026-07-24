// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #cone-delta oracles: delta seeding of the constrained forward pass
//! (`NY_CONE_REFRESH=1`) must be BIT-identical to the full-history seed path
//! and to the full-recompute reference on a real-shaped conv resnet across a
//! multi-split split→bound→replace-map→split lifecycle, and every fail-closed
//! condition must demonstrably route back to the full-history path.
//!
//! Discrimination technique: the two seed paths differ only in WHICH nodes are
//! recomputed, so a widened ("poisoned") base entry outside one path's cone
//! but inside the other's tells exactly which path ran — the recompute
//! intersects the poison away, the skip preserves it verbatim.

use std::collections::HashMap;
use std::sync::Arc;

use ndarray::{arr1, Array, ArrayD, IxDyn};

use crate::beta_crown::branching::{GraphNeuronConstraint, GraphSplitHistory};
use crate::beta_crown::domain::GraphBabDomain;
use crate::beta_crown::{BetaCrownConfig, BetaCrownVerifier};
use crate::layers::{AddConstantLayer, AddLayer, Conv2dLayer, MaxPool2dLayer, ReLULayer};
use crate::tests::{with_serialized_env_vars, with_serialized_env_vars_removed};
use crate::{BoundedTensor, GraphNetwork, GraphNode, Layer, NETWORK_INPUT};

// ---------- fixture (the #extract-skeleton conv resnet construction idioms) ----------

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

/// Real-shaped conv resnet: conv0 → relu0 → maxpool → identity residual block
/// (b1c1 → b1r1 → b1c2 → add1 with maxpool skip) → AddConstant → projection
/// block (b2r1 → b2c1 vs p2c1 → add2) → relu_out → conv_out. The b1r1 split's
/// cone crosses the add1 join while the join's other input (maxpool) stays
/// out-of-cone — the residual-join-straddling case.
fn conv_resnet_fixture() -> GraphNetwork {
    let mut rng = lcg(0xC0DE_DE17_A001);
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

// ---------- bit-exact comparison helpers ----------

fn tensor_bits(t: &BoundedTensor) -> (Vec<u32>, Vec<u32>) {
    let flat = t.flatten();
    (
        flat.lower().iter().map(|v| v.to_bits()).collect(),
        flat.upper().iter().map(|v| v.to_bits()).collect(),
    )
}

fn assert_maps_bit_identical(
    a: &HashMap<String, Arc<BoundedTensor>>,
    b: &HashMap<String, Arc<BoundedTensor>>,
    ctx: &str,
) {
    assert_eq!(a.len(), b.len(), "{ctx}: node-count mismatch");
    for (name, ta) in a {
        let tb = b
            .get(name)
            .unwrap_or_else(|| panic!("{ctx}: node '{name}' missing on one side"));
        assert_eq!(ta.shape(), tb.shape(), "{ctx}: shape of '{name}'");
        assert_eq!(
            tensor_bits(ta),
            tensor_bits(tb),
            "{ctx}: bits of '{name}' differ"
        );
    }
}

/// Widen one entry of an Arc'd base map by ±`amount` (the poison probe).
fn poison_entry(
    base: &HashMap<String, Arc<BoundedTensor>>,
    node: &str,
    amount: f32,
) -> HashMap<String, Arc<BoundedTensor>> {
    let mut out = base.clone();
    let orig = out.get(node).expect("poison target present");
    let widened = BoundedTensor::new(
        orig.lower().mapv(|v| v - amount),
        orig.upper().mapv(|v| v + amount),
    )
    .expect("widened tensor valid");
    out.insert(node.to_string(), Arc::new(widened));
    out
}

/// First unstable neuron (l < 0 < u) of `rname`'s pre-activation in `map`.
fn unstable_neuron(
    graph: &GraphNetwork,
    map: &HashMap<String, Arc<BoundedTensor>>,
    rname: &str,
) -> usize {
    let pre = pre_of(graph, rname);
    let bt = map.get(&pre).expect("pre bounds").flatten();
    (0..bt.len())
        .find(|&j| bt.lower()[[j]] < 0.0 && bt.upper()[[j]] > 0.0)
        .expect("fixture bug: no unstable neuron")
}

fn pre_of(graph: &GraphNetwork, rname: &str) -> String {
    graph
        .nodes
        .get(rname)
        .and_then(|n| n.inputs.first())
        .expect("relu input")
        .clone()
}

/// Root fixpoint of the constrained forward for the empty history: the exact
/// map the delta invariant is defined against (produced by the same routine
/// whose idempotence the delta path relies on).
fn root_fixpoint(
    verifier: &BetaCrownVerifier,
    graph: &GraphNetwork,
    input: &BoundedTensor,
) -> HashMap<String, Arc<BoundedTensor>> {
    let (map, _) = verifier
        .compute_constrained_forward_bounds(graph, input, &GraphSplitHistory::new(), None, None)
        .expect("root forward");
    map
}

/// Fixpoint of `history` seeded from `base` via the full-history path.
fn fixpoint_of(
    verifier: &BetaCrownVerifier,
    graph: &GraphNetwork,
    input: &BoundedTensor,
    history: &GraphSplitHistory,
    base: &HashMap<String, Arc<BoundedTensor>>,
) -> HashMap<String, Arc<BoundedTensor>> {
    let (map, _) = verifier
        .compute_constrained_forward_bounds(graph, input, history, Some(base), None)
        .expect("fixpoint forward");
    map
}

fn constraint(rname: &str, j: usize, is_active: bool) -> GraphNeuronConstraint {
    GraphNeuronConstraint {
        node_name: rname.to_string(),
        neuron_idx: j,
        is_active,
        score: 0.0,
    }
}

// ---------- oracle (a): lifecycle byte identity ----------

/// The critical oracle: simulate the real BaB cycle — split → bound →
/// replace map → split — over a 3-deep history (late split, residual-join-
/// straddling split, early split; both signs) on the conv resnet. At every
/// depth, the NY_CONE_REFRESH=1 delta path, the full-history-seed path, and
/// the full-recompute reference (`_inner(enable_upstream_cache=false)`) must
/// produce a bit-identical map (every node, `to_bits`) and a bit-identical
/// constrained input. Also pins the `delta_pre_nodes` bookkeeping the paths
/// rely on (append on split, clear on bound).
#[test]
fn cone_delta_lifecycle_byte_identity_conv_resnet() {
    with_serialized_env_vars(&[("NY_CONE_REFRESH", "1")], || {
        let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
        let graph = conv_resnet_fixture();
        let input = box_input(&[2, 6, 6], -1.0, 1.0);

        let map0 = root_fixpoint(&verifier, &graph, &input);
        // `GraphBabDomain::root` takes an owned plain map (unchanged API);
        // materialize the fixture's Arc map for construction only.
        let map0_plain: HashMap<String, BoundedTensor> = map0
            .iter()
            .map(|(k, v)| (k.clone(), v.as_ref().clone()))
            .collect();
        let mut domain =
            GraphBabDomain::root(map0_plain, -100.0, 100.0, &input, false).expect("root");
        // Root carries the delta-unknown sentinel; its map above IS the
        // routine's own empty-history fixpoint, so start the tracked cycle.
        assert_eq!(domain.delta_pre_nodes(), &[NETWORK_INPUT.to_string()]);
        domain.delta_pre_nodes.clear();

        // Late split, residual-join-straddling split, early split; both signs.
        let splits = [("relu_out", true), ("b1r1", false), ("relu0", true)];
        for (depth, (rname, is_active)) in splits.iter().enumerate() {
            let j = unstable_neuron(&graph, &domain.node_bounds, rname);
            let mut child = domain
                .with_constraint(&graph, constraint(rname, j, *is_active), false)
                .expect("with_constraint")
                .expect("feasible child");

            // Bookkeeping: parent delta was cleared post-bounding, so the
            // child's delta is exactly this split's pre-activation node.
            assert_eq!(
                child.delta_pre_nodes(),
                &[pre_of(&graph, rname)],
                "depth {depth}: delta = the new split's pre-node"
            );

            // The child inherited the parent map verbatim — the forward base.
            let base = &child.node_bounds;
            let (delta_map, delta_in) = verifier
                .compute_constrained_forward_bounds(
                    &graph,
                    &input,
                    &child.history,
                    Some(base),
                    Some(&child.delta_pre_nodes),
                )
                .expect("delta-path forward");
            let (full_map, full_in) = verifier
                .compute_constrained_forward_bounds(
                    &graph,
                    &input,
                    &child.history,
                    Some(base),
                    None,
                )
                .expect("full-history forward");
            let (ref_map, ref_in) = verifier
                .compute_constrained_forward_bounds_inner(
                    &graph,
                    &input,
                    &child.history,
                    Some(base),
                    None,
                    false,
                )
                .expect("full-recompute reference");

            let ctx = format!("depth {depth} ({rname}[{j}] active={is_active})");
            assert_eq!(
                tensor_bits(&delta_in),
                tensor_bits(&full_in),
                "{ctx}: constrained input delta vs full"
            );
            assert_eq!(
                tensor_bits(&full_in),
                tensor_bits(&ref_in),
                "{ctx}: constrained input full vs reference"
            );
            assert_maps_bit_identical(&delta_map, &full_map, &format!("{ctx}: delta vs full"));
            assert_maps_bit_identical(&full_map, &ref_map, &format!("{ctx}: full vs reference"));

            // Bound: replace the map with the fixpoint, clear the delta —
            // the exact post-bounding replacement the production sites do.
            child.node_bounds = full_map.clone();
            child.delta_pre_nodes.clear();
            domain = child;
        }
    });
}

// ---------- oracle (b): fail-closed battery ----------

/// Gate ON + a legitimate delta must actually ENGAGE the delta cone: a
/// poisoned out-of-delta-cone (but in-full-cone) entry survives verbatim
/// under the delta path and is intersected away by the full-history path
/// (gate OFF). This is the discriminating engagement/kill-switch pair.
#[test]
fn cone_delta_gate_on_engages_and_gate_off_ignores_delta() {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let graph = conv_resnet_fixture();
    let input = box_input(&[2, 6, 6], -1.0, 1.0);

    // Two-split fixpoint: early (relu0, pre conv0) then late (relu_out, pre add2).
    let map0 = root_fixpoint(&verifier, &graph, &input);
    let j_early = unstable_neuron(&graph, &map0, "relu0");
    let h1 = GraphSplitHistory::new().with_constraint(constraint("relu0", j_early, true));
    let map1 = fixpoint_of(&verifier, &graph, &input, &h1, &map0);
    let j_late = unstable_neuron(&graph, &map1, "relu_out");
    let h2 = h1.with_constraint(constraint("relu_out", j_late, false));
    let map2 = fixpoint_of(&verifier, &graph, &input, &h2, &map1);

    // Claimed delta: only the late split is new. conv0 is inside the FULL
    // cone (seeded by relu0's pre-node) but outside the DELTA cone (add2).
    let delta: Vec<String> = vec![pre_of(&graph, "relu_out")];
    let poisoned = poison_entry(&map2, "conv0", 1.0e6);

    let run = |delta_arg: Option<&[String]>| {
        verifier
            .compute_constrained_forward_bounds(&graph, &input, &h2, Some(&poisoned), delta_arg)
            .expect("forward")
            .0
    };

    with_serialized_env_vars(&[("NY_CONE_REFRESH", "1")], || {
        let on = run(Some(&delta));
        // Delta cone skipped conv0 ⇒ poison preserved verbatim.
        assert_eq!(
            tensor_bits(&on["conv0"]),
            tensor_bits(poisoned["conv0"].as_ref()),
            "gate ON: out-of-delta-cone poison must be preserved (delta path engaged)"
        );
    });

    with_serialized_env_vars_removed(&["NY_CONE_REFRESH"], || {
        let off = run(Some(&delta));
        let off_none = run(None);
        // Kill-switch: a supplied delta is byte-for-byte ignored.
        assert_maps_bit_identical(&off, &off_none, "gate OFF: delta arg vs None");
        // Full-history cone recomputed conv0 ⇒ poison intersected away.
        assert_ne!(
            tensor_bits(&off["conv0"]),
            tensor_bits(poisoned["conv0"].as_ref()),
            "gate OFF: full-history path must recompute conv0 (delta ignored)"
        );
    });
}

/// Stale/empty delta on a non-empty history ⇒ EMPTY cone ⇒ the whole parent
/// map is reused verbatim (increment-1 "aliasing": bit-equal entries; the Arc
/// aliasing itself lands with the cache conversion). Discriminated by a
/// poisoned in-full-cone entry that only the empty-cone path preserves.
#[test]
fn cone_delta_empty_delta_reuses_whole_parent_map() {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let graph = conv_resnet_fixture();
    let input = box_input(&[2, 6, 6], -1.0, 1.0);

    let map0 = root_fixpoint(&verifier, &graph, &input);
    let j = unstable_neuron(&graph, &map0, "relu_out");
    let h1 = GraphSplitHistory::new().with_constraint(constraint("relu_out", j, false));
    let map1 = fixpoint_of(&verifier, &graph, &input, &h1, &map0);

    // conv_out is inside the full-history cone (descendant of add2).
    let poisoned = poison_entry(&map1, "conv_out", 1.0e6);
    let empty_delta: Vec<String> = Vec::new();

    with_serialized_env_vars(&[("NY_CONE_REFRESH", "1")], || {
        let (aliased, _) = verifier
            .compute_constrained_forward_bounds(
                &graph,
                &input,
                &h1,
                Some(&poisoned),
                Some(&empty_delta),
            )
            .expect("empty-delta forward");
        // Empty cone: EVERY entry — including the poison — comes back verbatim.
        assert_maps_bit_identical(&aliased, &poisoned, "empty delta: whole parent map reused");
    });

    with_serialized_env_vars_removed(&["NY_CONE_REFRESH"], || {
        let (recomputed, _) = verifier
            .compute_constrained_forward_bounds(
                &graph,
                &input,
                &h1,
                Some(&poisoned),
                Some(&empty_delta),
            )
            .expect("gate-off forward");
        assert_ne!(
            tensor_bits(&recomputed["conv_out"]),
            tensor_bits(poisoned["conv_out"].as_ref()),
            "gate OFF: the full-history cone recomputes conv_out"
        );
    });
}

/// A `NETWORK_INPUT` delta must fail closed to the full-history path (input
/// splits kill the idempotence premise everywhere). Discriminated by an
/// out-of-full-cone poison: had the delta been accepted, its whole-graph cone
/// would have healed it.
#[test]
fn cone_delta_network_input_delta_fails_closed() {
    with_serialized_env_vars(&[("NY_CONE_REFRESH", "1")], || {
        let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
        let graph = conv_resnet_fixture();
        let input = box_input(&[2, 6, 6], -1.0, 1.0);

        let map0 = root_fixpoint(&verifier, &graph, &input);
        let j = unstable_neuron(&graph, &map0, "relu_out");
        let h1 = GraphSplitHistory::new().with_constraint(constraint("relu_out", j, true));
        let map1 = fixpoint_of(&verifier, &graph, &input, &h1, &map0);

        // b1c1 is OUTSIDE the full-history cone (descendants of add2); a
        // whole-graph delta cone would recompute (heal) it.
        let poisoned = poison_entry(&map1, "b1c1", 1.0e6);
        let bad_delta: Vec<String> = vec![NETWORK_INPUT.to_string()];

        let (with_bad, _) = verifier
            .compute_constrained_forward_bounds(
                &graph,
                &input,
                &h1,
                Some(&poisoned),
                Some(&bad_delta),
            )
            .expect("forward with NETWORK_INPUT delta");
        let (without, _) = verifier
            .compute_constrained_forward_bounds(&graph, &input, &h1, Some(&poisoned), None)
            .expect("forward without delta");

        assert_maps_bit_identical(
            &with_bad,
            &without,
            "NETWORK_INPUT delta must be byte-identical to the full-history path",
        );
        assert_eq!(
            tensor_bits(&with_bad["b1c1"]),
            tensor_bits(poisoned["b1c1"].as_ref()),
            "poison preserved: the whole-graph delta cone was NOT taken"
        );
    });
}

/// A delta seed that is not among the history's own pre-activation nodes
/// (inconsistent tracking) must fail closed to the full-history path.
#[test]
fn cone_delta_seed_outside_history_fails_closed() {
    with_serialized_env_vars(&[("NY_CONE_REFRESH", "1")], || {
        let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
        let graph = conv_resnet_fixture();
        let input = box_input(&[2, 6, 6], -1.0, 1.0);

        let map0 = root_fixpoint(&verifier, &graph, &input);
        let j_early = unstable_neuron(&graph, &map0, "relu0");
        let h1 = GraphSplitHistory::new().with_constraint(constraint("relu0", j_early, true));
        let map1 = fixpoint_of(&verifier, &graph, &input, &h1, &map0);

        // "b1c2" resolves in the graph but is NOT a pre-node of h1 — the
        // gate must refuse; the full-history cone (conv0-seeded, whole graph
        // downstream) then heals the poisoned b1c1.
        let poisoned = poison_entry(&map1, "b1c1", 1.0e6);
        let rogue_delta: Vec<String> = vec!["b1c2".to_string()];

        let (with_rogue, _) = verifier
            .compute_constrained_forward_bounds(
                &graph,
                &input,
                &h1,
                Some(&poisoned),
                Some(&rogue_delta),
            )
            .expect("forward with rogue delta");
        let (without, _) = verifier
            .compute_constrained_forward_bounds(&graph, &input, &h1, Some(&poisoned), None)
            .expect("forward without delta");
        assert_maps_bit_identical(
            &with_rogue,
            &without,
            "rogue delta must be byte-identical to the full-history path",
        );
        // Had the rogue delta been accepted, its cone (b1c2…) would EXCLUDE
        // b1c1 and the poison would have survived.
        assert_ne!(
            tensor_bits(&with_rogue["b1c1"]),
            tensor_bits(poisoned["b1c1"].as_ref()),
            "full-history cone recomputed b1c1 (rogue delta refused)"
        );
    });
}

/// A base map missing an entry must ERROR on both the delta and the
/// full-history path — never the old silent `unwrap_or` substitution.
#[test]
fn cone_delta_missing_base_entry_errors_never_substitutes() {
    with_serialized_env_vars(&[("NY_CONE_REFRESH", "1")], || {
        let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
        let graph = conv_resnet_fixture();
        let input = box_input(&[2, 6, 6], -1.0, 1.0);

        let map0 = root_fixpoint(&verifier, &graph, &input);
        let j = unstable_neuron(&graph, &map0, "relu_out");
        let h1 = GraphSplitHistory::new().with_constraint(constraint("relu_out", j, true));
        let map1 = fixpoint_of(&verifier, &graph, &input, &h1, &map0);

        let mut partial = map1;
        assert!(partial.remove("b1c1").is_some(), "fixture node b1c1");
        let delta: Vec<String> = vec![pre_of(&graph, "relu_out")];

        let with_delta = verifier.compute_constrained_forward_bounds(
            &graph,
            &input,
            &h1,
            Some(&partial),
            Some(&delta),
        );
        let without =
            verifier.compute_constrained_forward_bounds(&graph, &input, &h1, Some(&partial), None);
        let err_a = with_delta.expect_err("partial base must error (delta arg)");
        let err_b = without.expect_err("partial base must error (no delta)");
        assert!(
            !err_a.is_infeasible_domain() && !err_b.is_infeasible_domain(),
            "missing entry is a hard error, not infeasibility"
        );
        assert_eq!(
            err_a.to_string(),
            err_b.to_string(),
            "both paths fail closed identically"
        );
    });
}

/// Gate-interaction self-disable: `clip_in_alpha_crown`, `NY_INTERM_REFINE`,
/// or `NY_STABILIZE` present ⇒ the delta path must refuse (full-history
/// behavior), discriminated by the out-of-delta-cone poison being healed.
#[test]
fn cone_delta_disqualifying_gates_self_disable() {
    let graph = conv_resnet_fixture();
    let input = box_input(&[2, 6, 6], -1.0, 1.0);
    let reference = BetaCrownVerifier::new(BetaCrownConfig::default());

    let map0 = root_fixpoint(&reference, &graph, &input);
    let j_early = unstable_neuron(&graph, &map0, "relu0");
    let h1 = GraphSplitHistory::new().with_constraint(constraint("relu0", j_early, true));
    let map1 = fixpoint_of(&reference, &graph, &input, &h1, &map0);
    let j_late = unstable_neuron(&graph, &map1, "relu_out");
    let h2 = h1.with_constraint(constraint("relu_out", j_late, false));
    let map2 = fixpoint_of(&reference, &graph, &input, &h2, &map1);

    let delta: Vec<String> = vec![pre_of(&graph, "relu_out")];
    let poisoned = poison_entry(&map2, "conv0", 1.0e6);

    let poison_healed = |verifier: &BetaCrownVerifier| -> bool {
        let (map, _) = verifier
            .compute_constrained_forward_bounds(&graph, &input, &h2, Some(&poisoned), Some(&delta))
            .expect("forward");
        tensor_bits(&map["conv0"]) != tensor_bits(poisoned["conv0"].as_ref())
    };

    // Sanity: with no disqualifier the delta path engages (poison survives).
    with_serialized_env_vars(&[("NY_CONE_REFRESH", "1")], || {
        assert!(
            !poison_healed(&reference),
            "control: delta path engages without disqualifiers"
        );
    });

    // clip_in_alpha_crown ⇒ self-disable.
    with_serialized_env_vars(&[("NY_CONE_REFRESH", "1")], || {
        let config = BetaCrownConfig {
            clip_in_alpha_crown: true,
            ..Default::default()
        };
        let clip_verifier = BetaCrownVerifier::new(config);
        assert!(
            poison_healed(&clip_verifier),
            "clip_in_alpha_crown must self-disable the delta path"
        );
    });

    // NY_INTERM_REFINE set (any value) ⇒ self-disable.
    with_serialized_env_vars(
        &[("NY_CONE_REFRESH", "1"), ("NY_INTERM_REFINE", "1")],
        || {
            assert!(
                poison_healed(&reference),
                "NY_INTERM_REFINE must self-disable the delta path"
            );
        },
    );

    // NY_STABILIZE set (any value) ⇒ self-disable.
    with_serialized_env_vars(&[("NY_CONE_REFRESH", "1"), ("NY_STABILIZE", "5")], || {
        assert!(
            poison_healed(&reference),
            "NY_STABILIZE must self-disable the delta path"
        );
    });
}

/// Infeasibility-in-cone: an old clamp that turns infeasible under the newest
/// tightening must produce the IDENTICAL `InfeasibleDomain` on the delta path
/// and the full-history path.
#[test]
fn cone_delta_infeasibility_in_cone_identical() {
    with_serialized_env_vars(&[("NY_CONE_REFRESH", "1")], || {
        let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
        let graph = conv_resnet_fixture();
        let input = box_input(&[2, 6, 6], -1.0, 1.0);

        let map0 = root_fixpoint(&verifier, &graph, &input);
        let j_old = unstable_neuron(&graph, &map0, "b1r1");
        let h1 = GraphSplitHistory::new().with_constraint(constraint("b1r1", j_old, false));
        let mut map1 = fixpoint_of(&verifier, &graph, &input, &h1, &map0);

        // Simulate an earlier pass having proven add2[k] strictly negative,
        // then add an ACTIVE split on relu_out[k]: the pre-clamp inverts the
        // interval inside the delta cone on BOTH paths.
        let k = 0usize;
        let add2 = map1.get("add2").expect("add2 bounds");
        let flat = add2.flatten();
        let mut lo = flat.lower().clone();
        let mut hi = flat.upper().clone();
        lo[[k]] = -1.0;
        hi[[k]] = -0.5;
        let shape = add2.shape().to_vec();
        let tightened = BoundedTensor::new(
            lo.into_shape_clone(IxDyn(&shape)).expect("shape"),
            hi.into_shape_clone(IxDyn(&shape)).expect("shape"),
        )
        .expect("tightened add2");
        map1.insert("add2".to_string(), Arc::new(tightened));

        let h2 = h1.with_constraint(constraint("relu_out", k, true));
        let delta: Vec<String> = vec![pre_of(&graph, "relu_out")];

        let err_delta = verifier
            .compute_constrained_forward_bounds(&graph, &input, &h2, Some(&map1), Some(&delta))
            .expect_err("delta path must be infeasible");
        let err_full = verifier
            .compute_constrained_forward_bounds(&graph, &input, &h2, Some(&map1), None)
            .expect_err("full path must be infeasible");
        assert!(
            err_delta.is_infeasible_domain() && err_full.is_infeasible_domain(),
            "both paths signal InfeasibleDomain (delta: {err_delta}, full: {err_full})"
        );
        assert_eq!(
            err_delta.to_string(),
            err_full.to_string(),
            "identical infeasibility on both paths"
        );
    });
}

// ---------- #cone-delta increment 2: Arc-preserving cache oracles ----------
//
// The internal forward cache is `HashMap<String, Arc<BoundedTensor>>`: seeding
// from a parent map is `Arc::clone` per entry (the historical whole-map deep
// clone is deleted), and only recomputed (in-cone) nodes get fresh
// allocations. Sharing is safe because `BoundedTensor` is plain data (two
// `ArrayD<f32>` + an `Option<Box<L2Constraint>>` of plain arrays — no
// `Cell`/`RefCell`/`Mutex`/atomics anywhere in the type), so a shared tensor
// can only change if some path overwrote it in place through the `Arc`; the
// oracles below check both the sharing and the absence of any such mutation.

/// Design oracle 7 (pointer-sharing) + the allocation-count proxy, on the
/// PRODUCTION full-history seed path (`NY_CONE_REFRESH` unset): after bounding
/// a late-split child from its parent's map,
/// - every out-of-cone entry is `Arc::ptr_eq` with the parent's (the deep
///   clone is gone),
/// - every in-cone entry is a fresh allocation,
/// - the fresh-allocation count equals the cone size EXACTLY (per-child fresh
///   tensor allocation is cone-sized, not graph-sized), and
/// - the map is bit-identical to the full-recompute reference
///   (`_inner(enable_upstream_cache = false)`) — the Arc conversion changed
///   allocation behavior only, never a value.
#[test]
fn arc_cache_pointer_sharing_and_allocation_count_full_history_path() {
    with_serialized_env_vars_removed(&["NY_CONE_REFRESH"], || {
        let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
        let graph = conv_resnet_fixture();
        let input = box_input(&[2, 6, 6], -1.0, 1.0);

        let map0 = root_fixpoint(&verifier, &graph, &input);
        let j = unstable_neuron(&graph, &map0, "relu_out");
        let h1 = GraphSplitHistory::new().with_constraint(constraint("relu_out", j, true));

        let (child_map, _) = verifier
            .compute_constrained_forward_bounds(&graph, &input, &h1, Some(&map0), None)
            .expect("child forward");

        let cone = graph
            .descendants_inclusive(&[pre_of(&graph, "relu_out")])
            .expect("cone");
        assert!(
            cone.len() < map0.len(),
            "fixture: the late split's cone ({}) must be a strict subset of the graph ({})",
            cone.len(),
            map0.len()
        );

        assert_eq!(child_map.len(), map0.len(), "child keeps every node");
        let mut fresh = 0usize;
        for (name, parent_arc) in &map0 {
            let child_arc = child_map.get(name).expect("child entry for every node");
            if cone.contains(name.as_str()) {
                assert!(
                    !Arc::ptr_eq(child_arc, parent_arc),
                    "in-cone '{name}' must be a fresh allocation"
                );
                fresh += 1;
            } else {
                assert!(
                    Arc::ptr_eq(child_arc, parent_arc),
                    "out-of-cone '{name}' must alias the parent's tensor"
                );
            }
        }
        assert_eq!(
            fresh,
            cone.len(),
            "fresh allocations per child == cone size (cone-sized, not graph-sized)"
        );

        let (reference, _) = verifier
            .compute_constrained_forward_bounds_inner(&graph, &input, &h1, Some(&map0), None, false)
            .expect("full recompute reference");
        assert_maps_bit_identical(&child_map, &reference, "arc cache vs full recompute");
    });
}

/// Pointer sharing on the `NY_CONE_REFRESH=1` delta path, including the
/// empty-delta fast case: an empty delta over a fixpointed map yields an EMPTY
/// cone, so the child's map aliases the parent's map in FULL — every entry
/// `Arc::ptr_eq`, zero fresh tensor allocations.
#[test]
fn arc_cache_empty_delta_aliases_whole_parent_map() {
    with_serialized_env_vars(&[("NY_CONE_REFRESH", "1")], || {
        let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
        let graph = conv_resnet_fixture();
        let input = box_input(&[2, 6, 6], -1.0, 1.0);

        let map0 = root_fixpoint(&verifier, &graph, &input);
        let j = unstable_neuron(&graph, &map0, "relu_out");
        let h1 = GraphSplitHistory::new().with_constraint(constraint("relu_out", j, true));
        let map1 = fixpoint_of(&verifier, &graph, &input, &h1, &map0);

        // Re-bound the SAME fixpointed domain (the scalar β-iteration shape):
        // empty delta ⇒ empty cone ⇒ full aliasing.
        let empty_delta: Vec<String> = Vec::new();
        let (aliased, _) = verifier
            .compute_constrained_forward_bounds(
                &graph,
                &input,
                &h1,
                Some(&map1),
                Some(&empty_delta),
            )
            .expect("empty-delta forward");
        assert_eq!(aliased.len(), map1.len());
        for (name, parent_arc) in &map1 {
            assert!(
                Arc::ptr_eq(aliased.get(name).expect("entry"), parent_arc),
                "empty delta: '{name}' must alias the parent tensor"
            );
        }

        // Non-empty delta on the same map: only the delta cone is fresh.
        let j2 = unstable_neuron(&graph, &map1, "relu_out");
        let h2 = h1.with_constraint(constraint("relu_out", j2, false));
        let delta = vec![pre_of(&graph, "relu_out")];
        let (child, _) = verifier
            .compute_constrained_forward_bounds(&graph, &input, &h2, Some(&map1), Some(&delta))
            .expect("delta forward");
        let cone = graph.descendants_inclusive(&delta).expect("cone");
        for (name, parent_arc) in &map1 {
            let shared = Arc::ptr_eq(child.get(name).expect("entry"), parent_arc);
            assert_eq!(
                shared,
                !cone.contains(name.as_str()),
                "delta path sharing must match the delta cone at '{name}'"
            );
        }
    });
}

/// No-aliasing-mutation safety oracle: nothing downstream mutates a shared
/// tensor in place. Two sibling children are bounded from the SAME parent map
/// and a full constrained forward+backward propagation runs over that map as
/// `base_bounds`; afterwards every parent entry is bit-identical to a pristine
/// pre-run snapshot, and the first child still aliases the parent's
/// out-of-cone tensors (an in-place edit via `Arc::get_mut`/`make_mut` would
/// have broken either the bits or the aliasing).
#[test]
fn arc_cache_shared_entries_never_mutated_in_place() {
    with_serialized_env_vars_removed(&["NY_CONE_REFRESH"], || {
        let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
        let graph = conv_resnet_fixture();
        let input = box_input(&[2, 6, 6], -1.0, 1.0);

        let map0 = root_fixpoint(&verifier, &graph, &input);
        let snapshot: Vec<(String, (Vec<u32>, Vec<u32>))> = map0
            .iter()
            .map(|(k, v)| (k.clone(), tensor_bits(v)))
            .collect();

        // Sibling A: late split. Sibling B: residual-join-straddling split.
        let ja = unstable_neuron(&graph, &map0, "relu_out");
        let jb = unstable_neuron(&graph, &map0, "b1r1");
        let h_a = GraphSplitHistory::new().with_constraint(constraint("relu_out", ja, true));
        let h_b = GraphSplitHistory::new().with_constraint(constraint("b1r1", jb, false));
        let (child_a, _) = verifier
            .compute_constrained_forward_bounds(&graph, &input, &h_a, Some(&map0), None)
            .expect("sibling A forward");
        let (child_b, _) = verifier
            .compute_constrained_forward_bounds(&graph, &input, &h_b, Some(&map0), None)
            .expect("sibling B forward");

        // Full production propagation (forward + backward CROWN) over the
        // shared parent map as base_bounds.
        let context =
            crate::beta_crown::domain::GraphCrownContext::new(&h_a, None, Some(&map0), None);
        verifier
            .propagate_crown_with_graph_constraints(&graph, &input, &context, None, None)
            .expect("full propagation");

        // The parent map's tensors are untouched, bit for bit.
        for (name, bits) in &snapshot {
            assert_eq!(
                &tensor_bits(map0.get(name).expect("parent entry")),
                bits,
                "parent tensor '{name}' mutated through a shared Arc"
            );
        }
        // Both siblings still alias the parent's out-of-cone tensors — and
        // therefore (transitively) each other's, deep sibling chains share.
        let cone_a = graph
            .descendants_inclusive(&[pre_of(&graph, "relu_out")])
            .expect("cone A");
        let cone_b = graph
            .descendants_inclusive(&[pre_of(&graph, "b1r1")])
            .expect("cone B");
        for (name, parent_arc) in &map0 {
            if !cone_a.contains(name.as_str()) {
                assert!(
                    Arc::ptr_eq(child_a.get(name).expect("A entry"), parent_arc),
                    "sibling A lost aliasing at '{name}'"
                );
            }
            if !cone_b.contains(name.as_str()) {
                assert!(
                    Arc::ptr_eq(child_b.get(name).expect("B entry"), parent_arc),
                    "sibling B lost aliasing at '{name}'"
                );
            }
        }
    });
}
