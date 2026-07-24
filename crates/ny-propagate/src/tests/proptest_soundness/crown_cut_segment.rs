// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Soundness oracle for the CROWN-IBP sweep's backward-to-nearest-bounded-cut
//! (#crown-cut-segment, `NY_CROWN_CUT_SEGMENT`).
//!
//! The gate makes each per-node backward stop at the most recent already-swept
//! "cut" node and concretize against that node's bound-box instead of
//! expanding the full prefix to the input. Two properties are checked on a
//! 7-node Linear/ReLU stack:
//!
//! 1. ENCLOSURE (the hard soundness gate): for >= 200 random inputs inside
//!    the box, every node's true activation (evaluated as a degenerate-box
//!    IBP forward, exact to ~ULP) lies INSIDE the gate-ON bounds, for both
//!    N=1 (cut at every node) and N=2.
//! 2. LOOSER-OR-EQUAL (the expected direction): the gate-ON bounds CONTAIN
//!    the gate-OFF (full-prefix) bounds element-wise, up to fp noise. A
//!    gate-ON bound that is ever TIGHTER than gate-OFF beyond fp noise means
//!    the cut concretized against something narrower than a valid enclosure —
//!    a soundness bug, not a win.
//!
//! Gate-OFF determinism is also pinned: two independent `"0"` runs must be
//! bit-identical, and an N=1 run must actually CHANGE some bound (otherwise
//! the gate never fired and the lever is dead plumbing).

use crate::layers::{LinearLayer, ReLULayer};
use crate::network::{GraphNetwork, GraphNode};
use crate::tests::with_serialized_env_vars;
use crate::Layer;
use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;
use std::collections::HashMap;

/// Deterministic xorshift64* stream in [-1, 1).
fn xorshift_unit(state: &mut u64) -> f32 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    let u = (*state >> 11) as f64 / (1u64 << 53) as f64;
    (u * 2.0 - 1.0) as f32
}

fn random_matrix(rows: usize, cols: usize, seed: u64, scale: f32) -> Array2<f32> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    Array2::from_shape_fn((rows, cols), |_| xorshift_unit(&mut state) * scale)
}

fn random_vector(len: usize, seed: u64, scale: f32) -> Array1<f32> {
    let mut state = seed.wrapping_mul(0xD1B5_4A32_D192_ED03) | 1;
    Array1::from_shape_fn(len, |_| xorshift_unit(&mut state) * scale)
}

/// 4 -> 8 -> 8 -> 6 -> 3 Linear/ReLU stack (7 nodes: l1 r1 l2 r2 l3 r3 l4).
/// Built FRESH per collection run: the CROWN-IBP collection cache is
/// per-graph-object and would otherwise serve run 1's map to run 2.
fn build_graph() -> GraphNetwork {
    let dims = [4usize, 8, 8, 6, 3];
    let mut graph = GraphNetwork::new();
    for i in 0..4 {
        let linear = LinearLayer::new(
            random_matrix(dims[i + 1], dims[i], 1000 + i as u64, 0.7),
            Some(random_vector(dims[i + 1], 2000 + i as u64, 0.3)),
        )
        .expect("linear layer");
        let lname = format!("l{}", i + 1);
        if i == 0 {
            graph.add_node(GraphNode::from_input(lname.clone(), Layer::Linear(linear)));
        } else {
            graph.add_node(GraphNode::new(
                lname.clone(),
                Layer::Linear(linear),
                vec![format!("r{}", i)],
            ));
        }
        if i < 3 {
            graph.add_node(GraphNode::new(
                format!("r{}", i + 1),
                Layer::ReLU(ReLULayer),
                vec![lname],
            ));
        }
    }
    graph.set_output("l4");
    graph
}

fn input_box() -> BoundedTensor {
    let mut state = 0xABCD_EF01_2345_6789u64;
    let center: Vec<f32> = (0..4).map(|_| xorshift_unit(&mut state) * 0.5).collect();
    let lower = ArrayD::from_shape_vec(IxDyn(&[4]), center.iter().map(|c| c - 0.3).collect())
        .expect("lower");
    let upper = ArrayD::from_shape_vec(IxDyn(&[4]), center.iter().map(|c| c + 0.3).collect())
        .expect("upper");
    BoundedTensor::new(lower, upper).expect("input box")
}

/// Run the CROWN-IBP sweep with an EXPLICITLY injected cut segment on a fresh
/// graph (`0` = gate-OFF full-prefix baseline). Uses the explicit-segment core
/// entry instead of the `NY_CROWN_CUT_SEGMENT` env so parallel test threads
/// never observe a mutated process environment mid-collection.
fn sweep_bounds(segment: usize) -> HashMap<String, BoundedTensor> {
    let graph = build_graph();
    let input = input_box();
    let ibp = graph.collect_node_bounds(&input).expect("IBP collection");
    graph
        .collect_crown_ibp_bounds_core_inner_with_cut_segment(
            &input, ibp, None, None, None, segment,
        )
        .expect("CROWN-IBP collection")
        .bounds
}

/// True per-node activations for >= `num_samples` random in-box points, as
/// degenerate-box IBP forwards (each interval encloses the true activation to
/// ~ULP, so containment of BOTH endpoints certifies containment of the truth).
fn sample_true_activations(num_samples: usize) -> Vec<HashMap<String, BoundedTensor>> {
    let graph = build_graph();
    let input = input_box();
    let flat = input.flatten();
    let (lower, upper) = (flat.lower().to_owned(), flat.upper().to_owned());
    let mut state = 0x1357_9BDF_2468_ACE0u64;
    (0..num_samples)
        .map(|_| {
            let point: Vec<f32> = (0..lower.len())
                .map(|j| {
                    let u = f32::midpoint(xorshift_unit(&mut state), 1.0); // [0, 1)
                    lower[j] + u * (upper[j] - lower[j])
                })
                .collect();
            let point_nd = ArrayD::from_shape_vec(IxDyn(&[point.len()]), point).expect("point");
            let point_box = BoundedTensor::new(point_nd.clone(), point_nd).expect("degenerate box");
            graph
                .collect_node_bounds(&point_box)
                .expect("point IBP forward")
        })
        .collect()
}

/// fp-noise allowance for comparing two independently computed f32 bounds.
fn tol(reference: f32) -> f32 {
    1e-4_f32.max(2e-5 * reference.abs())
}

/// ENCLOSURE: every sampled true activation lies inside `bounds` at every node.
fn assert_encloses_samples(
    bounds: &HashMap<String, BoundedTensor>,
    samples: &[HashMap<String, BoundedTensor>],
    label: &str,
) {
    for (node, node_bounds) in bounds {
        let nb = node_bounds.flatten();
        let (nb_l, nb_u) = (nb.lower(), nb.upper());
        for (sample_idx, sample) in samples.iter().enumerate() {
            let Some(point) = sample.get(node) else {
                panic!("{label}: node '{node}' missing from point forward");
            };
            let pf = point.flatten();
            let (p_l, p_u) = (pf.lower(), pf.upper());
            assert_eq!(nb_l.len(), p_l.len(), "{label}: '{node}' length drift");
            for i in 0..nb_l.len() {
                assert!(
                    nb_l[i] - tol(p_l[i]) <= p_l[i] && p_u[i] <= nb_u[i] + tol(p_u[i]),
                    "{label}: UNSOUND at node '{node}'[{i}] sample {sample_idx}: \
                     true in [{}, {}] but bounds [{}, {}]",
                    p_l[i],
                    p_u[i],
                    nb_l[i],
                    nb_u[i],
                );
            }
        }
    }
}

/// LOOSER-OR-EQUAL: `on` (gate-ON) must contain `off` (gate-OFF) element-wise.
fn assert_contains(
    on: &HashMap<String, BoundedTensor>,
    off: &HashMap<String, BoundedTensor>,
    label: &str,
) {
    assert_eq!(
        {
            let mut k: Vec<_> = on.keys().collect();
            k.sort();
            k
        },
        {
            let mut k: Vec<_> = off.keys().collect();
            k.sort();
            k
        },
        "{label}: node key sets differ",
    );
    for (node, on_bounds) in on {
        let off_bounds = &off[node];
        let onf = on_bounds.flatten();
        let offf = off_bounds.flatten();
        let (on_l, on_u) = (onf.lower(), onf.upper());
        let (off_l, off_u) = (offf.lower(), offf.upper());
        for i in 0..on_l.len() {
            assert!(
                on_l[i] <= off_l[i] + tol(off_l[i]) && on_u[i] >= off_u[i] - tol(off_u[i]),
                "{label}: gate-ON TIGHTER than gate-OFF at node '{node}'[{i}]: \
                 on [{}, {}] vs off [{}, {}] — cut bounds must only ever be looser",
                on_l[i],
                on_u[i],
                off_l[i],
                off_u[i],
            );
        }
    }
}

/// Whether any bound element differs beyond fp noise (gate actually fired).
fn any_bound_differs(
    a: &HashMap<String, BoundedTensor>,
    b: &HashMap<String, BoundedTensor>,
) -> bool {
    a.iter().any(|(node, ab)| {
        let Some(bb) = b.get(node) else { return true };
        let af = ab.flatten();
        let bf = bb.flatten();
        af.lower()
            .iter()
            .zip(bf.lower().iter())
            .chain(af.upper().iter().zip(bf.upper().iter()))
            .any(|(x, y)| (x - y).abs() > 1e-6)
    })
}

#[test]
fn crown_cut_segment_bounds_enclose_truth_and_contain_full_prefix() {
    let off = sweep_bounds(0);
    let cut_n1 = sweep_bounds(1);
    let cut_n2 = sweep_bounds(2);
    let cut_n6 = sweep_bounds(6);
    let samples = sample_true_activations(200);

    // Hard soundness gate: every true activation inside the cut bounds.
    assert_encloses_samples(&off, &samples, "gate-OFF");
    assert_encloses_samples(&cut_n1, &samples, "N=1");
    assert_encloses_samples(&cut_n2, &samples, "N=2");
    assert_encloses_samples(&cut_n6, &samples, "N=6");

    // Expected direction: cuts only ever loosen.
    assert_contains(&cut_n1, &off, "N=1 vs gate-OFF");
    assert_contains(&cut_n2, &off, "N=2 vs gate-OFF");
    assert_contains(&cut_n6, &off, "N=6 vs gate-OFF");
    // Longer segments keep MORE of the exact prefix, so they can only be
    // tighter-or-equal than shorter ones (both sides use identical swept
    // ancestor boxes on this chain).
    assert_contains(&cut_n1, &cut_n6, "N=1 vs N=6");

    // The gate must actually fire: N=1 (cut at EVERY node) on a 7-node stack
    // must move at least one deep bound away from the full-prefix result.
    assert!(
        any_bound_differs(&cut_n1, &off),
        "N=1 produced bit-identical bounds to gate-OFF — the cut never fired",
    );

    // Diagnostic summary (visible with --nocapture): per-mode mean width vs
    // the full-prefix baseline, for judging how much the cuts loosen.
    for (label, on) in [("N=1", &cut_n1), ("N=2", &cut_n2), ("N=6", &cut_n6)] {
        let (mut on_width, mut off_width) = (0.0f64, 0.0f64);
        for (node, ob) in on {
            let (of, xf) = (ob.flatten(), off[node].flatten());
            for i in 0..of.lower().len() {
                on_width += (of.upper()[i] - of.lower()[i]) as f64;
                off_width += (xf.upper()[i] - xf.lower()[i]) as f64;
            }
        }
        eprintln!(
            "crown_cut_segment oracle: {label} total width {on_width:.4} vs gate-OFF \
             {off_width:.4} (ratio {:.3})",
            on_width / off_width,
        );
    }
}

#[test]
fn crown_cut_segment_gate_off_is_deterministic() {
    let first = sweep_bounds(0);
    let second = sweep_bounds(0);
    for (node, a) in &first {
        let b = &second[node];
        let (af, bf) = (a.flatten(), b.flatten());
        assert!(
            af.lower()
                .iter()
                .zip(bf.lower().iter())
                .chain(af.upper().iter().zip(bf.upper().iter()))
                .all(|(x, y)| x.to_bits() == y.to_bits()),
            "gate-OFF runs disagree at node '{node}' (nondeterminism)",
        );
    }
}

/// The production sweep must actually READ `NY_CROWN_CUT_SEGMENT`: an
/// env-driven N=1 run must be bit-identical to an injected-segment N=1 run.
/// This is the only test that touches the env, and it holds the serialized
/// env lock for a single sub-second collection.
#[test]
fn crown_cut_segment_env_gate_reaches_the_sweep() {
    let injected = sweep_bounds(1);
    let via_env = with_serialized_env_vars(&[("NY_CROWN_CUT_SEGMENT", "1")], || {
        let graph = build_graph();
        let input = input_box();
        graph
            .collect_crown_ibp_bounds_dag(&input)
            .expect("CROWN-IBP collection")
    });
    assert_eq!(injected.len(), via_env.len(), "node key sets differ");
    for (node, a) in &injected {
        let b = &via_env[node];
        let (af, bf) = (a.flatten(), b.flatten());
        assert!(
            af.lower()
                .iter()
                .zip(bf.lower().iter())
                .chain(af.upper().iter().zip(bf.upper().iter()))
                .all(|(x, y)| x.to_bits() == y.to_bits()),
            "env-driven N=1 differs from injected N=1 at node '{node}' — the \
             NY_CROWN_CUT_SEGMENT plumbing is broken",
        );
    }
}
