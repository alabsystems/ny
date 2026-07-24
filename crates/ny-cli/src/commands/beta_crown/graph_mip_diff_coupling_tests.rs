// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0
//
// #rel-diff-coupling tests. The load-bearing SOUNDNESS test is
// `delta_is_a_valid_upper_bound_on_the_paired_difference`: across many concrete
// inputs the derived δ must dominate the ACTUAL |f_node - g_node| at EVERY
// paired neuron — an under-tight δ would let a coupling row cut off a real
// point (false certified-UNSAT). `known_violated_pair_is_never_falsely_closed`
// is the end-to-end guard: a genuinely-different pair keeps a violating point
// FEASIBLE under the coupling rows.

use std::collections::HashMap;

use ndarray::{arr1, arr2, ArrayD, IxDyn};
use ny_core::Bound;
use ny_propagate::layers::{AddConstantLayer, LinearLayer, ReLULayer};
use ny_propagate::{build_difference_network, GraphNetwork, GraphNode, Layer, NETWORK_INPUT};
use ny_tensor::BoundedTensor;

use super::{
    apply_diff_coupling, attach_diff_coupling, compute_difference_bounds, detect_prefixes,
};

/// A tiny 2-input → 3-hidden(ReLU) → 2-output MLP with bias folded as a separate
/// `AddConstant` node (mirrors the ACAS ONNX shape: Linear = matmul, AddConstant
/// = bias). Weights/bias are caller-supplied so a partner net can perturb them.
fn mlp(w1: [[f32; 2]; 3], b1: [f32; 3], w2: [[f32; 3]; 2], b2: [f32; 2]) -> GraphNetwork {
    let mut g = GraphNetwork::new();
    g.try_add_node(GraphNode::from_input(
        "z1",
        Layer::Linear(LinearLayer::new(arr2(&w1), None).unwrap()),
    ))
    .unwrap();
    g.try_add_node(GraphNode::new(
        "z1b",
        Layer::AddConstant(AddConstantLayer::new(arr1(&b1).into_dyn())),
        vec!["z1".into()],
    ))
    .unwrap();
    g.try_add_node(GraphNode::new(
        "h1",
        Layer::ReLU(ReLULayer),
        vec!["z1b".into()],
    ))
    .unwrap();
    g.try_add_node(GraphNode::new(
        "z2",
        Layer::Linear(LinearLayer::new(arr2(&w2), None).unwrap()),
        vec!["h1".into()],
    ))
    .unwrap();
    g.try_add_node(GraphNode::new(
        "out",
        Layer::AddConstant(AddConstantLayer::new(arr1(&b2).into_dyn())),
        vec!["z2".into()],
    ))
    .unwrap();
    g.set_output("out");
    g.set_declared_shape(NETWORK_INPUT, vec![2]);
    g
}

fn base_f() -> GraphNetwork {
    mlp(
        [[0.7, -0.3], [0.2, 0.9], [-0.5, 0.4]],
        [0.1, -0.2, 0.05],
        [[0.6, -0.1, 0.8], [0.3, 0.7, -0.4]],
        [0.02, -0.03],
    )
}

/// Near-equal partner (small per-weight perturbation) — the isomorphic case.
fn perturbed_g(eps: f32) -> GraphNetwork {
    mlp(
        [
            [0.7 + eps, -0.3 - eps],
            [0.2 - eps, 0.9 + eps],
            [-0.5 + eps, 0.4 - eps],
        ],
        [0.1 - eps, -0.2 + eps, 0.05 + eps],
        [
            [0.6 - eps, -0.1 + eps, 0.8 - eps],
            [0.3 + eps, 0.7 - eps, -0.4 + eps],
        ],
        [0.02 + eps, -0.03 - eps],
    )
}

fn input_box() -> Vec<Bound> {
    vec![Bound::new(-1.0, 1.0), Bound::new(-0.5, 1.5)]
}

/// Faithful concrete forward of a single point through a net.
fn forward_point(graph: &GraphNetwork, values: &[f32]) -> Vec<f32> {
    let arr = ArrayD::from_shape_vec(IxDyn(&[values.len()]), values.to_vec()).unwrap();
    let pt = BoundedTensor::concrete(arr).unwrap();
    let out = graph.propagate_concrete_point(&pt, None, None).unwrap();
    out.lower().iter().copied().collect()
}

/// Stamp 1-D declared shapes on every node from its box (mirrors the finisher,
/// which the encoder requires for exact index/broadcast math).
fn stamp_shapes(diff: &mut GraphNetwork, flat: &HashMap<String, Vec<Bound>>) {
    for (name, v) in flat {
        if diff.declared_shape(name).is_none() {
            diff.set_declared_shape(name.clone(), vec![v.len()]);
        }
    }
}

/// CROWN-IBP per-node boxes over the input box, flattened — the boxes the
/// finisher feeds the encoder (and the coupling's magnitude bounds).
fn flat_bounds_of(diff: &GraphNetwork, input_bounds: &[Bound]) -> HashMap<String, Vec<Bound>> {
    let input = ny_propagate::Verifier::bounds_to_tensor(input_bounds, None).unwrap();
    let nb = diff.collect_node_bounds(&input).unwrap();
    let mut out = HashMap::new();
    for (name, bt) in &nb {
        let flat = bt.flatten();
        let lo = flat.lower();
        let hi = flat.upper();
        let v: Vec<Bound> = (0..lo.len())
            .map(|i| Bound::new(lo[[i]], hi[[i]]))
            .collect();
        out.insert(name.clone(), v);
    }
    out
}

#[test]
fn prefixes_detected_on_stitched_diff_net() {
    let diff = build_difference_network(&base_f(), &perturbed_g(0.01)).unwrap();
    let (pa, pb) = detect_prefixes(&diff).expect("stitched diff net has prefixes");
    assert_eq!((pa, pb), ("a_", "b_"));
    // A non-stitched net has no prefix pair.
    assert!(detect_prefixes(&base_f()).is_none());
}

/// Prefix-shaped node sets are not authority for a paired tower.  If the two
/// sides have different internal wiring, a bound propagated along the `a_`
/// edge is not necessarily valid for the `b_` operand and coupling must fail
/// open.
#[test]
fn prefix_shaped_but_topology_mismatched_fails_open() {
    let mut graph = GraphNetwork::new();
    graph
        .try_add_node(GraphNode::from_input("a_src", Layer::ReLU(ReLULayer)))
        .unwrap();
    graph
        .try_add_node(GraphNode::from_input("b_src", Layer::ReLU(ReLULayer)))
        .unwrap();
    graph
        .try_add_node(GraphNode::new(
            "a_out",
            Layer::ReLU(ReLULayer),
            vec!["a_src".into()],
        ))
        .unwrap();
    // Same suffix set, but b_out bypasses b_src instead of mirroring a_out's
    // tower-local edge.
    graph
        .try_add_node(GraphNode::from_input("b_out", Layer::ReLU(ReLULayer)))
        .unwrap();

    assert!(
        detect_prefixes(&graph).is_none(),
        "mismatched tower topology must disable all coupling"
    );
}

/// LOAD-BEARING SOUNDNESS TEST. Derive δ, then at many concrete inputs confirm
/// δ_X dominates the TRUE |f_X - g_X| at every paired neuron. If this ever fails
/// the coupling could cut off a real point → a false certified-UNSAT.
#[test]
fn delta_is_a_valid_upper_bound_on_the_paired_difference() {
    let f = base_f();
    let g = perturbed_g(0.03);
    let diff = build_difference_network(&f, &g).unwrap();
    let input_bounds = input_box();
    let flat = flat_bounds_of(&diff, &input_bounds);
    let (pa, pb) = detect_prefixes(&diff).unwrap();
    let diffb = compute_difference_bounds(&diff, &input_bounds, &flat, pa, pb).unwrap();

    // Deterministic LCG sampler over the input box, corners included.
    let mut rng: u64 = 0x243f_6a88_85a3_08d3;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        ((rng >> 40) as f64) / ((1u64 << 24) as f64)
    };
    let mut checked_neurons = 0usize;
    for s in 0..2000 {
        let x: Vec<f32> = input_bounds
            .iter()
            .enumerate()
            .map(|(i, b)| {
                // Mix interior samples with exact corners.
                if s < 4 {
                    if (s >> i) & 1 == 0 {
                        b.lower()
                    } else {
                        b.upper()
                    }
                } else {
                    let t = next() as f32;
                    b.lower() + t * (b.upper() - b.lower())
                }
            })
            .collect();
        let pt =
            BoundedTensor::concrete(ArrayD::from_shape_vec(IxDyn(&[x.len()]), x).unwrap()).unwrap();
        let acts = diff.collect_node_activations_pointwise(&pt, None).unwrap();
        for (suffix, delta) in &diffb {
            let a_name = format!("{pa}{suffix}");
            let b_name = format!("{pb}{suffix}");
            let (Some(a_bt), Some(b_bt)) = (acts.get(&a_name), acts.get(&b_name)) else {
                continue;
            };
            let a_lo = a_bt.flatten();
            let b_lo = b_bt.flatten();
            let av = a_lo.lower();
            let bv = b_lo.lower();
            if av.len() != delta.len() || bv.len() != delta.len() {
                continue;
            }
            for i in 0..delta.len() {
                if !delta[i].is_finite() {
                    continue;
                }
                let actual = (f64::from(av[[i]]) - f64::from(bv[[i]])).abs();
                assert!(
                    actual <= delta[i] + 1e-6,
                    "δ under-tight at node {suffix}[{i}] on sample {s}: \
                     actual |a-b|={actual:.6} > δ={:.6}",
                    delta[i]
                );
                checked_neurons += 1;
            }
        }
    }
    assert!(checked_neurons > 0, "test exercised no neurons");
}

/// The isomorphic difference bound must be SMALL relative to the individual
/// activation ranges — this is the property that makes the coupling a real
/// lever. With eps=0 (identical nets) the output δ is ~0; a small eps keeps it
/// bounded well below the activation magnitudes.
#[test]
fn identical_nets_have_near_zero_difference_bound() {
    let f = base_f();
    let diff = build_difference_network(&f, &f).unwrap();
    let input_bounds = input_box();
    let flat = flat_bounds_of(&diff, &input_bounds);
    let (pa, pb) = detect_prefixes(&diff).unwrap();
    let diffb = compute_difference_bounds(&diff, &input_bounds, &flat, pa, pb).unwrap();
    for (suffix, delta) in &diffb {
        for (i, d) in delta.iter().enumerate() {
            assert!(
                *d <= 1e-4,
                "identical nets must have ~0 coupling bound, node {suffix}[{i}] = {d}"
            );
        }
    }
}

/// END-TO-END SOUNDNESS. A genuinely-DIFFERENT pair (large perturbation) must
/// NOT be falsely closable: the derived δ dominates the real output difference,
/// so a concrete violating point still satisfies every coupling row (the row
/// band is >= the real difference). We assert δ_out >= the sampled max |f-g| —
/// i.e. the coupling never certifies the pair equal.
#[test]
fn known_violated_pair_is_never_falsely_closed() {
    let f = base_f();
    // A big perturbation: f and g differ well beyond any tight band.
    let g = perturbed_g(0.5);
    let diff = build_difference_network(&f, &g).unwrap();
    let input_bounds = input_box();
    let flat = flat_bounds_of(&diff, &input_bounds);
    let (pa, pb) = detect_prefixes(&diff).unwrap();
    let diffb = compute_difference_bounds(&diff, &input_bounds, &flat, pa, pb).unwrap();

    // Output operand's δ.
    let out_suffix = diff
        .node("diff_output")
        .and_then(|n| n.inputs().first().cloned())
        .and_then(|o| o.strip_prefix(pa).map(str::to_string))
        .unwrap();
    let out_delta = diffb.get(&out_suffix).unwrap();

    // Largest actual output difference across a dense grid must be <= δ_out.
    let mut worst = vec![0.0f64; out_delta.len()];
    for gx in 0..21 {
        for gy in 0..21 {
            let x = vec![-1.0 + 2.0 * gx as f32 / 20.0, -0.5 + 2.0 * gy as f32 / 20.0];
            let fo = forward_point(&f, &x);
            let go = forward_point(&g, &x);
            for i in 0..worst.len() {
                worst[i] = worst[i].max((f64::from(fo[i]) - f64::from(go[i])).abs());
            }
        }
    }
    for i in 0..out_delta.len() {
        assert!(
            worst[i] <= out_delta[i] + 1e-6,
            "δ_out[{i}]={:.4} must dominate the real max diff {:.4} (else false-closable)",
            out_delta[i],
            worst[i]
        );
        // And the real difference genuinely EXCEEDS a tight band — this pair is
        // a real violation, so a sound coupling must keep δ_out large.
        assert!(
            out_delta[i] >= worst[i],
            "coupling band must not undercut a real violation"
        );
    }
    // Sanity: this really is a violated pair (some output differs a lot).
    assert!(worst.iter().cloned().fold(0.0, f64::max) > 0.1);
}

/// The coupling rows are added to the problem (row count grows) and only ever
/// restrict — every emitted row is a two-sided band on a paired difference.
#[test]
fn coupling_rows_are_emitted_and_finite() {
    let f = base_f();
    let g = perturbed_g(0.02);
    let mut diff = build_difference_network(&f, &g).unwrap();
    let input_bounds = input_box();
    let flat = flat_bounds_of(&diff, &input_bounds);
    stamp_shapes(&mut diff, &flat);
    let mut enc = super::super::graph_mip::encode_graph(&diff, &input_bounds, &flat).unwrap();
    let rows_before = enc.problem.num_rows();
    let (added, out_delta) = attach_diff_coupling(&mut enc, &diff, &input_bounds, &flat);
    assert!(added > 0, "expected coupling rows on a paired MLP");
    assert_eq!(enc.problem.num_rows(), rows_before + added);
    let od = out_delta.expect("output delta present");
    assert!(od.iter().all(|d| d.is_finite()));
}

/// #rel-diff-coupling MEASUREMENT (ignored by default — needs the benchmark
/// ONNX; run with `--ignored --nocapture`). On the REAL instance_0 diff net,
/// compare the LP-relaxation (triangle) bound on `diff_output` WITHOUT vs WITH
/// the difference-coupling rows, and against the ±band. This is the honest test
/// of whether the coupling lets the output band propagate back through the
/// relaxation — the box widths do NOT shrink (coupling adds rows, not tighter
/// per-neuron boxes), so the lever must show up in the OUTPUT LP bound.
#[cfg(feature = "mip")]
#[test]
#[ignore = "requires benchmark ONNX; run explicitly"]
fn measure_output_lp_shrink_on_real_instance0() {
    use std::path::PathBuf;
    use std::time::{Duration, Instant};

    let base = PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../.."))
        .join("benchmarks/vnncomp2026/benchmarks/isomorphic_acasxu_2026/2.0");
    if !base.is_dir() {
        eprintln!("benchmarks absent; skipping");
        return;
    }
    let f = base.join("onnx/original/ACASXU_run2a_2_4_batch_2000.onnx");
    let g = base.join("onnx/perturbed/ACASXU_run2a_2_4_batch_2000_perturbed_0.onnx");
    let vnnlib = base.join("vnnlib/instance_0.vnnlib");
    let graph_f = crate::commands::vnncomp::load_graph_network(&f).unwrap();
    let graph_g = crate::commands::vnncomp::load_graph_network(&g).unwrap();
    let mut diff = build_difference_network(&graph_f, &graph_g).unwrap();

    let spec = ny_onnx::vnnlib::load_vnnlib(&vnnlib).unwrap();
    let dual = spec.dual_network.expect("dual");
    let input_bounds = crate::commands::vnncomp::bounds_from_f64(&dual.f_input_bounds).unwrap();
    let input = ny_propagate::Verifier::bounds_to_tensor(&input_bounds, None).unwrap();

    // CROWN-IBP per-node boxes (the finisher's big-M source).
    let nb = diff
        .collect_crown_ibp_bounds_dag_with_engine(&input, None)
        .unwrap();
    let mut flat: HashMap<String, Vec<Bound>> = HashMap::new();
    for (name, bt) in &nb {
        let fl = bt.flatten();
        let (lo, hi) = (fl.lower().clone(), fl.upper().clone());
        flat.insert(
            name.clone(),
            (0..lo.len())
                .map(|i| Bound::new(lo[[i]], hi[[i]]))
                .collect(),
        );
    }
    // Declared shapes (the finisher stamps these onto a clone before encode).
    if diff.declared_shape(NETWORK_INPUT).is_none() {
        diff.set_declared_shape(NETWORK_INPUT, input.shape().to_vec());
    }
    for (name, bt) in &nb {
        if diff.declared_shape(name).is_none() {
            diff.set_declared_shape(name.clone(), bt.shape().to_vec());
        }
    }
    let (wmax, _, _, _) = {
        let mut widths: Vec<f64> = flat
            .values()
            .flat_map(|v| v.iter().map(|b| f64::from(b.upper() - b.lower())))
            .filter(|w| w.is_finite())
            .collect();
        widths.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = widths.len();
        (
            widths.iter().copied().fold(0.0_f64, f64::max),
            widths[n / 2],
            0,
            n,
        )
    };
    eprintln!("instance0 CROWN-IBP max box width = {wmax:.1}");

    let out_lp = |with_coupling: bool| -> (Vec<(f64, f64)>, bool, usize) {
        let mut enc = super::super::graph_mip::encode_graph(&diff, &input_bounds, &flat).unwrap();
        let mut added = 0;
        if with_coupling {
            let (n, _) = attach_diff_coupling(&mut enc, &diff, &input_bounds, &flat);
            added = n;
        }
        let targets = enc.output_vars.clone();
        let chunk = targets.len().max(1);
        let deadline = Instant::now() + Duration::from_secs(45);
        let r = ny_mip::obbt_relaxation_bounds(
            &enc.problem,
            &targets,
            1,
            Duration::from_secs(20),
            deadline,
            chunk,
        )
        .unwrap();
        eprintln!(
            "  [obbt] with_coupling={with_coupling} rounds={} tightened={} infeasible={}",
            r.rounds, r.tightened, r.infeasible
        );
        (r.bounds, r.infeasible, added)
    };

    // δ statistics: how loose is the interval-propagated difference bound?
    let (pa, pb) = detect_prefixes(&diff).unwrap();
    let diffb = compute_difference_bounds(&diff, &input_bounds, &flat, pa, pb).unwrap();
    let mut dmax = 0.0f64;
    let mut d_finite = 0usize;
    let mut d_inf = 0usize;
    for delta in diffb.values() {
        for &d in delta {
            if d.is_finite() {
                dmax = dmax.max(d);
                d_finite += 1;
            } else {
                d_inf += 1;
            }
        }
    }
    let out_suffix = diff
        .node("diff_output")
        .and_then(|n| n.inputs().first().cloned())
        .and_then(|o| o.strip_prefix(pa).map(str::to_string))
        .unwrap();
    let out_delta = diffb.get(&out_suffix).unwrap();
    eprintln!(
        "δ stats: max δ over all paired neurons = {dmax:.2} ({d_finite} finite, {d_inf} inf); \
         output-layer δ = {out_delta:?}"
    );

    // MEASURED FINDING (2026-07-22): the coupling MECHANISM is proven to bind
    // (see `coupling_rows_bind_and_rigorous_delta_never_falsely_infeasible`), but
    // the RIGOROUS interval-propagated δ is far too loose to shrink the output
    // LP: δ_out ≈ 3e4 while the LP's own diff_output range is ≈ 3.5e3 — the
    // output-layer coupling band is ~10x WIDER than what the relaxation already
    // proves, so it is slack. The cause is structural: interval-propagating the
    // difference amplifies by ‖|W|‖₁ each layer (this is IBP-on-the-difference),
    // exactly the looseness α-CROWN / the LP triangle relaxation already beats by
    // LP duality. A biting δ would need CROWN-style (linear) DIFFERENCE bounds —
    // i.e. α-CROWN on the diff net, which the encoding already carries.
    let (base_b, _, _) = out_lp(false);
    let (coup_b, _, added) = out_lp(true);
    let band = 0.05_f64;
    eprintln!("coupling rows added = {added}");

    let mut base_closed = 0usize;
    let mut coup_closed = 0usize;
    for i in 0..base_b.len() {
        let (bl, bu) = base_b[i];
        let (cl, cu) = coup_b[i];
        let base_w = bu - bl;
        let coup_w = cu - cl;
        eprintln!(
            "diff_output[{i}]: baseline LP=[{bl:.4},{bu:.4}] w={base_w:.4} | \
             coupled LP=[{cl:.4},{cu:.4}] w={coup_w:.4} | band=±{band}"
        );
        if bl >= -band && bu <= band {
            base_closed += 1;
        }
        if cl >= -band && cu <= band {
            coup_closed += 1;
        }
    }
    eprintln!(
        "OUTPUTS within ±{band}: baseline {base_closed}/{} | coupled {coup_closed}/{}",
        base_b.len(),
        coup_b.len()
    );
    eprintln!(
        "CLOSURE (all outputs within band): baseline={} coupled={}",
        base_closed == base_b.len(),
        coup_closed == coup_b.len()
    );
}

/// MECHANISM + SOUNDNESS-HAZARD PROOF (fast — tiny net, LP solves instantly).
///
/// (1) The uncoupled relaxation is FEASIBLE and OBBT does real work (tightens the
///     output columns). (2) Adding coupling rows with an intentionally TOO-TIGHT
///     δ (below the real per-neuron difference) makes the WHOLE relaxation
///     INFEASIBLE — proving the coupling rows are genuinely enforced by the LP
///     (a row that was ignored could never flip feasibility). (3) This is exactly
///     the soundness hazard: an under-tight δ certifies a FALSE property (spurious
///     infeasible ⇒ false UNSAT), which is why our derived δ is OUTWARD-rounded
///     and validated by `delta_is_a_valid_upper_bound_on_the_paired_difference`.
///     (4) Re-deriving δ RIGOROUSLY keeps the model FEASIBLE — never a false
///     infeasible.
#[cfg(feature = "mip")]
#[test]
fn coupling_rows_bind_and_rigorous_delta_never_falsely_infeasible() {
    use std::time::{Duration, Instant};
    let f = base_f();
    let g = perturbed_g(0.5); // wide real difference
    let mut diff = build_difference_network(&f, &g).unwrap();
    let input_bounds = input_box();
    let flat = flat_bounds_of(&diff, &input_bounds);
    stamp_shapes(&mut diff, &flat);
    let (pa, pb) = detect_prefixes(&diff).unwrap();

    // Uniform-δ coupling variant (None = no coupling).
    let run = |delta: Option<f64>| -> (usize, bool) {
        let mut enc = super::super::graph_mip::encode_graph(&diff, &input_bounds, &flat).unwrap();
        if let Some(d) = delta {
            let diffb = compute_difference_bounds(&diff, &input_bounds, &flat, pa, pb).unwrap();
            let mut tmap: HashMap<String, Vec<f64>> = HashMap::new();
            for (s, v) in &diffb {
                tmap.insert(s.clone(), vec![d; v.len()]);
            }
            apply_diff_coupling(&mut enc, &tmap, pa, pb);
        }
        let targets = enc.output_vars.clone();
        let deadline = Instant::now() + Duration::from_secs(20);
        let r = ny_mip::obbt_relaxation_bounds(
            &enc.problem,
            &targets,
            2,
            Duration::from_secs(5),
            deadline,
            targets.len().max(1),
        )
        .unwrap();
        (r.tightened, r.infeasible)
    };

    // (1) uncoupled: feasible + real OBBT work.
    let (t_none, inf_none) = run(None);
    assert!(!inf_none, "uncoupled relaxation must be feasible");
    assert!(t_none > 0, "uncoupled OBBT should tighten the outputs");

    // (2)+(3) too-tight δ (below the real diff) ⇒ INFEASIBLE ⇒ rows bind, and a
    // false certified-UNSAT would follow — the soundness hazard our derivation
    // precludes.
    let (_t_tight, inf_tight) = run(Some(1e-4));
    assert!(
        inf_tight,
        "an under-tight δ must over-constrain the relaxation to infeasible \
         (proving the rows bind AND illustrating the false-UNSAT hazard)"
    );

    // (4) the RIGOROUS δ (outward, valid) must keep the model FEASIBLE — it is a
    // sound over-approximation, never a false infeasible.
    let mut enc = super::super::graph_mip::encode_graph(&diff, &input_bounds, &flat).unwrap();
    let (added, _) = attach_diff_coupling(&mut enc, &diff, &input_bounds, &flat);
    assert!(added > 0);
    let targets = enc.output_vars.clone();
    let r = ny_mip::obbt_relaxation_bounds(
        &enc.problem,
        &targets,
        2,
        Duration::from_secs(5),
        Instant::now() + Duration::from_secs(20),
        targets.len().max(1),
    )
    .unwrap();
    assert!(
        !r.infeasible,
        "rigorous outward δ must never make the relaxation falsely infeasible"
    );
}

/// `apply_diff_coupling` skips infinite δ (no row) and mismatched shapes.
#[test]
fn infinite_delta_emits_no_row() {
    let f = base_f();
    let g = perturbed_g(0.02);
    let mut diff = build_difference_network(&f, &g).unwrap();
    let input_bounds = input_box();
    let flat = flat_bounds_of(&diff, &input_bounds);
    stamp_shapes(&mut diff, &flat);
    let mut enc = super::super::graph_mip::encode_graph(&diff, &input_bounds, &flat).unwrap();
    let rows_before = enc.problem.num_rows();
    let mut diffb: HashMap<String, Vec<f64>> = HashMap::new();
    diffb.insert("out".into(), vec![f64::INFINITY, f64::INFINITY]);
    let added = apply_diff_coupling(&mut enc, &diffb, "a_", "b_");
    assert_eq!(added, 0);
    assert_eq!(enc.problem.num_rows(), rows_before);
}
