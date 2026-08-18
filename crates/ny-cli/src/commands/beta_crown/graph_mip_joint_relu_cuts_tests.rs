// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0
//
// #rel-joint-relu-cuts tests. The load-bearing SOUNDNESS test is
// `every_joint_cut_is_valid_over_the_exact_paired_relu`: for a broad sweep of
// boxes / δ, EVERY emitted envelope cut must hold at EVERY exact paired-ReLU
// point (a, b, relu(a), relu(b)) with (a, b) ∈ P. A cut that failed there would
// cut off a real point → false certified-UNSAT. `diagonal_cut_binds_near_the_
// diagonal` is the MECHANISM proof: the difference envelope forces y_f ≈ y_g
// when z_f ≈ z_g, strictly tighter than the product of two triangles.

use std::collections::HashMap;

#[cfg(test)]
use ndarray::{arr1, arr2, ArrayD, IxDyn};
use ny_core::Bound;
#[cfg(test)]
use ny_mip::ir::Col;
#[cfg(test)]
use ny_propagate::layers::{AddConstantLayer, LinearLayer, ReLULayer};
use ny_propagate::{build_difference_network, NETWORK_INPUT};
#[cfg(test)]
use ny_propagate::{GraphNetwork, GraphNode, Layer};
#[cfg(test)]
use ny_tensor::BoundedTensor;

use super::attach_joint_relu_cuts;
#[cfg(test)]
use super::{neuron_joint_cuts, CutKind, JointCut};

/// Evaluate a cut's row body `Σ terms · vals` given a value per column index.
#[cfg(test)]
fn eval_cut(cut: &JointCut, vals: &HashMap<usize, f64>) -> f64 {
    cut.terms
        .iter()
        .map(|&(c, w)| w * vals.get(&c.0).copied().unwrap_or(0.0))
        .sum()
}

#[cfg(test)]
fn relu(x: f64) -> f64 {
    x.max(0.0)
}

/// SOUNDNESS: every emitted cut is a VALID inequality of the exact paired-ReLU
/// set over P = [lf,uf]×[lg,ug] ∩ {|a−b| ≤ δ}. Sweep parameters; for each, sample
/// a dense grid of (a, b) ∈ P and require `lb ≤ body ≤ ub` for every cut at the
/// EXACT lift (yf=relu(a), yg=relu(b)). Columns are fixed sentinels: zf=0, zg=1,
/// yf=2, yg=3.
#[test]
fn every_joint_cut_is_valid_over_the_exact_paired_relu() {
    let (zf, zg, yf, yg) = (Col(0), Col(1), Col(2), Col(3));
    let params = [
        (-2.0, 3.0, -2.5, 2.0, 0.5),
        (-1.0, 1.0, -1.0, 1.0, 0.1),
        (-5.0, 4.0, -4.0, 6.0, 2.0),
        (-3.0, 1.0, -1.0, 3.0, 3.0),
        (-0.5, 0.5, -0.5, 0.5, 0.9),
        (-10.0, 2.0, -2.0, 10.0, 0.3),
        (-2.0, 2.0, -2.0, 2.0, 100.0), // δ huge: P ≈ full box (degenerate coupling)
        (-2.0, 2.0, -2.0, 2.0, 0.0),   // δ = 0: forced diagonal a = b
    ];
    let mut total_cuts = 0usize;
    for &(lf, uf, lg, ug, delta) in &params {
        let (cuts, _) = neuron_joint_cuts(zf, zg, yf, yg, lf, uf, lg, ug, delta, true);
        assert!(
            !cuts.is_empty(),
            "expected cuts for {lf},{uf},{lg},{ug},{delta}"
        );
        total_cuts += cuts.len();
        let steps = 60;
        for ia in 0..=steps {
            let a = lf + (uf - lf) * ia as f64 / steps as f64;
            for ib in 0..=steps {
                let b = lg + (ug - lg) * ib as f64 / steps as f64;
                if (a - b).abs() > delta {
                    continue; // outside P — cuts need not hold here
                }
                let vals: HashMap<usize, f64> =
                    [(zf.0, a), (zg.0, b), (yf.0, relu(a)), (yg.0, relu(b))]
                        .into_iter()
                        .collect();
                for cut in &cuts {
                    let body = eval_cut(cut, &vals);
                    // A tiny numeric slack for the grid eval itself; the cut's own
                    // outward rounding already covers derivation error.
                    let slack = 1e-9 * (1.0 + body.abs());
                    assert!(
                        body <= cut.ub + slack && body >= cut.lb - slack,
                        "cut {:?} VIOLATED at exact point a={a} b={b}: body={body} not in \
                         [{}, {}] (params {lf},{uf},{lg},{ug},{delta})",
                        cut.kind,
                        cut.lb,
                        cut.ub,
                    );
                }
            }
        }
    }
    assert!(total_cuts > 0);
}

/// A geometric deduplication tolerance must never erase the off-diagonal
/// vertices when the box scale is enormous relative to the coupling width.
/// Doing so collapses the inferred hull to `y_f = y_g` and can exclude a real
/// pair whose positive pre-activations differ by exactly `delta`.
#[test]
fn large_scale_tiny_delta_preserves_sound_off_diagonal_vertices() {
    let (zf, zg, yf, yg) = (Col(0), Col(1), Col(2), Col(3));
    let delta = 1.0e-6;
    let (cuts, _) = neuron_joint_cuts(zf, zg, yf, yg, -1.0e9, 1.0e9, -1.0e9, 1.0e9, delta, false);
    assert!(!cuts.is_empty());

    let a = 0.5;
    let b = a - delta;
    assert!((a - b).abs() <= delta);
    let vals: HashMap<usize, f64> = [(zf.0, a), (zg.0, b), (yf.0, relu(a)), (yg.0, relu(b))]
        .into_iter()
        .collect();

    for cut in &cuts {
        let body = eval_cut(cut, &vals);
        assert!(
            body <= cut.ub + 1.0e-12 && body >= cut.lb - 1.0e-12,
            "cut {:?} excluded an exact tiny-delta point: body={body} not in [{}, {}]",
            cut.kind,
            cut.lb,
            cut.ub,
        );
    }
}

/// MECHANISM: near the diagonal (a ≈ b, both active) the DIFFERENCE envelope
/// forces y_f − y_g ≈ z_f − z_g (magnitude ≤ ~δ), whereas the two independent
/// triangles allow y_f − y_g up to ≈ upper_tri_f(a) − 0. Assert that at least one
/// emitted Difference cut is an UPPER bound on (y_f − y_g) that, at the diagonal
/// point a = b = uf/2, is much tighter than the product-of-triangles bound.
#[test]
fn diagonal_cut_binds_near_the_diagonal() {
    let (zf, zg, yf, yg) = (Col(0), Col(1), Col(2), Col(3));
    let (lf, uf, lg, ug, delta) = (-2.0, 2.0, -2.0, 2.0, 0.2);
    let (cuts, tighten) = neuron_joint_cuts(zf, zg, yf, yg, lf, uf, lg, ug, delta, false);
    assert!(
        tighten > 0.5,
        "diagonal cut should tighten vs triangles: {tighten}"
    );

    // Evaluate the tightest Difference UPPER cut on (yf − yg) at the diagonal
    // point a = b = 1 (both active): true yf − yg = 0.
    let a = 1.0;
    let b = 1.0;
    let vals: HashMap<usize, f64> = [(zf.0, a), (zg.0, b), (yf.0, relu(a)), (yg.0, relu(b))]
        .into_iter()
        .collect();
    // A Difference cut with ub finite bounds cf·yf+cg·yg (cf=1,cg=−1) from above:
    //   body = (yf − yg) − α zf − β zg ≤ γ  ⇒  yf − yg ≤ γ + α zf + β zg.
    let mut best_ub_on_diff = f64::INFINITY;
    for cut in &cuts {
        if cut.kind != CutKind::Difference || !cut.ub.is_finite() {
            continue;
        }
        // reconstruct the implied bound on (yf − yg) at this point:
        // body = (yf−yg) + (coeff on zf)·zf + (coeff on zg)·zg ≤ ub
        // ⇒ yf−yg ≤ ub − (coeff_zf)·zf − (coeff_zg)·zg.
        let coeff_zf = cut
            .terms
            .iter()
            .find(|&&(c, _)| c == zf)
            .map(|&(_, w)| w)
            .unwrap_or(0.0);
        let coeff_zg = cut
            .terms
            .iter()
            .find(|&&(c, _)| c == zg)
            .map(|&(_, w)| w)
            .unwrap_or(0.0);
        let bound = cut.ub - coeff_zf * a - coeff_zg * b;
        best_ub_on_diff = best_ub_on_diff.min(bound);
        // sanity: the cut holds at the exact point.
        assert!(eval_cut(cut, &vals) <= cut.ub + 1e-9);
    }
    // The product-of-triangles upper bound on (yf − yg) at a=b=1:
    //   uf(a−lf)/(uf−lf) − max(0,b) = 2·(1+2)/4 − 1 = 1.5 − 1 = 0.5.
    let tri_bound = uf * (a - lf) / (uf - lf) - b.max(0.0);
    assert!(
        best_ub_on_diff < tri_bound - 0.1,
        "diagonal Difference cut ub {best_ub_on_diff} should beat triangle bound {tri_bound}"
    );
    // And it must remain a VALID (>= 0) upper bound on the true value 0.
    assert!(
        best_ub_on_diff >= -1e-9,
        "cut must not cut off the true point"
    );
}

/// A stable or degenerate neuron gets NO envelope cut (only the harmless δ-box
/// row is meaningful, and even that is skipped for a degenerate box).
#[test]
fn degenerate_box_emits_no_cuts() {
    let (zf, zg, yf, yg) = (Col(0), Col(1), Col(2), Col(3));
    // lf >= uf → degenerate, no cuts.
    let (cuts, _) = neuron_joint_cuts(zf, zg, yf, yg, 1.0, 1.0, -1.0, 1.0, 0.5, true);
    assert!(cuts.is_empty());
    // non-finite δ → no cuts.
    let (cuts2, _) = neuron_joint_cuts(zf, zg, yf, yg, -1.0, 1.0, -1.0, 1.0, f64::INFINITY, true);
    assert!(cuts2.is_empty());
}

// ── end-to-end on a stitched isomorphic diff net ─────────────────────────────

#[cfg(test)]
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

#[cfg(test)]
fn base_f() -> GraphNetwork {
    mlp(
        [[0.7, -0.3], [0.2, 0.9], [-0.5, 0.4]],
        [0.1, -0.2, 0.05],
        [[0.6, -0.1, 0.8], [0.3, 0.7, -0.4]],
        [0.02, -0.03],
    )
}

#[cfg(test)]
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

#[cfg(test)]
fn input_box() -> Vec<Bound> {
    vec![Bound::new(-1.0, 1.0), Bound::new(-0.5, 1.5)]
}

#[cfg(test)]
fn flat_bounds_of(diff: &GraphNetwork, input_bounds: &[Bound]) -> HashMap<String, Vec<Bound>> {
    let input = ny_propagate::Verifier::bounds_to_tensor(input_bounds, None).unwrap();
    let nb = diff.collect_node_bounds(&input).unwrap();
    let mut out = HashMap::new();
    for (name, bt) in &nb {
        let flat = bt.flatten();
        let lo = flat.lower();
        let hi = flat.upper();
        out.insert(
            name.clone(),
            (0..lo.len())
                .map(|i| Bound::new(lo[[i]], hi[[i]]))
                .collect(),
        );
    }
    out
}

#[cfg(test)]
fn stamp_shapes(diff: &mut GraphNetwork, flat: &HashMap<String, Vec<Bound>>) {
    for (name, v) in flat {
        if diff.declared_shape(name).is_none() {
            diff.set_declared_shape(name.clone(), vec![v.len()]);
        }
    }
}

/// END-TO-END SOUNDNESS: on a stitched diff net with a WIDE real difference
/// (perturbed_g(0.5)), the joint cuts must hold at EVERY reachable point — take a
/// grid of concrete inputs, forward each tower to the exact paired
/// (z, relu(z)) columns, and require every emitted joint cut to hold. A cut that
/// failed here would exclude a real (possibly violating) point → false UNSAT.
#[cfg(feature = "mip")]
#[test]
fn joint_cuts_hold_at_every_reachable_point_on_a_real_diff_net() {
    let f = base_f();
    let g = perturbed_g(0.5);
    let mut diff = build_difference_network(&f, &g).unwrap();
    let input_bounds = input_box();
    let flat = flat_bounds_of(&diff, &input_bounds);
    stamp_shapes(&mut diff, &flat);

    let mut enc = super::super::graph_mip::encode_graph(&diff, &input_bounds, &flat).unwrap();
    let (added, diag) = attach_joint_relu_cuts(&mut enc, &diff, &input_bounds, &flat);
    assert!(added > 0, "expected joint cuts on this diff net");
    assert!(diag.paired_unstable > 0);

    // The joint-cut rows we just appended are the LAST `added` rows.
    let all_rows = enc.problem.rows().to_vec();
    let joint_rows = all_rows[all_rows.len() - added..].to_vec();

    // Sample a grid of concrete inputs; for each, get EXACT per-node values via
    // IBP on the degenerate (point) box (IBP is exact on a point for this net),
    // map node_cols → column values, and check every joint-cut row holds.
    let steps = 12;
    let (lo0, hi0) = (
        input_bounds[0].lower() as f64,
        input_bounds[0].upper() as f64,
    );
    let (lo1, hi1) = (
        input_bounds[1].lower() as f64,
        input_bounds[1].upper() as f64,
    );
    for ia in 0..=steps {
        let x0 = lo0 + (hi0 - lo0) * ia as f64 / steps as f64;
        for ib in 0..=steps {
            let x1 = lo1 + (hi1 - lo1) * ib as f64 / steps as f64;
            let pt = ArrayD::from_shape_vec(IxDyn(&[2]), vec![x0 as f32, x1 as f32]).unwrap();
            let point = BoundedTensor::concrete(pt).unwrap();
            let node_vals = diff.collect_node_bounds(&point).unwrap();
            let mut colval: HashMap<usize, f64> = HashMap::new();
            for (name, cols) in &enc.node_cols {
                if let Some(bt) = node_vals.get(name) {
                    let fl = bt.flatten();
                    let lo = fl.lower();
                    let hi = fl.upper();
                    for (i, &c) in cols.iter().enumerate() {
                        if i < lo.len() {
                            // exact point → lo == hi; use the midpoint defensively.
                            colval.insert(c.0, 0.5 * (lo[[i]] as f64 + hi[[i]] as f64));
                        }
                    }
                }
            }
            for row in &joint_rows {
                let body: f64 = row
                    .coeffs
                    .iter()
                    .map(|&(c, w)| w * colval.get(&c).copied().unwrap_or(0.0))
                    .sum();
                let slack = 1e-5 * (1.0 + body.abs());
                assert!(
                    body <= row.ub + slack && body >= row.lb - slack,
                    "joint cut row VIOLATED at reachable input ({x0},{x1}): body={body} not in \
                     [{}, {}]",
                    row.lb,
                    row.ub,
                );
            }
        }
    }
}

/// FAST MECHANISM MEASUREMENT (instant — tiny net, δ TIGHT). Compare the
/// `diff_output` LP range under (a) baseline triangles, (b) + rel-diff-coupling
/// rows, (c) + JOINT paired-ReLU envelope cuts, on a small isomorphic diff net
/// whose per-neuron δ is genuinely small. This isolates the MECHANISM: when δ is
/// tight the diagonal hull SHOULD shrink the output range below the product of
/// two triangles. (The instance_0 measurement is an explicit corpus harness;
/// there δ is IBP-loose, so this is the clean mechanism gauge.)
#[cfg(feature = "mip")]
#[test]
fn measure_joint_cut_output_lp_range_on_tiny_tight_delta_net() {
    use std::time::{Duration, Instant};
    let f = base_f();
    let g = perturbed_g(0.02); // small perturbation ⇒ tight per-neuron δ
    let mut diff = build_difference_network(&f, &g).unwrap();
    let input_bounds = input_box();
    let flat = flat_bounds_of(&diff, &input_bounds);
    stamp_shapes(&mut diff, &flat);

    let range = |mode: u8| -> (f64, usize) {
        let mut enc = super::super::graph_mip::encode_graph(&diff, &input_bounds, &flat).unwrap();
        let added = match mode {
            0 => 0,
            1 => {
                super::super::graph_mip_diff_coupling::attach_diff_coupling(
                    &mut enc,
                    &diff,
                    &input_bounds,
                    &flat,
                )
                .0
            }
            _ => attach_joint_relu_cuts(&mut enc, &diff, &input_bounds, &flat).0,
        };
        let targets = enc.output_vars.clone();
        let chunk = targets.len().max(1);
        let deadline = Instant::now() + Duration::from_secs(30);
        let r = ny_mip::obbt_relaxation_bounds(
            &enc.problem,
            &targets,
            1,
            Duration::from_secs(10),
            deadline,
            chunk,
        )
        .unwrap();
        let maxw = r.bounds.iter().map(|&(l, u)| u - l).fold(0.0_f64, f64::max);
        (maxw, added)
    };

    let (base_w, _) = range(0);
    let (coup_w, coup_n) = range(1);
    let (joint_w, joint_n) = range(2);
    eprintln!(
        "TINY-NET diff_output max LP range: baseline={base_w:.4} | coupling(+{coup_n})={coup_w:.4} \
         | joint(+{joint_n})={joint_w:.4}"
    );
    // Mechanism assertion: with tight δ the joint cuts must NOT widen the range,
    // and should shrink it strictly below baseline (the diagonal hull bites).
    assert!(
        joint_w <= base_w + 1e-6,
        "joint cuts must never widen the range"
    );
    assert!(joint_n > 0, "joint cuts should be emitted on this net");
    // Soundness of the LP itself: the range must still CONTAIN the true reachable
    // diff (never falsely collapse below it). The true |f−g| here is ~O(0.02··),
    // so the range must remain > 0 (feasible, non-empty).
    assert!(joint_w > 0.0, "range must stay feasible (non-empty)");
}

/// #rel-joint-relu-cuts THE KEY MEASUREMENT. On the REAL instance_0 diff net,
/// compare the
/// LP-relaxation bound on `diff_output` under three encodings:
///   (a) baseline (product of two triangles),
///   (b) + rel-diff-coupling rows (|node diff| ≤ δ),
///   (c) + JOINT paired-ReLU envelope cuts (the diagonal-coupling hull facets),
/// against the ±0.05 band. The lever's success signal is (c)'s diff_output LP
/// range shrinking toward ±0.05; a meaningful shrink (3514 → e.g. 10) is a
/// breakthrough even short of full closure.
pub(crate) fn measure_joint_cut_output_lp_shrink_on_real_instance0(base: &std::path::Path) {
    use std::time::{Duration, Instant};

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
    if diff.declared_shape(NETWORK_INPUT).is_none() {
        diff.set_declared_shape(NETWORK_INPUT, input.shape().to_vec());
    }
    for (name, bt) in &nb {
        if diff.declared_shape(name).is_none() {
            diff.set_declared_shape(name.clone(), bt.shape().to_vec());
        }
    }

    #[derive(Clone, Copy)]
    enum Mode {
        Baseline,
        Coupling,
        Joint,
    }
    let out_lp = |mode: Mode| -> (Vec<(f64, f64)>, usize, f64) {
        let mut enc = super::super::graph_mip::encode_graph(&diff, &input_bounds, &flat).unwrap();
        let added = match mode {
            Mode::Baseline => 0,
            Mode::Coupling => {
                super::super::graph_mip_diff_coupling::attach_diff_coupling(
                    &mut enc,
                    &diff,
                    &input_bounds,
                    &flat,
                )
                .0
            }
            Mode::Joint => attach_joint_relu_cuts(&mut enc, &diff, &input_bounds, &flat).0,
        };
        // The exact-rational ay LP is slow on this diff net; measure a subset of
        // output columns (default 1) for a fast, comparable range readout.
        let ncols = std::env::var("NY_JOINT_MEAS_NCOLS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(1)
            .min(enc.output_vars.len())
            .max(1);
        let targets: Vec<_> = enc.output_vars.iter().copied().take(ncols).collect();
        let chunk = 1usize;
        let t0 = Instant::now();
        // Budget knobs (env-overridable) — the exact-rational OBBT LP on this
        // diff net is slow, so keep each mode bounded. `NY_JOINT_MEAS_PERSOLVE_S`
        // / `NY_JOINT_MEAS_DEADLINE_S`.
        let per_solve = std::env::var("NY_JOINT_MEAS_PERSOLVE_S")
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(15.0);
        let mode_deadline = std::env::var("NY_JOINT_MEAS_DEADLINE_S")
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(40.0);
        let deadline = Instant::now() + Duration::from_secs_f64(mode_deadline);
        let r = ny_mip::obbt_relaxation_bounds(
            &enc.problem,
            &targets,
            1,
            Duration::from_secs_f64(per_solve),
            deadline,
            chunk,
        )
        .unwrap();
        (r.bounds, added, t0.elapsed().as_secs_f64())
    };

    let band = 0.05_f64;
    let report = |label: &str, (b, added, secs): (Vec<(f64, f64)>, usize, f64)| -> bool {
        let mut all = true;
        let mut maxw = 0.0f64;
        for (i, &(l, u)) in b.iter().enumerate() {
            let w = u - l;
            maxw = maxw.max(w);
            eprintln!("  [{label}] diff_output[{i}] = [{l:.4}, {u:.4}] w={w:.4}");
            if !(l >= -band && u <= band) {
                all = false;
            }
        }
        eprintln!("[{label}] rows_added={added} solve={secs:.1}s maxwidth={maxw:.4} closed={all}");
        all
    };

    eprintln!("=== #rel-joint-relu-cuts KEY MEASUREMENT (instance_0, band ±{band}) ===");
    let base_closed = report("baseline", out_lp(Mode::Baseline));
    let coup_closed = report("coupling", out_lp(Mode::Coupling));
    let joint_closed = report("joint", out_lp(Mode::Joint));
    eprintln!("CLOSURE: baseline={base_closed} coupling={coup_closed} joint={joint_closed}");
}
