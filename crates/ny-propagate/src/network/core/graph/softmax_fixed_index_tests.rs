// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #softmax-fixed-index soundness tests.
//!
//! vit_2023 rows are ALL truth-unsat, so no attack can ever catch a false
//! `unsat` this rewrite might mint: soundness has to be carried by these
//! checks. Each one evaluates the REWRITTEN graph (never a hand-copy of the
//! formula), so a construction defect — a double-counted `+1`, a wrong
//! selector, a lost row — fails here.

use super::*;
use crate::layers::SoftmaxLayer;
use ndarray::{ArrayD, IxDyn};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

/// Slack for comparing an IBP enclosure against an f64 reference softmax.
///
/// The rewritten graph's IBP is DIRECTED end to end (MulConstant, ReduceSum,
/// Linear, Sub, Exp, Reciprocal, MulBinary and Clip all round outward), so an
/// enclosure of an f32 point must contain the exact real value. This only
/// absorbs the f64 reference's own rounding (~1e-16 relative).
const IBP_TOL: f64 = 1e-12;

/// Slack for CROWN enclosures: the concretization matvec rounds to nearest in
/// f32 (same constant, same reason as the constant-shift rewrite's tests). A
/// relaxation defect is orders of magnitude larger.
const CROWN_TOL: f64 = 1e-5;

fn box_tensor(shape: &[usize], lower: Vec<f32>, upper: Vec<f32>) -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(shape), lower).unwrap(),
        ArrayD::from_shape_vec(IxDyn(shape), upper).unwrap(),
    )
    .unwrap()
}

fn point_box(x: &ArrayD<f32>) -> BoundedTensor {
    BoundedTensor::new(x.clone(), x.clone()).unwrap()
}

fn softmax_graph(axis: i32) -> GraphNetwork {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "sm",
        Layer::Softmax(SoftmaxLayer::new(axis)),
    ));
    graph.set_output("sm");
    graph
}

/// True softmax along the LAST axis, in f64 (max-shifted, so exact to f64).
fn true_softmax_last_axis(x: &ArrayD<f32>) -> Vec<f64> {
    let n = *x.shape().last().unwrap();
    let flat: Vec<f64> = x.iter().map(|&v| f64::from(v)).collect();
    let mut out = Vec::with_capacity(flat.len());
    for row in flat.chunks_exact(n) {
        let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = row.iter().map(|&v| (v - max).exp()).collect();
        let sum: f64 = exps.iter().sum();
        out.extend(exps.iter().map(|e| e / sum));
    }
    out
}

/// Argmax per row of an f32 tensor (last axis, first maximum).
fn argmax_rows(x: &ArrayD<f32>) -> Vec<usize> {
    let n = *x.shape().last().unwrap();
    let flat: Vec<f64> = x.iter().map(|&v| f64::from(v)).collect();
    fixed_index_rows(&flat, n).unwrap()
}

/// The frozen j* of every row, read back from the planted one-hot selector.
fn jstar_of(graph: &GraphNetwork, softmax: &str) -> Vec<usize> {
    let name = format!("{softmax}/sm_fx_sel_mask");
    let Layer::MulConstant(select) = graph.node(&name).expect("selector node").layer() else {
        panic!("{name} must be the one-hot MulConstant selector");
    };
    let constant = select.constant();
    let n = *constant.shape().last().unwrap();
    let flat: Vec<f32> = constant.iter().copied().collect();
    flat.chunks_exact(n)
        .map(|row| {
            assert!(
                row.iter().all(|&v| v == 0.0 || v == 1.0),
                "selector must be 0/1"
            );
            assert_eq!(
                row.iter().filter(|&&v| v == 1.0).count(),
                1,
                "exactly one selected index per row"
            );
            row.iter().position(|&v| v == 1.0).unwrap()
        })
        .collect()
}

fn random_box(rng: &mut StdRng, n: usize, centre: f32, radius: (f32, f32)) -> (Vec<f32>, Vec<f32>) {
    let centres: Vec<f32> = (0..n).map(|_| rng.random_range(-centre..centre)).collect();
    let radii: Vec<f32> = (0..n)
        .map(|_| rng.random_range(radius.0..radius.1))
        .collect();
    (
        centres.iter().zip(&radii).map(|(c, r)| c - r).collect(),
        centres.iter().zip(&radii).map(|(c, r)| c + r).collect(),
    )
}

fn sample_in(rng: &mut StdRng, lower: &[f32], upper: &[f32]) -> Vec<f32> {
    lower
        .iter()
        .zip(upper)
        .map(|(&l, &u)| rng.random_range(l..=u))
        .collect()
}

fn assert_encloses(bounds: &BoundedTensor, truth: &[f64], tol: f64, label: &str) {
    for (i, ((l, u), t)) in bounds
        .lower()
        .iter()
        .zip(bounds.upper().iter())
        .zip(truth)
        .enumerate()
    {
        assert!(
            f64::from(*l) <= t + tol && t - tol <= f64::from(*u),
            "{label}: true softmax {t} at coordinate {i} outside [{l}, {u}]"
        );
    }
}

/// Structure: the exact chain is planted, the softmax node keeps its name as
/// the closing `Clip(0, 1)`, and `j*` is the per-ROW argmax of the box centre
/// (different rows get different indices).
#[test]
fn softmax_fixed_index_rewrite_structure_and_per_row_centre_argmax() {
    let mut graph = softmax_graph(-1);
    // Centres: row0 [0, 1, 3, -1] (argmax 2), row1 [5, 2, -3, 4] (argmax 0),
    // row2 [-2, 0, 1, 2] (argmax 3); radius 0.5 everywhere.
    let centres = [
        0.0f32, 1.0, 3.0, -1.0, 5.0, 2.0, -3.0, 4.0, -2.0, 0.0, 1.0, 2.0,
    ];
    let input = box_tensor(
        &[3, 4],
        centres.iter().map(|c| c - 0.5).collect(),
        centres.iter().map(|c| c + 0.5).collect(),
    );
    let report = graph.decompose_softmax_complex_fixed_index(&input).unwrap();
    assert_eq!(report.decomposed, vec!["sm".to_string()]);
    assert_eq!((report.skipped).len(), 0);
    assert_eq!((report.wide_ibp).len(), 0);
    assert_eq!(graph.num_nodes(), 1 + FIXED_INDEX_SUFFIXES.len());

    let clip = graph.node("sm").unwrap();
    let Layer::Clip(unit) = clip.layer() else {
        panic!("the softmax node must close as Clip(0, 1)");
    };
    assert_eq!((unit.min, unit.max), (0.0, 1.0));
    assert_eq!(clip.inputs(), ["sm/sm_fx_mul"]);
    assert_eq!(graph.output_name(), "sm");

    let expect = |name: &str, inputs: &[&str]| {
        let node = graph.node(name).unwrap_or_else(|| panic!("missing {name}"));
        assert_eq!(node.inputs(), inputs, "{name} inputs");
        node.layer().clone()
    };
    assert!(matches!(
        expect("sm/sm_fx_sel_mask", &[NETWORK_INPUT]),
        Layer::MulConstant(_)
    ));
    assert!(matches!(
        expect("sm/sm_fx_sel", &["sm/sm_fx_sel_mask"]),
        Layer::ReduceSum(_)
    ));
    let Layer::Linear(bcast) = expect("sm/sm_fx_bcast", &["sm/sm_fx_sel"]) else {
        panic!("broadcast must be a Linear");
    };
    assert_eq!((bcast.out_features(), bcast.in_features()), (4, 1));
    assert!(matches!(
        expect("sm/sm_fx_diff", &[NETWORK_INPUT, "sm/sm_fx_bcast"]),
        Layer::Sub(_)
    ));
    assert!(matches!(
        expect("sm/sm_fx_shift", &["sm/sm_fx_diff"]),
        Layer::MulConstant(_)
    ));
    assert!(matches!(
        expect("sm/sm_fx_exp", &["sm/sm_fx_shift"]),
        Layer::Exp(_)
    ));
    assert!(matches!(
        expect("sm/sm_fx_sum", &["sm/sm_fx_exp"]),
        Layer::ReduceSum(_)
    ));
    assert!(matches!(
        expect("sm/sm_fx_recip", &["sm/sm_fx_sum"]),
        Layer::Reciprocal(_)
    ));
    assert!(matches!(
        expect("sm/sm_fx_mul", &["sm/sm_fx_exp", "sm/sm_fx_recip"]),
        Layer::MulBinary(_)
    ));

    assert_eq!(jstar_of(&graph, "sm"), vec![2, 0, 3]);
    // The shift mask is exactly the complement of the selector.
    let Layer::MulConstant(mask) = graph.node("sm/sm_fx_shift").unwrap().layer() else {
        unreachable!()
    };
    let Layer::MulConstant(select) = graph.node("sm/sm_fx_sel_mask").unwrap().layer() else {
        unreachable!()
    };
    for (m, s) in mask.constant().iter().zip(select.constant().iter()) {
        assert_eq!(m + s, 1.0);
    }

    assert_eq!(
        graph.softmax_fixed_index_census(),
        (vec!["sm/sm_fx_shift".to_string()], 0)
    );

    // Idempotence: no Softmax remains, so a second call is a no-op.
    let again = graph.decompose_softmax_complex_fixed_index(&input).unwrap();
    assert!(again.decomposed.is_empty() && again.skipped.is_empty());
}

/// Requirement: the REWRITTEN graph's forward equals softmax EXACTLY at
/// random points — including points whose argmax is not the frozen `j*` —
/// and the unsound "keep every term AND add 1" denominator would be caught.
#[test]
fn softmax_fixed_index_decomposition_exact_pointwise() {
    for (shape, seed) in [
        (vec![3usize, 5], 0x005e_edf1_u64),
        (vec![2, 3, 4], 0x005e_edf2),
    ] {
        let total: usize = shape.iter().product();
        let n = *shape.last().unwrap();
        let mut rng = StdRng::seed_from_u64(seed);
        let (mut lower, mut upper) = random_box(&mut rng, total, 2.0, (0.2, 1.0));
        // Force a near-tie in row 0 so many samples have argmax != j*.
        lower[0] = -0.9;
        upper[0] = 1.1;
        lower[1] = -1.0;
        upper[1] = 1.0;
        let input = box_tensor(&shape, lower.clone(), upper.clone());

        let mut graph = softmax_graph(-1);
        let report = graph.decompose_softmax_complex_fixed_index(&input).unwrap();
        assert_eq!(report.decomposed.len(), 1, "{shape:?}: must decompose");
        let jstar = jstar_of(&graph, "sm");

        let mut argmax_differs = 0usize;
        for trial in 0..4000 {
            let x =
                ArrayD::from_shape_vec(IxDyn(&shape), sample_in(&mut rng, &lower, &upper)).unwrap();
            argmax_differs += argmax_rows(&x)
                .iter()
                .zip(&jstar)
                .filter(|(a, j)| a != j)
                .count();
            let truth = true_softmax_last_axis(&x);
            let nodes = graph
                .collect_node_bounds_with_engine_and_deadline(&point_box(&x), None, None)
                .unwrap();
            let out = &nodes["sm"];
            let label = format!("{shape:?} trial {trial}");
            assert_encloses(out, &truth, IBP_TOL, &label);
            for (l, u) in out.lower().iter().zip(out.upper().iter()) {
                assert!(
                    f64::from(*u) - f64::from(*l) < 1e-5,
                    "{label}: point enclosure unexpectedly wide [{l}, {u}]"
                );
            }

            // Negative control: the UNSOUND double count, computed from the
            // rewritten graph's own Exp and ReduceSum values, lies OUTSIDE the
            // point enclosure on some coordinate of every sample — so the
            // enclosure check above cannot pass a double-counted `+1`.
            let exp: Vec<f64> = nodes["sm/sm_fx_exp"]
                .lower()
                .iter()
                .zip(nodes["sm/sm_fx_exp"].upper().iter())
                .map(|(&l, &u)| f64::midpoint(f64::from(l), f64::from(u)))
                .collect();
            let sums: Vec<f64> = nodes["sm/sm_fx_sum"]
                .lower()
                .iter()
                .zip(nodes["sm/sm_fx_sum"].upper().iter())
                .map(|(&l, &u)| f64::midpoint(f64::from(l), f64::from(u)))
                .collect();
            let out_lower: Vec<f64> = out.lower().iter().map(|&v| f64::from(v)).collect();
            let out_upper: Vec<f64> = out.upper().iter().map(|&v| f64::from(v)).collect();
            let caught = exp.iter().enumerate().any(|(i, e)| {
                let unsound = e / (sums[i / n] + 1.0);
                out_upper[i] + IBP_TOL < unsound || unsound < out_lower[i] - IBP_TOL
            });
            assert!(caught, "{label}: a double-counted +1 would pass unnoticed");
        }
        assert!(
            argmax_differs > 0,
            "{shape:?}: the fixture must exercise rows whose argmax is not j*"
        );
    }
}

/// Requirement: IBP and CROWN bounds of the rewritten graph contain the true
/// softmax at the centre, random points and box vertices of random boxes; IBP
/// stays inside `[0, 1]`; near-degenerate boxes (f32 rounding dominates) are
/// contained with the strict IBP tolerance.
#[test]
fn softmax_fixed_index_bounds_contain_samples() {
    let shape = [2usize, 4usize];
    let total: usize = shape.iter().product();
    let mut rng = StdRng::seed_from_u64(0xc0ff_ee12);

    for round in 0..30 {
        let (lower, upper) = if round < 24 {
            random_box(&mut rng, total, 4.0, (0.05, 2.5))
        } else {
            // f32-rounding regime: radius a few ulps of the centre.
            let centres: Vec<f32> = (0..total).map(|_| rng.random_range(-4.0..4.0)).collect();
            (
                centres
                    .iter()
                    .map(|c| c - (c.abs() * 4.0 * f32::EPSILON).max(f32::MIN_POSITIVE))
                    .collect(),
                centres
                    .iter()
                    .map(|c| c + (c.abs() * 4.0 * f32::EPSILON).max(f32::MIN_POSITIVE))
                    .collect(),
            )
        };
        let input = box_tensor(&shape, lower.clone(), upper.clone());
        let mut graph = softmax_graph(-1);
        let report = graph.decompose_softmax_complex_fixed_index(&input).unwrap();
        assert_eq!(report.decomposed.len(), 1, "round {round}");

        let ibp = graph.propagate_ibp(&input).unwrap();
        for (l, u) in ibp.lower().iter().zip(ibp.upper().iter()) {
            assert!(
                *l >= 0.0 && *u <= 1.0 && l <= u,
                "round {round}: IBP [{l}, {u}] must lie in [0, 1]"
            );
        }
        let crown = graph
            .propagate_crown_with_engine_and_deadline(&input, None, None)
            .unwrap()
            .bounds;
        assert!(
            crown
                .lower()
                .iter()
                .chain(crown.upper().iter())
                .all(|v| v.is_finite()),
            "round {round}: CROWN bounds must be finite"
        );

        let mut samples: Vec<Vec<f32>> = vec![lower
            .iter()
            .zip(&upper)
            .map(|(&l, &u)| f64::midpoint(f64::from(l), f64::from(u)) as f32)
            .collect()];
        for _ in 0..64 {
            samples.push(sample_in(&mut rng, &lower, &upper));
        }
        for mask in 0..(1usize << total) {
            samples.push(
                (0..total)
                    .map(|i| {
                        if (mask >> i) & 1 == 1 {
                            upper[i]
                        } else {
                            lower[i]
                        }
                    })
                    .collect(),
            );
        }
        for sample in &samples {
            let x = ArrayD::from_shape_vec(IxDyn(&shape), sample.clone()).unwrap();
            let truth = true_softmax_last_axis(&x);
            assert_encloses(&ibp, &truth, IBP_TOL, &format!("IBP round {round}"));
            assert_encloses(&crown, &truth, CROWN_TOL, &format!("CROWN round {round}"));
        }
    }
}

/// Requirement: the denominator floor `Σ ≥ 1` exists ONLY because the `j*`
/// term is present (as an exactly-zero Exp input), and the construction adds
/// no constant to the sum.
#[test]
fn softmax_fixed_index_denominator_floor_requires_jstar_term() {
    // Every non-j* logit sits far below x_{j*}: their Exp terms alone cannot
    // reach 1, so any floor at 1 must come from the j* term.
    let centres = [4.0f32, -6.0, -7.0, -8.0, -9.0, -8.0, 5.0, -7.0];
    for radius in [0.5f32, 3.0] {
        let input = box_tensor(
            &[2, 4],
            centres.iter().map(|c| c - radius).collect(),
            centres.iter().map(|c| c + radius).collect(),
        );
        let mut graph = softmax_graph(-1);
        graph.decompose_softmax_complex_fixed_index(&input).unwrap();
        let jstar = jstar_of(&graph, "sm");
        assert_eq!(jstar, vec![0, 2]);

        // Structure: the Reciprocal reads the bare ReduceSum of ALL Exp terms;
        // there is no constant anywhere in the denominator chain.
        assert_eq!(
            graph.node("sm/sm_fx_recip").unwrap().inputs(),
            ["sm/sm_fx_sum"]
        );
        assert_eq!(
            graph.node("sm/sm_fx_sum").unwrap().inputs(),
            ["sm/sm_fx_exp"]
        );
        assert!(
            graph
                .node_names()
                .iter()
                .all(|name| !matches!(graph.node(name).unwrap().layer(), Layer::AddConstant(_))),
            "the fixed-index rewrite must not add a constant to the denominator"
        );

        let nodes = graph
            .collect_node_bounds_with_engine_and_deadline(&input, None, None)
            .unwrap();
        let flat = |name: &str| -> (Vec<f32>, Vec<f32>) {
            let b = &nodes[name];
            (
                b.lower().iter().copied().collect(),
                b.upper().iter().copied().collect(),
            )
        };
        let (shift_l, shift_u) = flat("sm/sm_fx_shift");
        let (diff_l, diff_u) = flat("sm/sm_fx_diff");
        let (exp_l, exp_u) = flat("sm/sm_fx_exp");
        let (sum_l, _) = flat("sm/sm_fx_sum");
        for (row, &j) in jstar.iter().enumerate() {
            let at = row * 4 + j;
            // The j* coordinate is EXACTLY zero under plain IBP (the mask) ...
            assert_eq!((shift_l[at], shift_u[at]), (0.0, 0.0), "row {row}");
            // ... so its Exp term is 1 to within outward rounding ...
            assert!(f64::from(exp_l[at]) >= 1.0 - 1e-6 && f64::from(exp_u[at]) <= 1.0 + 1e-6);
            // ... and the floor holds at the IBP level.
            assert!(
                f64::from(sum_l[row]) >= 1.0 - 1e-6,
                "row {row}: denominator floor lost (sum lower {})",
                sum_l[row]
            );

            // Negative control 1: without the j* term the rest cannot reach 1.
            let rest_upper: f64 = (0..4)
                .filter(|&k| k != j)
                .map(|k| f64::from(exp_u[row * 4 + k]))
                .sum();
            assert!(
                rest_upper < 1.0,
                "row {row}: the non-j* terms alone reach {rest_upper}; the fixture no \
                 longer isolates the j* term"
            );
            // Negative control 2: WITHOUT the exact-zero mask, plain IBP of
            // x_{j*} - x_{j*} is a symmetric interval, not zero, so exp of its
            // lower end is < 1 and the floor would be lost.
            assert!(
                diff_l[at] < 0.0 && diff_u[at] > 0.0,
                "row {row}: unmasked x_j - x_j must be a nonzero IBP interval, got [{}, {}]",
                diff_l[at],
                diff_u[at]
            );
            assert!(f64::from(diff_l[at]).exp() < 1.0 - 1e-3);
        }
    }
}

/// The DAG alpha loop runs over the rewritten graph, returns a sound bound no
/// looser than IBP, keeps the Reciprocal alpha, and has NO alpha on the Exp
/// node (abc's `disable_optimization: ['Exp']`: the tangent stays frozen).
///
/// Run on BOTH forms of the typed `fix_interm_bounds: false` route: uncapped
/// (historical post-loop re-collection) and under #vit-crown-interm's
/// `crown_ibp_reference_cap_secs` (capped reference collection, init-map reuse
/// for the loop's initial bound, artifact publication, capped post-loop pass).
#[test]
fn softmax_fixed_index_alpha_loop_is_sound_and_keeps_exp_frozen() {
    let shape = [2usize, 3usize];
    let input = box_tensor(
        &shape,
        vec![-1.0, 0.5, -0.5, 1.0, -2.0, 0.0],
        vec![1.0, 2.0, 0.5, 3.0, 0.0, 1.5],
    );
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "sm",
        Layer::Softmax(SoftmaxLayer::new(-1)),
    ));
    graph.add_node(GraphNode::new(
        "out",
        Layer::ReduceSum(ReduceSumLayer::new(vec![0, 1], false)),
        vec!["sm".to_string()],
    ));
    graph.set_output("out");
    let report = graph.decompose_softmax_complex_fixed_index(&input).unwrap();
    assert_eq!(report.decomposed.len(), 1);

    let ibp = graph.propagate_ibp(&input).unwrap();
    let (ibp_l, ibp_u) = (ibp.flatten().lower()[0], ibp.flatten().upper()[0]);

    for cap in [None, Some(30u64)] {
        let config = crate::bounds::AlphaCrownConfig {
            iterations: 8,
            gradient_method: crate::bounds::GradientMethod::AnalyticChain,
            fix_interm_bounds: false,
            crown_ibp_reference_cap_secs: cap,
            adaptive_skip: false,
            adaptive_skip_pilot: false,
            ..crate::bounds::AlphaCrownConfig::default()
        };
        assert_eq!(
            config.crown_ibp_reference_cap().map(|d| d.as_secs()),
            cap,
            "the cap must resolve on the typed route"
        );
        let (bounds, alpha_state) = graph
            .collect_alpha_crown_bounds_dag(&input, &config)
            .expect("alpha-CROWN DAG over the fixed-index softmax should succeed");
        let out = bounds.get(graph.output_name()).expect("output bounds");
        let (al, au) = (out.flatten().lower()[0], out.flatten().upper()[0]);
        assert!(
            al.is_finite() && au.is_finite(),
            "cap {cap:?}: bounds [{al}, {au}]"
        );
        // Two softmax rows each summing to exactly 1.
        assert!(
            al <= 2.0 + 1e-4 && 2.0 - 1e-4 <= au,
            "cap {cap:?}: true output 2.0 outside alpha bounds [{al}, {au}]"
        );
        assert!(
            al >= ibp_l - 1e-4,
            "cap {cap:?}: alpha lower {al} looser than IBP {ibp_l}"
        );
        assert!(
            au <= ibp_u + 1e-4,
            "cap {cap:?}: alpha upper {au} looser than IBP {ibp_u}"
        );
        // Every published intermediate is an enclosure too: each softmax
        // output coordinate lies in [0, 1] up to rounding and is finite.
        let sm = bounds.get("sm").expect("softmax output bounds");
        for (l, u) in sm.lower().iter().zip(sm.upper().iter()) {
            assert!(
                l.is_finite() && u.is_finite() && *l <= *u,
                "cap {cap:?}: softmax output [{l}, {u}] must be a finite interval"
            );
            assert!(
                *l <= 1.0 + 1e-6 && *u >= -1e-6,
                "cap {cap:?}: softmax output [{l}, {u}] misses [0, 1]"
            );
        }

        let exp = "sm/sm_fx_exp";
        assert!(!alpha_state.alphas.contains_key(exp));
        assert!(!alpha_state.monotone_s_shaped_alphas.contains_key(exp));
        assert!(!alpha_state.sqrt_alphas.contains_key(exp));
        assert!(!alpha_state.reciprocal_alphas.contains_key(exp));
        assert!(
            alpha_state.reciprocal_alphas.contains_key("sm/sm_fx_recip"),
            "cap {cap:?}: the Reciprocal tangent alpha must stay optimizable (its input is >= 1)"
        );
    }
}

/// A box far too wide for the constant-shift guard is still decomposed (and
/// counted in `wide_ibp`); every IBP enclosure stays inside `[0, 1]` and
/// contains the truth.
#[test]
fn softmax_fixed_index_wide_box_decomposes_and_stays_in_the_unit_interval() {
    let mut graph = softmax_graph(-1);
    let lower = vec![-200.0f32; 3];
    let upper = vec![200.0f32; 3];
    let input = box_tensor(&[1, 3], lower.clone(), upper.clone());
    let report = graph.decompose_softmax_complex_fixed_index(&input).unwrap();
    assert_eq!(report.decomposed, vec!["sm".to_string()]);
    assert_eq!(report.wide_ibp, vec!["sm".to_string()]);
    assert_eq!((report.skipped).len(), 0);

    let ibp = graph.propagate_ibp(&input).unwrap();
    for (l, u) in ibp.lower().iter().zip(ibp.upper().iter()) {
        assert!(
            *l >= 0.0 && *u <= 1.0 && l <= u,
            "IBP [{l}, {u}] must lie in [0, 1]"
        );
    }
    let mut rng = StdRng::seed_from_u64(0x00ab_cdef);
    for _ in 0..256 {
        let x =
            ArrayD::from_shape_vec(IxDyn(&[1, 3]), sample_in(&mut rng, &lower, &upper)).unwrap();
        assert_encloses(&ibp, &true_softmax_last_axis(&x), IBP_TOL, "wide box");
    }
}

/// A softmax over a non-last axis keeps the direct-LSE path (with a reason);
/// the census then reports it as a remaining softmax.
#[test]
fn softmax_fixed_index_skips_a_non_last_axis_and_keeps_the_lse_path() {
    let mut graph = softmax_graph(0);
    let input = box_tensor(&[4, 3], vec![-1.0; 12], vec![1.0; 12]);
    let report = graph.decompose_softmax_complex_fixed_index(&input).unwrap();
    assert_eq!((report.decomposed).len(), 0);
    assert_eq!(report.skipped.len(), 1);
    assert!(
        report.skipped[0].1.contains("last axis"),
        "skip reason should name the axis: {}",
        report.skipped[0].1
    );
    assert_eq!(graph.num_nodes(), 1);
    assert!(matches!(
        graph.node("sm").unwrap().layer(),
        Layer::Softmax(_)
    ));
    assert_eq!(graph.softmax_fixed_index_census(), (Vec::new(), 1));
}

/// The frozen index is the FIRST maximum of each row; malformed input is
/// refused rather than guessed.
#[test]
fn fixed_index_rows_takes_the_first_maximum_and_refuses_malformed_rows() {
    assert_eq!(
        fixed_index_rows(&[0.0, 2.0, 2.0, 1.0, 5.0, -1.0, 5.0, 0.0], 4),
        Some(vec![1, 0])
    );
    assert_eq!(fixed_index_rows(&[3.0], 1), Some(vec![0]));
    assert_eq!(fixed_index_rows(&[0.0, f64::NAN], 2), None);
    assert_eq!(fixed_index_rows(&[0.0, 1.0, 2.0], 2), None);
    assert_eq!(fixed_index_rows(&[0.0, 1.0], 0), None);
}
