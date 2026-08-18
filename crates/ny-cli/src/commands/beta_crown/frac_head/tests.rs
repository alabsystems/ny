// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for the normalized-power fractional-head verifier: directed
//! rounding, threshold-vertex head interval (enclosure + tightness),
//! detection on a synthetic head graph, and end-to-end unsat/sat driving.

use std::time::{Duration, Instant};

use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_onnx::vnnlib::{OutputConstraint, VnnLibSpec};
use ny_propagate::layers::{
    DivLayer, LinearLayer, PowConstantLayer, ReLULayer, ReduceSumLayer, SubLayer,
};
use ny_propagate::{BabVerificationStatus, GraphNetwork, GraphNode, Layer};

use super::super::BetaCrownModel;
use super::*;

// -----------------------------------------------------------------------
// Directed arithmetic
// -----------------------------------------------------------------------

#[test]
fn pow_bounds_are_directed() {
    for &x in &[0.0_f64, 1e-9, 0.3, 1.0, 1.5, 7.25, 123.456] {
        for k in 1..=8u32 {
            let exact = x.powi(k as i32);
            assert!(pow_down(x, k) <= exact, "pow_down({x},{k})");
            assert!(pow_up(x, k) >= exact, "pow_up({x},{k})");
            // Within a few ULPs of exact.
            assert!(pow_up(x, k) - pow_down(x, k) <= exact * 1e-12 + f64::MIN_POSITIVE);
        }
    }
}

#[test]
fn f32_conversion_is_outward() {
    for &x in &[0.1_f64, -0.1, 1.0 / 3.0, 5.997603738969738, -7.33e-5] {
        assert!(f64::from(f32_down(x)) <= x);
        assert!(f64::from(f32_up(x)) >= x);
    }
}

// -----------------------------------------------------------------------
// Threshold-vertex head interval
// -----------------------------------------------------------------------

/// Brute-force the exact range over all 2^n vertices (extrema of a
/// linear-fractional function over a box lie at vertices).
fn brute_force_range(pl: &[f64], pu: &[f64], coeffs: &[f64], bias: f64) -> (f64, f64) {
    let n = pl.len();
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for mask in 0..(1usize << n) {
        let p: Vec<f64> = (0..n)
            .map(|i| if mask & (1 << i) != 0 { pu[i] } else { pl[i] })
            .collect();
        let den: f64 = p.iter().sum();
        if den <= 0.0 {
            continue;
        }
        let num: f64 = p.iter().zip(coeffs).map(|(&pi, &ci)| ci * pi).sum();
        let v = num / den + bias;
        lo = lo.min(v);
        hi = hi.max(v);
    }
    (lo, hi)
}

#[test]
fn head_range_encloses_and_is_tight() {
    // Deterministic pseudo-random boxes (LCG), pensieve-like and adversarial
    // (signed coefficients, ties, zeros in pl).
    let mut state = 0x243F_6A88_85A3_08D3_u64;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) as f64) / f64::from(u32::MAX)
    };
    for case in 0..200 {
        let n = 2 + (case % 5);
        let coeffs: Vec<f64> = (0..n).map(|_| (next() - 0.3) * 300.0).collect();
        let bias = (next() - 0.5) * 10.0;
        let mut pl = Vec::with_capacity(n);
        let mut pu = Vec::with_capacity(n);
        for i in 0..n {
            let a = if i == 0 {
                0.05 + next()
            } else {
                next() * (case % 3) as f64
            };
            let w = next() * 2.0;
            pl.push(a);
            pu.push(a + w);
        }
        let Some((lo, hi)) = head_range(&pl, &pu, &coeffs, bias) else {
            panic!("head_range returned None for a positive-denominator box");
        };
        let (bf_lo, bf_hi) = brute_force_range(&pl, &pu, &coeffs, bias);
        // Enclosure.
        assert!(
            lo <= bf_lo + 1e-12,
            "case {case}: lower {lo} above brute-force {bf_lo}"
        );
        assert!(
            hi >= bf_hi - 1e-12,
            "case {case}: upper {hi} below brute-force {bf_hi}"
        );
        // Tightness: the threshold-vertex bound IS the vertex extreme.
        assert!(
            (lo - bf_lo).abs() <= 1e-9 * (1.0 + bf_lo.abs()),
            "case {case} loose lower"
        );
        assert!(
            (hi - bf_hi).abs() <= 1e-9 * (1.0 + bf_hi.abs()),
            "case {case} loose upper"
        );
        // Random interior points stay inside.
        for _ in 0..50 {
            let p: Vec<f64> = (0..n).map(|i| pl[i] + next() * (pu[i] - pl[i])).collect();
            let den: f64 = p.iter().sum();
            let num: f64 = p.iter().zip(&coeffs).map(|(&pi, &ci)| ci * pi).sum();
            let v = num / den + bias;
            assert!(
                v >= lo - 1e-9 && v <= hi + 1e-9,
                "case {case}: interior point escaped"
            );
        }
    }
}

#[test]
fn head_range_rejects_vanishing_denominator() {
    // All lower bounds zero: the box contains p = 0 (0/0) — unclaimable.
    assert!(head_range(&[0.0, 0.0], &[1.0, 2.0], &[10.0, 20.0], 0.0).is_none());
    // Malformed boxes.
    assert!(head_range(&[1.0, -0.1], &[2.0, 1.0], &[10.0, 20.0], 0.0).is_none());
    assert!(head_range(&[1.0, 2.0], &[2.0, 1.0], &[10.0, 20.0], 0.0).is_none());
    assert!(head_range(&[1.0], &[2.0], &[10.0, 20.0], 0.0).is_none());
}

// -----------------------------------------------------------------------
// Synthetic end-to-end
// -----------------------------------------------------------------------

/// Build one head: Linear(W,b) -> ReLU -> Pow(3) -> ReduceSum -> Div ->
/// Linear(coeffs). Returns the head's output node name.
fn add_head(
    graph: &mut GraphNetwork,
    tag: &str,
    w: Array2<f32>,
    b: Array1<f32>,
    coeffs: Vec<f32>,
) -> String {
    let n = coeffs.len();
    graph.add_node(GraphNode::from_input(
        format!("{tag}_lin"),
        Layer::Linear(LinearLayer::new(w, Some(b)).unwrap()),
    ));
    graph.add_node(GraphNode::new(
        format!("{tag}_relu"),
        Layer::ReLU(ReLULayer::new()),
        vec![format!("{tag}_lin")],
    ));
    graph.add_node(GraphNode::new(
        format!("{tag}_pow"),
        Layer::PowConstant(PowConstantLayer::new(3.0)),
        vec![format!("{tag}_relu")],
    ));
    graph.set_declared_shape(format!("{tag}_pow"), vec![1, n]);
    graph.add_node(GraphNode::new(
        format!("{tag}_rsum"),
        Layer::ReduceSum(ReduceSumLayer::new(vec![-1], true)),
        vec![format!("{tag}_pow")],
    ));
    graph.add_node(GraphNode::new(
        format!("{tag}_div"),
        Layer::Div(DivLayer),
        vec![format!("{tag}_pow"), format!("{tag}_rsum")],
    ));
    let coeff_mat = Array2::from_shape_vec((1, n), coeffs).unwrap();
    graph.add_node(GraphNode::new(
        format!("{tag}_score"),
        Layer::Linear(LinearLayer::new(coeff_mat, None).unwrap()),
        vec![format!("{tag}_div")],
    ));
    format!("{tag}_score")
}

/// Y = s_A - s_B where head A's logits sit near [3, 1] and head B's near
/// [1, 3] over x in [0, 0.1]^2, coeffs [10, 20]:
///   s_A ≈ (10·27 + 20·1)/28 ≈ 10.7,  s_B ≈ (10·1 + 20·27)/28 ≈ 19.6,
///   Y ≈ -8.9 (and Y < 0 over the whole box).
fn synthetic_model() -> GraphNetwork {
    let mut graph = GraphNetwork::new();
    let w = Array2::from_shape_vec((2, 2), vec![1.0_f32, 0.0, 0.0, 1.0]).unwrap();
    let a = add_head(
        &mut graph,
        "a",
        w.clone(),
        Array1::from_vec(vec![3.0_f32, 1.0]),
        vec![10.0, 20.0],
    );
    let b = add_head(
        &mut graph,
        "b",
        w,
        Array1::from_vec(vec![1.0_f32, 3.0]),
        vec![10.0, 20.0],
    );
    graph.add_node(GraphNode::new("y", Layer::Sub(SubLayer), vec![a, b]));
    graph.set_output("y");
    graph
}

fn spec_with(constraint: OutputConstraint) -> VnnLibSpec {
    let mut spec = VnnLibSpec::new();
    spec.num_inputs = 2;
    spec.num_outputs = 1;
    spec.input_bounds = vec![(0.0, 0.1), (0.0, 0.1)];
    spec.output_constraints = vec![constraint.clone()];
    spec.output_constraint_clauses = vec![vec![constraint]];
    spec.is_disjunction = true;
    spec
}

#[test]
fn detects_synthetic_head_pair() {
    let graph = synthetic_model();
    let spec = spec_with(OutputConstraint::GreaterEqConst(0, 0.0));
    let plan = detect(&graph, &[1, 2], &spec).expect("should detect");
    assert_eq!(plan.heads[0].coeffs, vec![10.0, 20.0]);
    assert_eq!(plan.heads[0].exponent, 3);
    assert_eq!(plan.heads[1].coeffs, vec![10.0, 20.0]);
    // Both dims influence both heads here (shared input).
    assert_eq!(plan.heads[0].dims.len(), 2);
}

#[test]
fn verifies_unsat_when_upper_cover_suffices() {
    // Violation iff Y >= 0; true Y ≈ [-9.6, -8.2] — must verify.
    let graph = synthetic_model();
    let model = BetaCrownModel::Graph(Box::new(graph));
    let spec = spec_with(OutputConstraint::GreaterEqConst(0, 0.0));
    let result = try_frac_head_verification(
        &model,
        &[1, 2],
        &spec,
        Some(Instant::now() + Duration::from_secs(20)),
    )
    .expect("should decide");
    assert!(matches!(result.result, BabVerificationStatus::Verified));
}

#[test]
fn unbounded_deadline_preserves_structural_verification() {
    let graph = synthetic_model();
    let model = BetaCrownModel::Graph(Box::new(graph));
    let spec = spec_with(OutputConstraint::GreaterEqConst(0, 0.0));
    let result = try_frac_head_verification(&model, &[1, 2], &spec, None)
        .expect("bounded structural search should complete without a deadline");
    assert!(matches!(result.result, BabVerificationStatus::Verified));
}

#[test]
fn reports_sat_from_violating_center() {
    // Violation iff Y <= 0; Y ≈ -8.9 at the center — a real counterexample.
    let graph = synthetic_model();
    let model = BetaCrownModel::Graph(Box::new(graph));
    let spec = spec_with(OutputConstraint::LessEqConst(0, 0.0));
    let result = try_frac_head_verification(
        &model,
        &[1, 2],
        &spec,
        Some(Instant::now() + Duration::from_secs(20)),
    )
    .expect("should decide");
    match result.result {
        BabVerificationStatus::Violated {
            counterexample,
            output,
        } => {
            assert_eq!(counterexample.len(), 2);
            assert_eq!(output.len(), 1);
            assert!(output[0] <= 0.0);
        }
        other => panic!("expected Violated, got {other:?}"),
    }
}

#[test]
fn falls_through_on_non_matching_graph() {
    // Plain Linear output: no Sub-of-heads.
    let mut graph = GraphNetwork::new();
    let w = Array2::from_shape_vec((1, 2), vec![1.0_f32, 1.0]).unwrap();
    graph.add_node(GraphNode::from_input(
        "lin",
        Layer::Linear(LinearLayer::new(w, None).unwrap()),
    ));
    graph.set_output("lin");
    let model = BetaCrownModel::Graph(Box::new(graph));
    let spec = spec_with(OutputConstraint::GreaterEqConst(0, 0.0));
    assert!(try_frac_head_verification(
        &model,
        &[1, 2],
        &spec,
        Some(Instant::now() + Duration::from_secs(5)),
    )
    .is_none());
}

#[test]
fn falls_through_without_declared_pow_shape() {
    let mut graph = GraphNetwork::new();
    let w = Array2::from_shape_vec((2, 2), vec![1.0_f32, 0.0, 0.0, 1.0]).unwrap();
    let a = add_head(
        &mut graph,
        "a",
        w.clone(),
        Array1::from_vec(vec![3.0_f32, 1.0]),
        vec![10.0, 20.0],
    );
    let b = add_head(
        &mut graph,
        "b",
        w,
        Array1::from_vec(vec![1.0_f32, 3.0]),
        vec![10.0, 20.0],
    );
    graph.add_node(GraphNode::new("y", Layer::Sub(SubLayer), vec![a, b]));
    graph.set_output("y");
    // Erase the declared shapes by rebuilding without them is intrusive;
    // instead check detect() on a fresh graph that never set them.
    let mut bare = GraphNetwork::new();
    for name in graph.node_names() {
        bare.add_node(graph.node(name).unwrap().clone());
    }
    bare.set_output("y");
    let spec = spec_with(OutputConstraint::GreaterEqConst(0, 0.0));
    assert!(detect(&bare, &[1, 2], &spec).is_none());
}

#[test]
fn multi_output_spec_is_rejected() {
    let graph = synthetic_model();
    let mut spec = spec_with(OutputConstraint::GreaterEqConst(0, 0.0));
    spec.num_outputs = 2;
    assert!(detect(&graph, &[1, 2], &spec).is_none());
}

// -----------------------------------------------------------------------
// Sanity: the synthetic model's true output really is what the tests
// assume (guards the fixtures themselves).
// -----------------------------------------------------------------------

#[test]
fn synthetic_model_forward_matches_formula() {
    let graph = synthetic_model();
    let point = vec![0.05_f32, 0.05];
    let arr = ArrayD::from_shape_vec(IxDyn(&[1, 2]), point).unwrap();
    let input = ny_tensor::BoundedTensor::concrete(arr).unwrap();
    let out = graph.propagate_ibp(&input).unwrap();
    let y = f64::from(out.lower().iter().copied().next().unwrap());
    // logits A = [3.05, 1.05], B = [1.05, 3.05], k = 3, c = [10, 20].
    let pa = [3.05_f64.powi(3), 1.05_f64.powi(3)];
    let pb = [1.05_f64.powi(3), 3.05_f64.powi(3)];
    let sa = (10.0 * pa[0] + 20.0 * pa[1]) / (pa[0] + pa[1]);
    let sb = (10.0 * pb[0] + 20.0 * pb[1]) / (pb[0] + pb[1]);
    assert!(
        (y - (sa - sb)).abs() < 1e-3,
        "graph {y} vs formula {}",
        sa - sb
    );
}

// -----------------------------------------------------------------------
// Denominator-constrained vertex bound (budget LP + Dinkelbach bisection)
// -----------------------------------------------------------------------

/// Sampled enclosure check: every feasible point of
/// `p ∈ [pl, pu] ∩ {Σp ∈ [dl, du]}` must have its mediant value inside the
/// claimed range, and the constrained range must never be WIDER than the
/// unconstrained exact vertex range.
#[test]
fn head_range_with_denominator_encloses_and_tightens() {
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) as f64) / f64::from(u32::MAX)
    };
    let mut tightened = 0usize;
    let mut total_checked = 0usize;
    for case in 0..200 {
        let n = 2 + (case % 5);
        let coeffs: Vec<f64> = (0..n).map(|_| (next() - 0.3) * 300.0).collect();
        let bias = (next() - 0.5) * 10.0;
        let mut pl = Vec::with_capacity(n);
        let mut pu = Vec::with_capacity(n);
        for i in 0..n {
            let a = if i == 0 { 0.1 + next() } else { next() };
            pl.push(a);
            pu.push(a + next() * 2.0);
        }
        let sum_lo: f64 = pl.iter().sum();
        let sum_hi: f64 = pu.iter().sum();
        // A strict sub-window of the box-implied sum range (the interesting
        // case: the budget constraint actually binds).
        let a = next();
        let b = next();
        let (fa, fb) = if a < b { (a, b) } else { (b, a) };
        let dl = sum_lo + fa * (sum_hi - sum_lo);
        let du = sum_lo + fb * (sum_hi - sum_lo);
        let Some((lo, hi)) = head_range_with_denominator(&pl, &pu, &coeffs, bias, dl, du) else {
            panic!("case {case}: constrained range unexpectedly None");
        };
        // Never wider than the unconstrained exact range (superset set).
        let (ulo, uhi) = head_range(&pl, &pu, &coeffs, bias).unwrap();
        assert!(
            lo >= ulo - 1e-9 && hi <= uhi + 1e-9,
            "case {case}: wider than box range"
        );
        if lo > ulo + 1e-9 || hi < uhi - 1e-9 {
            tightened += 1;
        }
        // Feasible sampled points stay inside (rejection sampling on Σp).
        let mut checked = 0usize;
        for _ in 0..3000 {
            let p: Vec<f64> = (0..n).map(|i| pl[i] + next() * (pu[i] - pl[i])).collect();
            let den: f64 = p.iter().sum();
            if den < dl || den > du {
                continue;
            }
            checked += 1;
            let num: f64 = p.iter().zip(&coeffs).map(|(&pi, &ci)| ci * pi).sum();
            let v = num / den + bias;
            assert!(
                v >= lo - 1e-9 && v <= hi + 1e-9,
                "case {case}: feasible point value {v} escaped [{lo}, {hi}]"
            );
        }
        total_checked += checked;
    }
    // Narrow tail windows can be unpopulated per-case; require coverage in
    // aggregate instead.
    assert!(
        total_checked > 10_000,
        "too few feasible samples ({total_checked})"
    );
    // The whole point of the constraint: it must actually tighten sometimes.
    assert!(
        tightened > 20,
        "budget constraint almost never tightened ({tightened}/200)"
    );
}

#[test]
fn head_range_with_denominator_fails_open() {
    // Denominator not provably positive.
    assert!(
        head_range_with_denominator(&[0.0, 0.0], &[1.0, 1.0], &[1.0, 2.0], 0.0, -1.0, 5.0)
            .is_none()
    );
    // Empty budget window after clamping.
    assert!(
        head_range_with_denominator(&[1.0, 1.0], &[2.0, 2.0], &[1.0, 2.0], 0.0, 10.0, 11.0)
            .is_none()
    );
    // Malformed box.
    assert!(
        head_range_with_denominator(&[2.0, 1.0], &[1.0, 2.0], &[1.0, 2.0], 0.0, 2.0, 4.0).is_none()
    );
}

// -----------------------------------------------------------------------
// Geometric refinement grid
// -----------------------------------------------------------------------

#[test]
fn geometric_offsets_are_fine_near_edge_and_cover_span() {
    let w = 8.0;
    let offs = geometric_offsets(w, STAGE_POINTS);
    assert_eq!(offs.len(), STAGE_POINTS - 1);
    // Strictly increasing, all in (0, w], finest step next to 0.
    let mut prev = 0.0;
    for &o in &offs {
        assert!(o > prev && o <= w + 1e-12);
        prev = o;
    }
    let finest = offs[0];
    let coarsest = offs[offs.len() - 1] - offs[offs.len() - 2];
    assert!(
        (offs[offs.len() - 1] - w).abs() < 1e-12,
        "must reach the full span"
    );
    assert!(finest <= w / 1000.0, "finest step {finest} too coarse");
    assert!(coarsest > finest * 10.0, "grid should be graded");
    // Degenerate inputs.
    assert!(geometric_offsets(0.0, STAGE_POINTS).is_empty());
    assert!(geometric_offsets(-1.0, STAGE_POINTS).is_empty());
    assert!(geometric_offsets(1.0, 1).is_empty());
}
