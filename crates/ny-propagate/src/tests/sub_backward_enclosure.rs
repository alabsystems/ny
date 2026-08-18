// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! DECISIVE enclosure probe for `SubLayer::propagate_linear_binary`
//! (`layers/binary_ops/sub.rs`).
//!
//! Question settled: when CROWN backward rewrites an accumulated relation
//! `lA·y + lb <= f <= uA·y + ub` (with `y = a - b`) in terms of the operands,
//! the B-side carrier must be `(-lA, -uA)` (negate only — exact substitution,
//! matching `SubConstantLayer` reverse and `MulConstantLayer` negative-scale).
//! The former `(-uA, -lA)` (negate AND swap) form was NOT enclosing: it
//! produced a demonstrably FALSE upper bound on `ReLU(a - b)` (see the cases
//! below; `a ∈ [-1,-0.5]`, `b ∈ [-1,0]` gave CROWN upper 0.1667 against an
//! attainable 0.5).
//!
//! The probe builds a *valid* incoming relation (a textbook ReLU relaxation of
//! `f = ReLU(y)` over the exact range of `y`), pushes it through ny's Sub
//! backward, merges the two operand carriers into one relation over the joint
//! `(a, b)` box exactly as the graph accumulator does when `a` and `b` are
//! distinct input coordinates, concretizes, and compares against brute-force
//! min/max of `ReLU(a - b)` on a dense grid.

use std::collections::HashMap;

use ndarray::{arr1, arr2, Array1, Array2, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;

use crate::layers::binary_ops::SubLayer;
use crate::layers::{LinearLayer, ReLULayer};
use crate::{GraphNetwork, GraphNode, Layer, LinearBounds};

// ── helpers ────────────────────────────────────────────────────────────────

fn box2(al: f32, au: f32, bl: f32, bu: f32) -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![al, bl]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![au, bu]).unwrap(),
    )
    .unwrap()
}

/// Ground truth: min/max of `ReLU(a - b)` over a dense grid of the operand box.
fn brute_force_relu_sub(al: f32, au: f32, bl: f32, bu: f32, n: usize) -> (f32, f32) {
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    for i in 0..=n {
        let a = al + (au - al) * (i as f32 / n as f32);
        for j in 0..=n {
            let b = bl + (bu - bl) * (j as f32 / n as f32);
            let v = (a - b).max(0.0);
            lo = lo.min(v);
            hi = hi.max(v);
        }
    }
    (lo, hi)
}

/// A textbook, indisputably valid CROWN relaxation of `f = ReLU(y)` on
/// `y ∈ [l, u]` with `l < 0 < u`:
///   lower: `alpha·y`                  (valid for any alpha ∈ [0, 1])
///   upper: `s·y - s·l`, `s = u/(u-l)` (the chord through (l,0) and (u,u))
///
/// This is the same shape ny's own `relu_crown_relaxation` produces, with
/// `alpha` pinned so the probe can sweep the lower-slope choice.
fn relu_relaxation_over(l: f32, u: f32, alpha: f32) -> LinearBounds {
    assert!(l < 0.0 && u > 0.0, "case must be an unstable ReLU");
    let s = u / (u - l);
    let t = -s * l;
    LinearBounds::new(
        Array2::from_shape_vec((1, 1), vec![alpha]).unwrap(),
        Array1::from_vec(vec![0.0]),
        Array2::from_shape_vec((1, 1), vec![s]).unwrap(),
        Array1::from_vec(vec![t]),
    )
    .unwrap()
}

/// Merge the two per-operand carriers into ONE relation over the joint `(a, b)`
/// input vector — column 0 is `a`, column 1 is `b` — which is exactly what the
/// graph accumulator builds when `a` and `b` are distinct input coordinates
/// (the split half-biases sum back to the original bias).
fn merge_carriers(ba: &LinearBounds, bb: &LinearBounds) -> LinearBounds {
    LinearBounds::new(
        arr2(&[[ba.lower_a()[[0, 0]], bb.lower_a()[[0, 0]]]]),
        arr1(&[ba.lower_b()[0] + bb.lower_b()[0]]),
        arr2(&[[ba.upper_a()[[0, 0]], bb.upper_a()[[0, 0]]]]),
        arr1(&[ba.upper_b()[0] + bb.upper_b()[0]]),
    )
    .unwrap()
}

/// Exact interval concretization in f64, WITHOUT ny's `lower > upper` inversion
/// repair — so a bound that is wrong in both directions is reported as the
/// literal numbers CROWN computed rather than as the repaired `[-inf, +inf]`.
fn raw_concretize(lb: &LinearBounds, dom: &BoundedTensor) -> (f64, f64) {
    let dom = dom.flatten();
    let mut lo = lb.lower_b()[0] as f64;
    let mut hi = lb.upper_b()[0] as f64;
    for j in 0..lb.num_inputs() {
        let (il, iu) = (dom.lower()[j] as f64, dom.upper()[j] as f64);
        let la = lb.lower_a()[[0, j]] as f64;
        let ua = lb.upper_a()[[0, j]] as f64;
        lo += la.max(0.0) * il + la.min(0.0) * iu;
        hi += ua.max(0.0) * iu + ua.min(0.0) * il;
    }
    (lo, hi)
}

/// The candidate FIX: negate without swapping (exact substitution). Mirrors
/// `sub.rs` byte-for-byte except for the B-side coefficient assignment, so the
/// same cases can be re-scored under the corrected rule.
fn sub_backward_negate_no_swap(bounds: &LinearBounds) -> (LinearBounds, LinearBounds) {
    let lower_b_half = bounds.lower_b().mapv(|v| ny_tensor::next_down_f32(v * 0.5));
    let upper_b_half = bounds.upper_b().mapv(|v| ny_tensor::next_up_f32(v * 0.5));
    let ba = LinearBounds::new_or_conservative(
        bounds.lower_a().clone(),
        lower_b_half.clone(),
        bounds.upper_a().clone(),
        upper_b_half.clone(),
    )
    .unwrap();
    let bb = LinearBounds::new_or_conservative(
        -bounds.lower_a(),
        lower_b_half,
        -bounds.upper_a(),
        upper_b_half,
    )
    .unwrap();
    (ba, bb)
}

struct Case {
    name: &'static str,
    a: (f32, f32),
    b: (f32, f32),
    alpha: f32,
}

const CASES: &[Case] = &[
    // The reviewer's exact predicted regime: both operands in [-1, 0].
    Case {
        name: "a,b in [-1,0], alpha=0",
        a: (-1.0, 0.0),
        b: (-1.0, 0.0),
        alpha: 0.0,
    },
    Case {
        name: "a,b in [-1,0], alpha=1",
        a: (-1.0, 0.0),
        b: (-1.0, 0.0),
        alpha: 1.0,
    },
    Case {
        name: "a,b in [-1,0], alpha=0.5",
        a: (-1.0, 0.0),
        b: (-1.0, 0.0),
        alpha: 0.5,
    },
    // Negative operands, alpha=0 is what ny's own adaptive rule picks here
    // (upper 0.5 <= -lower 1.0) — one-sided FALSE UPPER, no inversion repair.
    Case {
        name: "a in [-1,-0.5], b in [-1,0], alpha=0 (ny-adaptive)",
        a: (-1.0, -0.5),
        b: (-1.0, 0.0),
        alpha: 0.0,
    },
    // Negative operands where BOTH directions break (lower > upper).
    Case {
        name: "a in [-2,-1.5], b in [-1.6,-1], alpha=0 (ny-adaptive)",
        a: (-2.0, -1.5),
        b: (-1.6, -1.0),
        alpha: 0.0,
    },
    // Positive-operand mirror in the alpha=1 regime.
    Case {
        name: "a in [1.5,2], b in [1.25,2], alpha=1 (ny-adaptive)",
        a: (1.5, 2.0),
        b: (1.25, 2.0),
        alpha: 1.0,
    },
];

// ── the probe ──────────────────────────────────────────────────────────────

#[test]
fn sub_crown_backward_encloses_relu_of_difference() {
    let mut failures: Vec<String> = Vec::new();
    let layer = SubLayer;

    for case in CASES {
        let (al, au) = case.a;
        let (bl, bu) = case.b;
        let dom = box2(al, au, bl, bu);
        // Exact range of y = a - b (operands are independent coordinates).
        let l = al - bu;
        let u = au - bl;
        let bounds = relu_relaxation_over(l, u, case.alpha);

        let (ba, bb) = layer.propagate_linear_binary(&bounds).unwrap();
        let shipped = merge_carriers(&ba, &bb);
        let (lo, hi) = raw_concretize(&shipped, &dom);
        let shipped_repaired = shipped.concretize(&dom);

        let (fix_ba, fix_bb) = sub_backward_negate_no_swap(&bounds);
        let (fix_lo, fix_hi) = raw_concretize(&merge_carriers(&fix_ba, &fix_bb), &dom);

        let (true_lo, true_hi) = brute_force_relu_sub(al, au, bl, bu, 400);

        println!(
            "case {:<46} y=[{:+.4},{:+.4}] a=[{:+.2},{:+.2}] b=[{:+.2},{:+.2}] alpha={:.2}\n     \
             ny sub.rs (as built) = [{:+.6}, {:+.6}]  (after ny inversion repair: [{}, {}])\n     \
             negate-only reference= [{:+.6}, {:+.6}]\n     \
             ground truth         = [{:+.6}, {:+.6}]",
            case.name,
            l,
            u,
            al,
            au,
            bl,
            bu,
            case.alpha,
            lo,
            hi,
            shipped_repaired.lower()[0],
            shipped_repaired.upper()[0],
            fix_lo,
            fix_hi,
            true_lo,
            true_hi
        );

        let tol = 1e-5_f64;
        // Score the bound ny would actually hand downstream (post-repair).
        let used_lo = shipped_repaired.lower()[0] as f64;
        let used_hi = shipped_repaired.upper()[0] as f64;
        if used_lo > true_lo as f64 + tol {
            failures.push(format!(
                "FALSE LOWER BOUND [{}]: a∈[{},{}] b∈[{},{}] alpha={} → ny lower={} \
                 but ReLU(a-b)={} is attainable",
                case.name, al, au, bl, bu, case.alpha, used_lo, true_lo
            ));
        }
        if used_hi < true_hi as f64 - tol {
            failures.push(format!(
                "FALSE UPPER BOUND [{}]: a∈[{},{}] b∈[{},{}] alpha={} → ny upper={} \
                 but ReLU(a-b)={} is attainable",
                case.name, al, au, bl, bu, case.alpha, used_hi, true_hi
            ));
        }
        // The candidate fix must enclose on every case, pre-repair.
        assert!(
            fix_lo <= true_lo as f64 + tol && fix_hi >= true_hi as f64 - tol,
            "negate-without-swap failed to enclose on {}: [{}, {}] vs truth [{}, {}]",
            case.name,
            fix_lo,
            fix_hi,
            true_lo,
            true_hi
        );
    }

    assert!(
        failures.is_empty(),
        "Sub CROWN backward produced non-enclosing bounds:\n{}",
        failures.join("\n")
    );
}

// ── end-to-end through the shipped graph CROWN backward ────────────────────

/// x ∈ R² → a = x0, b = x1 → sub = a - b → relu.
fn sub_relu_graph() -> GraphNetwork {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "a",
        Layer::Linear(LinearLayer::new(arr2(&[[1.0_f32, 0.0]]), Some(arr1(&[0.0_f32]))).unwrap()),
    ));
    graph.add_node(GraphNode::from_input(
        "b",
        Layer::Linear(LinearLayer::new(arr2(&[[0.0_f32, 1.0]]), Some(arr1(&[0.0_f32]))).unwrap()),
    ));
    graph.add_node(GraphNode::new(
        "sub",
        Layer::Sub(SubLayer),
        vec!["a".to_string(), "b".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["sub".to_string()],
    ));
    graph.set_output("relu");
    graph
}

fn assert_graph_crown_encloses(al: f32, au: f32, bl: f32, bu: f32, label: &str) {
    let graph = sub_relu_graph();
    let input = box2(al, au, bl, bu);
    let ibp = graph.collect_node_bounds(&input).unwrap();
    let crown = graph
        .propagate_crown_to_node(
            &input,
            "relu",
            &HashMap::new(),
            &ibp,
            None,
            None,
            None,
            None,
        )
        .unwrap();

    let (true_lo, true_hi) = brute_force_relu_sub(al, au, bl, bu, 400);
    let relu_ibp = ibp.get("relu").unwrap();
    println!(
        "graph[{label}]: CROWN relu = [{:+.6}, {:+.6}] | IBP relu = [{:+.6}, {:+.6}] | truth = [{:+.6}, {:+.6}]",
        crown.lower()[0],
        crown.upper()[0],
        relu_ibp.lower()[0],
        relu_ibp.upper()[0],
        true_lo,
        true_hi
    );

    let tol = 1e-5_f32;
    assert!(
        crown.lower()[0] <= true_lo + tol,
        "[{label}] FALSE LOWER BOUND from graph CROWN: {} > attainable {}",
        crown.lower()[0],
        true_lo
    );
    assert!(
        crown.upper()[0] >= true_hi - tol,
        "[{label}] FALSE UPPER BOUND from graph CROWN: {} < attainable {}",
        crown.upper()[0],
        true_hi
    );
}

/// One-sided violation: `lower <= upper` still holds, so ny's inversion repair
/// never fires and the false bound is handed downstream verbatim.
#[test]
fn graph_crown_sub_relu_one_sided_encloses() {
    assert_graph_crown_encloses(-1.0, -0.5, -1.0, 0.0, "one-sided");
}

/// Two-sided violation: the computed `lower > upper`, which ny's concretize
/// repairs to `[-inf, +inf]` (sound but vacuous) — a symptom, not a fix.
#[test]
fn graph_crown_sub_relu_two_sided_encloses() {
    assert_graph_crown_encloses(-2.0, -1.5, -1.6, -1.0, "two-sided");
}
