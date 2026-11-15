// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for #vjp-side-branches: routing (chain and resnet plans refuse, the
//! side-branch plan builds), forward parity with the graph's own concrete
//! forward, and the gradient: main-subgraph exact gradient + the plan's side
//! term == the full graph's exact gradient.

use super::super::point_vjp_batched_resnet::PointVjpWavePlan;
use crate::layers::{AddConstantLayer, AddLayer, Layer, LinearLayer, MulConstantLayer, ReLULayer};
use crate::network::core::{GraphNetwork, GraphNode};
use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;

struct Rng(u64);
impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

fn lin(rng: &mut Rng, o: usize, i: usize) -> Layer {
    let w = Array2::from_shape_fn((o, i), |_| rng.next_f32());
    let b = Array1::from_shape_fn(o, |_| rng.next_f32() * 0.3);
    Layer::Linear(LinearLayer::new(w, Some(b)).expect("linear"))
}

/// `out = main(x) + s1(x) + s2(x) + bias` with
/// main = lin(6→8) → relu → lin(8→4),
/// s1   = lin(6→5) → relu → lin(5→4),
/// s2   = lin(6→3) → *0.5 → relu → lin(3→4),
/// nested as `AddConstant(Add(Add(main, s1), s2))`.
/// With `bad_branch` BOTH the main chain and the second branch carry a SECOND
/// ReLU, so two addends are not side-branch shaped and the plan must refuse.
fn fixture(bad_branch: bool) -> (GraphNetwork, BoundedTensor) {
    let mut rng = Rng(0xC0FF_EE11);
    let mut g = GraphNetwork::new();
    g.add_node(GraphNode::from_input("m0", lin(&mut rng, 6, 6)));
    let m_in = if bad_branch {
        g.add_node(GraphNode::new(
            "mx",
            Layer::ReLU(ReLULayer),
            vec!["m0".into()],
        ));
        "mx"
    } else {
        "m0"
    };
    g.add_node(GraphNode::new("m1", lin(&mut rng, 8, 6), vec![m_in.into()]));
    g.add_node(GraphNode::new(
        "mr",
        Layer::ReLU(ReLULayer),
        vec!["m1".into()],
    ));
    g.add_node(GraphNode::new("m2", lin(&mut rng, 4, 8), vec!["mr".into()]));
    g.add_node(GraphNode::from_input("a1", lin(&mut rng, 5, 6)));
    g.add_node(GraphNode::new(
        "ar",
        Layer::ReLU(ReLULayer),
        vec!["a1".into()],
    ));
    g.add_node(GraphNode::new("a2", lin(&mut rng, 4, 5), vec!["ar".into()]));
    g.add_node(GraphNode::from_input("b1", lin(&mut rng, 3, 6)));
    let scale = Layer::MulConstant(
        MulConstantLayer::try_new(ArrayD::from_elem(IxDyn(&[1]), 0.5)).expect("mul"),
    );
    g.add_node(GraphNode::new("bs", scale, vec!["b1".into()]));
    let b_pre = if bad_branch {
        g.add_node(GraphNode::new(
            "bx",
            Layer::ReLU(ReLULayer),
            vec!["bs".into()],
        ));
        "bx"
    } else {
        "bs"
    };
    g.add_node(GraphNode::new(
        "br",
        Layer::ReLU(ReLULayer),
        vec![b_pre.into()],
    ));
    g.add_node(GraphNode::new("b2", lin(&mut rng, 4, 3), vec!["br".into()]));
    g.add_node(GraphNode::new(
        "add1",
        Layer::Add(AddLayer),
        vec!["m2".into(), "a2".into()],
    ));
    g.add_node(GraphNode::new(
        "add2",
        Layer::Add(AddLayer),
        vec!["add1".into(), "b2".into()],
    ));
    let bias = Layer::AddConstant(
        AddConstantLayer::try_new(
            ArrayD::from_shape_vec(IxDyn(&[4]), vec![0.25, -0.5, 0.0, 1.0]).expect("bias"),
        )
        .expect("addc"),
    );
    g.add_node(GraphNode::new("out", bias, vec!["add2".into()]));
    g.set_output("out");
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[6]), -1.0_f32),
        ArrayD::from_elem(IxDyn(&[6]), 1.0_f32),
    )
    .expect("box");
    (g, input)
}

fn points(rng: &mut Rng, n: usize) -> Vec<Vec<f32>> {
    (0..n)
        .map(|_| (0..6).map(|_| rng.next_f32()).collect())
        .collect()
}

#[test]
fn chain_and_resnet_refuse_and_the_side_branch_plan_routes() {
    let (g, input) = fixture(false);
    assert!(g.build_point_vjp_batch_plan(&input).is_none());
    assert!(g.build_point_vjp_resnet_plan(&input).is_none());
    match PointVjpWavePlan::build(&g, &input) {
        Some(PointVjpWavePlan::SideBranched(p)) => {
            assert_eq!(p.branches.len(), 2);
            assert!(matches!(p.main.as_ref(), PointVjpWavePlan::Chain(_)));
            assert_eq!((p.input_dim(), p.output_dim()), (6, 4));
        }
        _ => panic!("expected the side-branch plan"),
    }
}

#[test]
fn a_non_affine_branch_is_refused() {
    let (g, input) = fixture(true);
    assert!(g.build_point_vjp_side_branch_plan(&input).is_none());
}

#[test]
fn side_branch_forward_matches_the_graph_forward() {
    let (g, input) = fixture(false);
    let plan = g.build_point_vjp_side_branch_plan(&input).expect("plan");
    let pts = points(&mut Rng(7), 16);
    let (masks, outs) = plan.forward_masks(&pts).expect("forward");
    for ((x, y), m) in pts.iter().zip(&outs).zip(&masks) {
        assert_eq!(m.len(), plan.main_slots + 2);
        let arr = ArrayD::from_shape_vec(IxDyn(&[6]), x.clone()).unwrap();
        let full = g
            .propagate_concrete_point(&BoundedTensor::concrete(arr).unwrap(), None, None)
            .unwrap()
            .center();
        for (a, b) in full.iter().zip(y) {
            assert!((a - b).abs() <= 1e-5, "x={x:?} graph={full:?} plan={y:?}");
        }
    }
}

/// main-subgraph exact gradient + the plan's side term == the full graph's
/// exact gradient, at points on both sides of every branch ReLU.
#[test]
fn main_gradient_plus_side_term_is_the_full_exact_gradient() {
    let (g, input) = fixture(false);
    let plan = g.build_point_vjp_side_branch_plan(&input).expect("plan");
    let main = g.ancestor_subgraph("m2").expect("main subgraph");
    let pts = points(&mut Rng(99), 12);
    let (masks, _) = plan.forward_masks(&pts).expect("forward");
    let mut rng = Rng(5);
    let spec: Vec<f32> = (0..pts.len() * 4).map(|_| rng.next_f32()).collect();
    let mut grads: Vec<Vec<f32>> = Vec::new();
    for (k, x) in pts.iter().enumerate() {
        let arr = ArrayD::from_shape_vec(IxDyn(&[6]), x.clone()).unwrap();
        let row = Array2::from_shape_vec((1, 4), spec[k * 4..(k + 1) * 4].to_vec()).unwrap();
        let gm = main
            .attack_point_gradient(&arr, &row, None, None)
            .unwrap()
            .expect("main gradient");
        grads.push(gm.iter().copied().collect());
    }
    plan.add_side_vjp(&masks, &spec, &mut grads);
    let mut side_active = 0;
    for (k, x) in pts.iter().enumerate() {
        side_active += masks[k][plan.main_slots..]
            .iter()
            .flatten()
            .filter(|&&v| v > 0.0)
            .count();
        let arr = ArrayD::from_shape_vec(IxDyn(&[6]), x.clone()).unwrap();
        let row = Array2::from_shape_vec((1, 4), spec[k * 4..(k + 1) * 4].to_vec()).unwrap();
        let full = g
            .attack_point_gradient(&arr, &row, None, None)
            .unwrap()
            .expect("full gradient");
        for (a, b) in full.iter().zip(&grads[k]) {
            assert!(
                (a - b).abs() <= 1e-4,
                "x={x:?} full={full:?} plan={:?}",
                grads[k]
            );
        }
    }
    assert!(
        side_active > 0,
        "the fixture must exercise active side-branch units"
    );
}

// ---------------------------------------------------------------------------
// #sbench-f64-attack: the exact-f64 twin.
// ---------------------------------------------------------------------------

use super::PointVjpF64Plan;
use crate::network::graph_ibp_f64_cell::Interval64;

/// The graph's own sound f64 cell walk at a point, as the midpoint of its
/// (ULP-tight) enclosure — the reference the twin must reproduce.
fn exact_f64_forward(g: &GraphNetwork, x: &[f64]) -> Vec<f64> {
    let arr = ArrayD::from_shape_vec(IxDyn(&[x.len()]), x.to_vec()).unwrap();
    let out = g
        .propagate_ibp_f64_cell(&Interval64::point(arr))
        .expect("fixture is inside the f64 cell op set");
    out.lower
        .iter()
        .zip(out.upper.iter())
        .map(|(&lo, &hi)| 0.5 * lo + 0.5 * hi)
        .collect()
}

fn widen(x: &[f32]) -> Vec<f64> {
    x.iter().map(|&v| f64::from(v)).collect()
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .fold(0.0_f64, |m, (&x, &y)| m.max((x - y).abs()))
}

/// The twin's forward is the network's EXACT real function (to f64 rounding),
/// not the f32 template's rounded one: it agrees with the graph's own sound f64
/// cell walk to 1e-10 on the side-branch fixture, with both branches' pre AND
/// post maps exact. The fixture's post maps are BIASED `Linear`s, whose f32
/// unit-vector capture rounds `W + b` (review arithmetic, emulating this RNG:
/// one `b2` entry off by 2^-26 at a hidden unit active on 13 of these 24
/// points, ~1e-8 at the output against a 1e-10 tolerance), so this pins the
/// post re-capture too, not only the pre one.
#[test]
fn f64_twin_forward_matches_the_exact_f64_cell_walk() {
    let (g, input) = fixture(false);
    let twin = PointVjpF64Plan::build(&g, &input, None).expect("twin");
    assert_eq!(twin.kind(), "side");
    assert_eq!(twin.branch_count(), 2);
    assert_eq!(
        twin.exact_branch_maps(),
        2,
        "both branches' pre and post maps must be exact (re-captured through the f64 cell)"
    );
    assert_eq!((twin.input_dim(), twin.output_dim()), (6, 4));
    for x in points(&mut Rng(31), 24) {
        let x64 = widen(&x);
        let (y, _) = twin.forward(&x64).expect("forward");
        let reference = exact_f64_forward(&g, &x64);
        let scale = 1.0 + reference.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        assert!(
            max_abs_diff(&y, &reference) <= 1e-10 * scale,
            "x={x:?} twin={y:?} exact={reference:?}"
        );
    }
}

/// Without the exact re-capture the twin would inherit the f32 unit-vector
/// capture's rounding — the same order (~1e-7) as the SoundnessBench margins it
/// exists to resolve. Pin that the widened capture is measurably inexact and the
/// re-captured one is not, so a regression to widening cannot pass silently.
#[test]
fn f64_twin_exact_recapture_removes_the_f32_capture_rounding() {
    let (g, input) = fixture(false);
    let plan = g.build_point_vjp_side_branch_plan(&input).expect("plan");
    let widened = PointVjpF64Plan::from_side_branch_plan(None, &plan, None).expect("widened");
    let exact = PointVjpF64Plan::from_side_branch_plan(Some(&g), &plan, None).expect("exact");
    assert_eq!(widened.exact_branch_maps(), 0);
    assert_eq!(exact.exact_branch_maps(), 2);
    let mut widened_err = 0.0_f64;
    let mut exact_err = 0.0_f64;
    let mut active = 0usize;
    for x in points(&mut Rng(77), 32) {
        let x64 = widen(&x);
        let reference = exact_f64_forward(&g, &x64);
        let (yw, tape) = widened.forward(&x64).expect("widened forward");
        let (ye, _) = exact.forward(&x64).expect("exact forward");
        active += tape.branch_masks.iter().flatten().filter(|&&on| on).count();
        widened_err = widened_err.max(max_abs_diff(&yw, &reference));
        exact_err = exact_err.max(max_abs_diff(&ye, &reference));
    }
    assert!(
        active > 0,
        "the fixture must exercise active side-branch units"
    );
    assert!(exact_err <= 1e-10, "exact re-capture error {exact_err:e}");
    assert!(
        widened_err > 1e-12 && widened_err < 1e-4,
        "the widened f32 capture should be close but not exact (err {widened_err:e})"
    );
}

/// The f32 post capture is trusted as exact ONLY for a single bias-free
/// `Linear` (plus data movement): the fixture's biased post `Linear`s must be
/// re-captured, because their f32 capture stores `fl(fl(W + b) - b)`.
#[test]
fn f64_twin_trusts_the_f32_post_capture_only_when_it_is_provably_exact() {
    let (g, input) = fixture(false);
    let plan = g.build_point_vjp_side_branch_plan(&input).expect("plan");
    assert_eq!(plan.branches.len(), 2);
    for br in &plan.branches {
        assert!(
            br.b.iter().any(|&v| v != 0.0),
            "the fixture's post Linears carry a bias"
        );
        assert!(
            !super::f32_post_capture_is_exact(&g, br),
            "a biased post capture rounds W + b and must not be trusted"
        );
        // The same single-`Linear` structure with a zero captured bias is the
        // bias-free carrier shape, whose capture is exact.
        let mut unbiased = br.clone();
        unbiased.b.iter_mut().for_each(|v| *v = 0.0);
        assert!(super::f32_post_capture_is_exact(&g, &unbiased));
    }
}

/// The twin's VJP is the exact derivative of its own forward (central f64
/// differences inside one activation region) and agrees with the graph's
/// exact f32 point gradient.
#[test]
fn f64_twin_vjp_is_the_exact_derivative_of_its_forward() {
    let (g, input) = fixture(false);
    let twin = PointVjpF64Plan::build(&g, &input, None).expect("twin");
    let mut rng = Rng(12);
    let mut checked_fd = 0usize;
    for x in points(&mut Rng(4242), 16) {
        let x64 = widen(&x);
        let s: Vec<f64> = (0..4).map(|_| f64::from(rng.next_f32())).collect();
        let (_, tape) = twin.forward(&x64).expect("forward");
        let grad = twin.vjp(&tape, &s).expect("vjp");
        assert_eq!(grad.len(), 6);

        // Against the graph's exact f32 point gradient.
        let arr = ArrayD::from_shape_vec(IxDyn(&[6]), x).unwrap();
        let row = Array2::from_shape_vec((1, 4), s.iter().map(|&v| v as f32).collect()).unwrap();
        let full = g
            .attack_point_gradient(&arr, &row, None, None)
            .unwrap()
            .expect("full gradient");
        for (a, b) in full.iter().zip(&grad) {
            assert!(
                (f64::from(*a) - b).abs() <= 1e-4,
                "f32={full:?} f64={grad:?}"
            );
        }

        // Against central f64 differences, where no unit changes phase.
        let h = 1e-6;
        let dot = |y: &[f64]| y.iter().zip(&s).map(|(a, b)| a * b).sum::<f64>();
        for d in 0..6 {
            let mut xp = x64.clone();
            let mut xm = x64.clone();
            xp[d] += h;
            xm[d] -= h;
            let (yp, tp) = twin.forward(&xp).expect("forward +h");
            let (ym, tm) = twin.forward(&xm).expect("forward -h");
            let same_region = tp.main_masks == tape.main_masks
                && tm.main_masks == tape.main_masks
                && tp.branch_masks == tape.branch_masks
                && tm.branch_masks == tape.branch_masks;
            if !same_region {
                continue;
            }
            let fd = (dot(&yp) - dot(&ym)) / (2.0 * h);
            assert!(
                (fd - grad[d]).abs() <= 1e-7 * (1.0 + grad[d].abs()),
                "d={d} fd={fd} vjp={}",
                grad[d]
            );
            checked_fd += 1;
        }

        // Several rows at once agree with one-at-a-time.
        let rows = vec![s.clone(), s.iter().map(|v| -2.0 * v).collect()];
        let batched = twin.vjp_rows(&tape, &rows).expect("vjp rows");
        assert_eq!(batched[0], grad);
        for (a, b) in batched[1].iter().zip(&grad) {
            assert!((a + 2.0 * b).abs() <= 1e-12 * (1.0 + b.abs()));
        }
    }
    assert!(
        checked_fd > 0,
        "at least one in-region difference must be checked"
    );
}

/// A pure chain gets the chain twin; it matches both the f32 template (to f32
/// accuracy) and the exact f64 cell walk (to f64 accuracy).
#[test]
fn f64_twin_of_a_pure_chain_matches_template_and_cell() {
    let (g, input) = fixture(false);
    let main = g.ancestor_subgraph("m2").expect("main subgraph");
    let twin = PointVjpF64Plan::build(&main, &input, None).expect("chain twin");
    assert_eq!(twin.kind(), "chain");
    assert_eq!(twin.branch_count(), 0);
    let chain = main.build_point_vjp_batch_plan(&input).expect("chain plan");
    let pts = points(&mut Rng(9), 12);
    let (_, outs32) =
        super::super::point_vjp_batched::point_vjp_forward_masks(&chain, &pts).expect("f32");
    for (x, y32) in pts.iter().zip(&outs32) {
        let x64 = widen(x);
        let (y, _) = twin.forward(&x64).expect("forward");
        assert!(max_abs_diff(&y, &widen(y32)) <= 1e-4);
        let reference = exact_f64_forward(&main, &x64);
        assert!(
            max_abs_diff(&y, &reference)
                <= 1e-10 * (1.0 + y.iter().fold(0.0_f64, |m, v| m.max(v.abs())))
        );
    }
}

/// The twin never widens the fragment: a graph both f32 plans refuse gets no
/// twin, and a wrong-length input is refused rather than read out of bounds.
#[test]
fn f64_twin_refuses_what_the_f32_plans_refuse() {
    let (bad, bad_input) = fixture(true);
    assert!(PointVjpF64Plan::build(&bad, &bad_input, None).is_none());
    let (g, input) = fixture(false);
    let twin = PointVjpF64Plan::build(&g, &input, None).expect("twin");
    assert!(twin.forward(&[0.0; 5]).is_none());
    assert!(twin.forward(&[f64::NAN; 6]).is_none());
    let (_, tape) = twin.forward(&[0.1; 6]).expect("forward");
    assert!(twin.vjp(&tape, &[1.0; 3]).is_none());
}
