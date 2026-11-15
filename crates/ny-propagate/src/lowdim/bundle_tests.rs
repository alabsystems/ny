// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #lowdim-bundle soundness tests.
//!
//! * (1) `Op::Sigmoid`: every plane encloses the sigmoid in EXACT RATIONAL
//!   arithmetic against an independent enclosure (with a negative control
//!   showing the falsifier bites); lane bounds on a sigmoid graph never exceed
//!   a sampled value; the lane never verifies a sampled-violated threshold.
//!   Varying-dims admission: the folded program IS the full program on the
//!   box, its bounds contain sampled values, a candidate is expanded to the
//!   full input, and the lane never verifies a violated threshold.
//! * (2) Boundary alpha: alpha bounds never exceed a sampled value, never fall
//!   below the unrefined bound, their planes hold on the box; zero iterations
//!   is the historical pass bit for bit.
//! * (3) Leaf attack: a hit is inside its leaf and violates on the lowering;
//!   a true margin never yields one; the lane still proves true margins.
//! * Every key is dark by default (`LowDimConfig::default()`), and unarmed
//!   the historical declines are unchanged.
//! * #m8-cgan-leaf-batch (dist_shift): a one-ulp-widened declared point is a
//!   NARROW pin folded as its interval (bounds contain the full program's
//!   values at every point of the pinned intervals); armed, an equal-count
//!   reshape to an image and an outer-axis concat lower as flat relabels and
//!   agree with the graph's own forward, while an inner-axis concat declines;
//!   the dist_shift-shaped lane never verifies a violated threshold and
//!   expands its candidate inside the widened box.
//!
//! The shared worklist's tests (4) live with the fold's fixtures in
//! `fold_tests`.

use std::time::{Duration, Instant};

use ndarray::{Array1, Array2, ArrayD, IxDyn};
use num_rational::BigRational;
use ny_tensor::BoundedTensor;

use super::search::{bound_chunk, AlphaSpec};
use super::*;
use crate::layers::{ConcatLayer, LinearLayer, ReLULayer, ReshapeLayer, SigmoidLayer, SliceLayer};
use crate::network::GraphNode;

/// Deterministic LCG in `[-1, 1)`.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
}

fn q(v: f64) -> BigRational {
    BigRational::from_float(v).expect("finite test value")
}

fn linear(rng: &mut Lcg, n_out: usize, n_in: usize, scale: f64) -> Layer {
    let w = Array2::from_shape_fn((n_out, n_in), |_| (rng.next() * scale) as f32);
    let b = Array1::from_shape_fn(n_out, |_| (rng.next() * 0.3) as f32);
    Layer::Linear(LinearLayer::new(w, Some(b)).expect("linear"))
}

fn boxed(lo: &[f64], hi: &[f64]) -> BoundedTensor {
    let d = lo.len();
    let l = ArrayD::from_shape_vec(IxDyn(&[1, d]), lo.iter().map(|&v| v as f32).collect())
        .expect("shape");
    let h = ArrayD::from_shape_vec(IxDyn(&[1, d]), hi.iter().map(|&v| v as f32).collect())
        .expect("shape");
    BoundedTensor::new(l, h).expect("box")
}

fn sample(rng: &mut Lcg, lo: &[f64], hi: &[f64]) -> Vec<f64> {
    lo.iter()
        .zip(hi)
        .map(|(l, h)| l + (h - l) * 0.5 * (rng.next() + 1.0))
        .collect()
}

/// f32-representable root box (the f64 program box and the f32 tensor agree).
fn root_box(d: usize) -> (Vec<f64>, Vec<f64>) {
    let lo = (0..d)
        .map(|i| f64::from(-0.75f32 + 0.125 * i as f32))
        .collect();
    let hi = (0..d)
        .map(|i| f64::from(0.5f32 + 0.25 * i as f32))
        .collect();
    (lo, hi)
}

/// `Linear -> ReLU -> Linear -> Sigmoid -> Linear` over a `d`-input box.
fn sigmoid_graph(seed: u64, d: usize) -> GraphNetwork {
    let mut rng = Lcg(seed);
    let mut g = GraphNetwork::new();
    g.add_node(GraphNode::new(
        "l1",
        linear(&mut rng, 8, d, 1.5),
        vec![NETWORK_INPUT.into()],
    ));
    g.add_node(GraphNode::new(
        "r1",
        Layer::ReLU(ReLULayer::new()),
        vec!["l1".into()],
    ));
    g.add_node(GraphNode::new(
        "l2",
        linear(&mut rng, 6, 8, 1.2),
        vec!["r1".into()],
    ));
    g.add_node(GraphNode::new(
        "s",
        Layer::Sigmoid(SigmoidLayer::new()),
        vec!["l2".into()],
    ));
    g.add_node(GraphNode::new(
        "out",
        linear(&mut rng, 2, 6, 2.0),
        vec!["s".into()],
    ));
    g.set_output("out");
    g
}

/// A ReLU trunk over a `d`-input box (the boundary-alpha and leaf-attack
/// fixture): `Linear -> ReLU -> Linear -> ReLU -> Linear(2)`.
fn relu_graph(seed: u64, d: usize, width: usize) -> GraphNetwork {
    let mut rng = Lcg(seed);
    let mut g = GraphNetwork::new();
    g.add_node(GraphNode::new(
        "l1",
        linear(&mut rng, width, d, 1.5),
        vec![NETWORK_INPUT.into()],
    ));
    g.add_node(GraphNode::new(
        "r1",
        Layer::ReLU(ReLULayer::new()),
        vec!["l1".into()],
    ));
    g.add_node(GraphNode::new(
        "l2",
        linear(&mut rng, width, width, 0.7),
        vec!["r1".into()],
    ));
    g.add_node(GraphNode::new(
        "r2",
        Layer::ReLU(ReLULayer::new()),
        vec!["l2".into()],
    ));
    g.add_node(GraphNode::new(
        "out",
        linear(&mut rng, 2, width, 0.7),
        vec!["r2".into()],
    ));
    g.set_output("out");
    g
}

/// A dist_shift-shaped graph over 24 inputs: `Linear(24 -> 10) -> ReLU ->
/// Linear -> Sigmoid`, concatenated with a `Slice` of the raw input (so the
/// embed path is exercised, not only the folded input Linear), then a head.
const PINNED_D: usize = 24;
const PINNED_FREE: [usize; 2] = [3, 21];

fn pinned_graph(seed: u64) -> GraphNetwork {
    let mut rng = Lcg(seed);
    let mut g = GraphNetwork::new();
    let inp = NETWORK_INPUT.to_string();
    g.add_node(GraphNode::new(
        "l1",
        linear(&mut rng, 10, PINNED_D, 0.8),
        vec![inp.clone()],
    ));
    g.add_node(GraphNode::new(
        "r1",
        Layer::ReLU(ReLULayer::new()),
        vec!["l1".into()],
    ));
    g.add_node(GraphNode::new(
        "l2",
        linear(&mut rng, 12, 10, 1.0),
        vec!["r1".into()],
    ));
    g.add_node(GraphNode::new(
        "s",
        Layer::Sigmoid(SigmoidLayer::new()),
        vec!["l2".into()],
    ));
    g.add_node(GraphNode::new(
        "sl",
        Layer::Slice(SliceLayer::new(1, 20, 22)),
        vec![inp],
    ));
    g.add_node(GraphNode::new(
        "cat",
        Layer::Concat(ConcatLayer::new(1)),
        vec!["s".into(), "sl".into()],
    ));
    g.add_node(GraphNode::new(
        "out",
        linear(&mut rng, 2, 14, 1.5),
        vec!["cat".into()],
    ));
    g.set_output("out");
    g
}

/// The 24-input box: every coordinate pinned to an f32 value except
/// [`PINNED_FREE`], which get f32-representable intervals.
fn pinned_box() -> (Vec<f64>, Vec<f64>) {
    let mut rng = Lcg(99);
    let mut lo = vec![0.0f64; PINNED_D];
    for v in &mut lo {
        *v = f64::from((rng.next() * 0.9) as f32);
    }
    let mut hi = lo.clone();
    for (k, &i) in PINNED_FREE.iter().enumerate() {
        lo[i] = f64::from(-0.5f32 - 0.25 * k as f32);
        hi[i] = f64::from(0.75f32 + 0.125 * k as f32);
    }
    (lo, hi)
}

fn armed_varying() -> LowDimConfig {
    LowDimConfig {
        batch: 64,
        chunk: 8,
        varying_dims: true,
        ..LowDimConfig::default()
    }
}

fn plain_cfg() -> LowDimConfig {
    LowDimConfig {
        batch: 64,
        chunk: 8,
        ..LowDimConfig::default()
    }
}

/// An independent enclosure of `sigmoid(z)` for the TEST: the f64 value with
/// a `1e-14` relative margin (libm's `exp` is within an ulp, so four orders
/// tighter than the production code's `1e-12`, which keeps this oracle from
/// flagging planes that touch the function at a tangent point), clamped to
/// `(0, 1)` — sigmoid never leaves it.
fn test_enclosure(z: f64) -> (f64, f64) {
    let s = 1.0 / (1.0 + (-z).exp());
    (
        (next_down(s * (1.0 - 1e-14)) - 1e-300).max(0.0),
        (next_up(s * (1.0 + 1e-14)) + 1e-300).min(1.0),
    )
}

/// Every `z` a plane check evaluates on `[l, u]`: both ends, 0 when inside,
/// the midpoint and a 65-point grid, clamped into the box.
fn plane_points(l: f64, u: f64) -> Vec<f64> {
    let mut zs = vec![l, u, 0.5 * l + 0.5 * u];
    if l < 0.0 && u > 0.0 {
        zs.push(0.0);
    }
    for k in 0..=64 {
        zs.push(l + (u - l) * (k as f64 / 64.0));
    }
    zs.into_iter().map(|z| z.max(l).min(u)).collect()
}

/// `true` when the plane `a z + b` is BELOW (`lower`) or ABOVE the sigmoid at
/// every point of [`plane_points`], checked in exact rational arithmetic
/// against [`test_enclosure`].
fn plane_holds(l: f64, u: f64, a: f64, b: f64, lower: bool) -> bool {
    plane_points(l, u).into_iter().all(|z| {
        let (lo, hi) = test_enclosure(z);
        let v = q(a) * q(z) + q(b);
        if lower {
            v <= q(lo)
        } else {
            v >= q(hi)
        }
    })
}

fn sigmoid_boxes() -> Vec<(f64, f64)> {
    let mut boxes = vec![
        (-4.0, 4.0),
        (-1e-3, 1e-3),
        (0.0, 0.0),
        (-50.0, -45.0),
        (45.0, 50.0),
        (-700.0, 3.0),
        (-3.0, 700.0),
        (-1e6, 1e6),
        (0.25, 0.25 + 1e-12),
        (-8.0, 0.0),
        (0.0, 8.0),
        (-2.0, -1.0),
        (1.0, 2.0),
        (-1e31, 1e31),
        (-0.5, 30.0),
        (-30.0, 0.5),
    ];
    let mut rng = Lcg(7);
    for _ in 0..400 {
        let c = rng.next() * 12.0;
        let w = 10f64.powf(rng.next() * 3.0 - 1.0);
        let (a, b) = (rng.next().abs(), rng.next().abs());
        boxes.push((c - w * a, c + w * b));
    }
    boxes
}

// ------------------------------------------------------------- (1) sigmoid

/// Every plane encloses the sigmoid on its box, checked EXACTLY (rational
/// arithmetic on the stored floats) against an independent enclosure; slopes
/// lie in `[0, 1/4]`; a non-finite or unordered box yields no plane.
#[test]
fn sigmoid_planes_enclose_the_function_in_exact_rational_arithmetic() {
    for (l, u) in sigmoid_boxes() {
        let [a_lo, b_lo, a_hi, b_hi] = sigmoid::planes(l, u).expect("finite ordered box");
        assert!(
            (0.0..=0.25).contains(&a_lo) && (0.0..=0.25).contains(&a_hi),
            "slopes {a_lo} {a_hi} on [{l}, {u}]"
        );
        assert!(b_lo.is_finite() && b_hi.is_finite(), "[{l}, {u}]");
        assert!(
            plane_holds(l, u, a_lo, b_lo, true),
            "SOUNDNESS: lower plane ({a_lo}, {b_lo}) above sigmoid on [{l}, {u}]"
        );
        assert!(
            plane_holds(l, u, a_hi, b_hi, false),
            "SOUNDNESS: upper plane ({a_hi}, {b_hi}) below sigmoid on [{l}, {u}]"
        );
    }
    assert!(sigmoid::planes(f64::NAN, 1.0).is_none());
    assert!(sigmoid::planes(-1.0, f64::INFINITY).is_none());
    assert!(sigmoid::planes(1.0, -1.0).is_none());
}

/// NEGATIVE CONTROL: the falsifier above bites. The naive planes a sloppy
/// port would use — the chord as a LOWER plane across 0, the tangent at 0 as
/// a lower plane, and a tangent at the midpoint with the rounded intercept
/// `sigmoid(t) - sigmoid'(t) t` on the concave side — are each caught.
#[test]
fn the_sigmoid_plane_falsifier_catches_naive_planes() {
    let sig = |z: f64| 1.0 / (1.0 + (-z).exp());
    let (l, u) = (-4.0, 4.0);
    let k = (sig(u) - sig(l)) / (u - l);
    assert!(
        !plane_holds(l, u, k, sig(l) - k * l, true),
        "the chord across 0 is not a lower plane and must be caught"
    );
    assert!(
        !plane_holds(l, u, 0.25, 0.5, true),
        "the tangent at 0 is not a lower plane and must be caught"
    );
    // On the concave side the tangent is an UPPER plane; as a lower plane it
    // is wrong everywhere but the touching point.
    let (l, u) = (0.5, 3.0);
    let t = f64::midpoint(l, u);
    let d = sig(t) * (1.0 - sig(t));
    assert!(
        !plane_holds(l, u, d, sig(t) - d * t, true),
        "a concave-side tangent is not a lower plane and must be caught"
    );
}

/// The planes are not vacuous: on a narrow box the lower and upper planes
/// pinch the function (the gap at the midpoint is tiny), and across 0 the
/// lower plane has a positive slope (not just the constant plane).
#[test]
fn sigmoid_planes_are_tight_on_narrow_boxes() {
    let mut rng = Lcg(11);
    for _ in 0..200 {
        let c = rng.next() * 10.0;
        let (l, u) = (c - 1e-4, c + 1e-4);
        let [a_lo, b_lo, a_hi, b_hi] = sigmoid::planes(l, u).expect("box");
        let m = f64::midpoint(l, u);
        let gap = (a_hi * m + b_hi) - (a_lo * m + b_lo);
        assert!((0.0..1e-7).contains(&gap), "gap {gap} at {c}");
    }
    let [a_lo, _, a_hi, _] = sigmoid::planes(-1.0, 1.0).expect("box");
    assert!(a_lo > 0.0 && a_hi > 0.0, "{a_lo} {a_hi}");
}

/// DARK BY DEFAULT: unarmed, a graph with a Sigmoid is declined exactly as
/// before (the historical decline names the op); armed, it lowers and the
/// lowered program is the graph at points.
#[test]
fn a_sigmoid_graph_is_declined_unarmed_and_lowered_armed() {
    assert!(!LowDimConfig::default().varying_dims);
    let d = 3;
    let g = sigmoid_graph(5, d);
    let (lo, hi) = root_box(d);
    let input = boxed(&lo, &hi);
    let out = prove_objective_lower_bound(
        &g,
        &input,
        &[1.0, 0.0],
        -10.0,
        Instant::now() + Duration::from_secs(5),
        &plain_cfg(),
    );
    assert!(
        matches!(&out, LowDimOutcome::Declined(r) if r.contains("Sigmoid")),
        "{out:?}"
    );
    assert!(lower_graph(&g, &input).is_err());
    let prog = lower_graph_opts(
        &g,
        &input,
        LowerOpts {
            sigmoid: true,
            flat_reshape: true,
        },
    )
    .expect("lowers armed");
    assert_eq!(sigmoid_ops(&prog), 1);
    let mut rng = Lcg(3);
    for _ in 0..40 {
        let x: Vec<f64> = sample(&mut rng, &lo, &hi)
            .into_iter()
            .map(|v| f64::from(v as f32))
            .collect();
        let y = g.propagate_ibp(&boxed(&x, &x)).expect("ibp");
        for j in 0..2 {
            let mut o = vec![0.0; 2];
            o[j] = 1.0;
            let p = point_objective(&prog, &x, &o);
            let (gl, gh) = (
                f64::from(y.lower()[IxDyn(&[0, j])]),
                f64::from(y.upper()[IxDyn(&[0, j])]),
            );
            assert!(
                p >= gl - 1e-3 && p <= gh + 1e-3,
                "program y{j}={p} outside graph IBP point enclosure [{gl}, {gh}]"
            );
        }
    }
}

/// Lane bounds through `Op::Sigmoid` never exceed a value the lowered
/// program attains (sampled points and every corner), on sub-boxes of
/// assorted sizes bounded in one batch with an odd chunk size.
#[test]
fn sigmoid_lane_bounds_never_exceed_a_sampled_value() {
    let d = 3;
    for seed in [1u64, 2, 3] {
        let g = sigmoid_graph(seed, d);
        let (lo, hi) = root_box(d);
        let prog = lower_graph_opts(
            &g,
            &boxed(&lo, &hi),
            LowerOpts {
                sigmoid: true,
                flat_reshape: true,
            },
        )
        .expect("lowers");
        let mut rng = Lcg(seed * 31 + 5);
        let (mut blo, mut bhi) = (Vec::new(), Vec::new());
        for i in 0..24 {
            let c = sample(&mut rng, &lo, &hi);
            let r = [0.6, 0.1, 0.01, 1e-4][i % 4];
            for k in 0..d {
                blo.push((c[k] - r).max(lo[k]));
                bhi.push((c[k] + r).min(hi[k]));
            }
        }
        let o = vec![rng.next(), rng.next()];
        let res = bound_batch(&prog, &blo, &bhi, 24, &o, 7);
        assert_eq!(res.len(), 24);
        for (b, (lb, _)) in res.iter().enumerate() {
            assert!(lb.is_finite(), "seed {seed} box {b}: non-finite bound");
            let (l, h) = (&blo[b * d..(b + 1) * d], &bhi[b * d..(b + 1) * d]);
            let mut min_seen = f64::INFINITY;
            for _ in 0..400 {
                min_seen = min_seen.min(point_objective(&prog, &sample(&mut rng, l, h), &o));
            }
            for corner in 0..(1 << d) {
                let x: Vec<f64> = (0..d)
                    .map(|k| if corner >> k & 1 == 1 { h[k] } else { l[k] })
                    .collect();
                min_seen = min_seen.min(point_objective(&prog, &x, &o));
            }
            assert!(
                *lb <= min_seen + 1e-9,
                "SOUNDNESS: seed {seed} box {b}: bound {lb} exceeds attained value {min_seen}"
            );
        }
    }
}

/// End to end through the armed lane: a threshold a sampled point already
/// violates never verifies; a threshold with a real margin does.
#[test]
fn the_sigmoid_lane_never_verifies_a_violated_threshold_and_proves_a_true_margin() {
    let d = 3;
    let g = sigmoid_graph(4, d);
    let (lo, hi) = root_box(d);
    let input = boxed(&lo, &hi);
    let prog = lower_graph_opts(
        &g,
        &input,
        LowerOpts {
            sigmoid: true,
            flat_reshape: true,
        },
    )
    .expect("lowers");
    let o = [1.0f64, 0.0];
    let mut rng = Lcg(77);
    let (mut min_seen, mut max_seen) = (f64::INFINITY, f64::NEG_INFINITY);
    for _ in 0..20_000 {
        let v = point_objective(&prog, &sample(&mut rng, &lo, &hi), &o);
        min_seen = min_seen.min(v);
        max_seen = max_seen.max(v);
    }
    let spread = (max_seen - min_seen).max(1e-3);
    let deadline = Instant::now() + Duration::from_secs(60);
    let cfg = armed_varying();
    let false_thr = (min_seen + 0.02 * spread) as f32;
    match prove_objective_lower_bound(&g, &input, &[1.0, 0.0], false_thr, deadline, &cfg) {
        LowDimOutcome::Verified(s) => {
            panic!("SOUNDNESS: verified lower > {false_thr} but a point attains {min_seen} ({s:?})")
        }
        LowDimOutcome::Unproved(..) => {}
        LowDimOutcome::Declined(r) => panic!("declined: {r}"),
    }
    let true_thr = (min_seen - 0.2 * spread) as f32;
    match prove_objective_lower_bound(&g, &input, &[1.0, 0.0], true_thr, deadline, &cfg) {
        LowDimOutcome::Verified(s) => assert!(s.closed >= 1),
        other => panic!("expected a proof of lower > {true_thr} (min seen {min_seen}): {other:?}"),
    }
}

// -------------------------------------------------------- (1) varying dims

/// DARK BY DEFAULT: unarmed, a 24-input box is declined as before, whatever
/// its pinned coordinates; armed, it is admitted over its 2 free ones.
#[test]
fn a_pinned_box_is_declined_unarmed_and_admitted_by_its_varying_dims_armed() {
    let g = pinned_graph(1);
    let (lo, hi) = pinned_box();
    let input = boxed(&lo, &hi);
    let out = prove_objective_lower_bound(
        &g,
        &input,
        &[1.0, 0.0],
        -100.0,
        Instant::now() + Duration::from_secs(5),
        &plain_cfg(),
    );
    assert!(
        matches!(&out, LowDimOutcome::Declined(r) if r.contains("input dimension")),
        "{out:?}"
    );
    assert!(admit(&g, &input).is_err());
    assert!(admit_with(&g, &input, &plain_cfg()).is_err());
    let adm = admit_with(&g, &input, &armed_varying()).expect("admitted armed");
    assert_eq!(adm.input_dim(), PINNED_FREE.len());
    // Too many varying coordinates still decline, armed.
    let (mut lo2, mut hi2) = (lo, hi);
    for i in 0..=MAX_INPUT_DIM {
        lo2[i] = -0.5;
        hi2[i] = 0.5;
    }
    assert!(admit_with(&g, &boxed(&lo2, &hi2), &armed_varying()).is_err());
}

/// The folded program IS the full program on the box: at 200 random free
/// points it agrees with the unfolded lowering evaluated at the expanded
/// point (the input `Linear` folded into an interval bias, the `Slice`
/// reading the free coordinate through the embed).
#[test]
fn the_folded_program_is_the_full_program_on_the_box() {
    let g = pinned_graph(2);
    let (lo, hi) = pinned_box();
    let full = lower_graph_opts(
        &g,
        &boxed(&lo, &hi),
        LowerOpts {
            sigmoid: true,
            flat_reshape: true,
        },
    )
    .expect("lowers");
    let fold = pinned::PinnedFold::of(&lo, &hi)
        .expect("admissible")
        .expect("wider than the cap");
    assert_eq!(fold.free_dim(), PINNED_FREE.len());
    let (folded, input_linears) = fold.apply(full.clone()).expect("folds");
    assert_eq!(input_linears, 1, "the input Linear must be folded");
    assert_eq!(folded.d, PINNED_FREE.len());
    fold.cross_check(&full, &folded).expect("cross-check");
    let (flo, fhi) = (fold.restrict(&lo), fold.restrict(&hi));
    let mut rng = Lcg(5);
    for _ in 0..200 {
        let xf = sample(&mut rng, &flo, &fhi);
        let a = point_values(&folded, &xf);
        let b = point_values(&full, &fold.expand(&xf));
        assert_eq!(a.len(), b.len());
        for (u, v) in a.iter().zip(&b) {
            assert!(
                (u - v).abs() <= 1e-9 * (1.0 + v.abs()),
                "folded {u} vs full {v}"
            );
        }
    }
    let x = fold.expand(&flo);
    for i in 0..PINNED_D {
        if !PINNED_FREE.contains(&i) {
            assert_eq!(
                x[i], lo[i],
                "a pinned coordinate must expand to its box value"
            );
        }
    }
}

/// Bounds of the folded admission contain the FULL program's sampled values
/// on sub-boxes of the free coordinates.
#[test]
fn folded_bounds_never_exceed_a_sampled_value_of_the_full_program() {
    let g = pinned_graph(3);
    let (lo, hi) = pinned_box();
    let input = boxed(&lo, &hi);
    let cfg = armed_varying();
    let adm = admit_with(&g, &input, &cfg).expect("admitted");
    let full = lower_graph_opts(
        &g,
        &input,
        LowerOpts {
            sigmoid: true,
            flat_reshape: true,
        },
    )
    .expect("lowers");
    let fold = pinned::PinnedFold::of(&lo, &hi)
        .expect("admissible")
        .expect("folds");
    let (flo, fhi) = (fold.restrict(&lo), fold.restrict(&hi));
    let d = flo.len();
    let mut rng = Lcg(17);
    let (mut blo, mut bhi) = (Vec::new(), Vec::new());
    for i in 0..16 {
        let c = sample(&mut rng, &flo, &fhi);
        let r = [0.5, 0.1, 0.01, 1e-4][i % 4];
        for k in 0..d {
            blo.push((c[k] - r).max(flo[k]));
            bhi.push((c[k] + r).min(fhi[k]));
        }
    }
    let objectives = vec![vec![1.0, 0.0], vec![-0.5, 1.0]];
    let lbs = bound_boxes(&adm, &blo, &bhi, &objectives, &cfg).expect("bounds");
    assert_eq!(lbs.len(), 16);
    for (b, row) in lbs.iter().enumerate() {
        let (l, h) = (&blo[b * d..(b + 1) * d], &bhi[b * d..(b + 1) * d]);
        for (o, lb) in objectives.iter().zip(row) {
            assert!(lb.is_finite(), "box {b}: non-finite bound");
            let mut min_seen = f64::INFINITY;
            for _ in 0..300 {
                let x = fold.expand(&sample(&mut rng, l, h));
                min_seen = min_seen.min(point_objective(&full, &x, o));
            }
            assert!(
                *lb <= min_seen + 1e-9,
                "SOUNDNESS: box {b}: folded bound {lb} exceeds the full program's {min_seen}"
            );
        }
    }
}

/// End to end, armed: a sampled-violated threshold never verifies and its
/// candidate comes back as a FULL-dimension point inside the box (pinned
/// coordinates at their box value) that violates on the full program; a true
/// margin verifies.
#[test]
fn the_varying_lane_never_verifies_a_violated_threshold_and_expands_its_candidate() {
    let g = pinned_graph(4);
    let (lo, hi) = pinned_box();
    let input = boxed(&lo, &hi);
    let full = lower_graph_opts(
        &g,
        &input,
        LowerOpts {
            sigmoid: true,
            flat_reshape: true,
        },
    )
    .expect("lowers");
    let fold = pinned::PinnedFold::of(&lo, &hi)
        .expect("admissible")
        .expect("folds");
    let (flo, fhi) = (fold.restrict(&lo), fold.restrict(&hi));
    let e0 = [1.0f64, 0.0];
    let mut rng = Lcg(23);
    let (mut min_seen, mut max_seen) = (f64::INFINITY, f64::NEG_INFINITY);
    for _ in 0..20_000 {
        let v = point_objective(&full, &fold.expand(&sample(&mut rng, &flo, &fhi)), &e0);
        min_seen = min_seen.min(v);
        max_seen = max_seen.max(v);
    }
    let spread = (max_seen - min_seen).max(1e-3);
    let deadline = Instant::now() + Duration::from_secs(60);
    let cfg = armed_varying();
    let t = (min_seen + 0.05 * spread) as f32;
    let mut w = None;
    match prove_first_output_witnessed(&g, &input, t, false, deadline, &cfg, &mut w) {
        LowDimOutcome::Verified(s) => {
            panic!("SOUNDNESS: verified y0 > {t} but a point attains {min_seen} ({s:?})")
        }
        LowDimOutcome::Unproved(_, LowDimStop::Candidate) => {
            let x = w.expect("a Candidate stop must hand back its point");
            assert_eq!(x.len(), PINNED_D, "the candidate must be full-dimension");
            for i in 0..PINNED_D {
                assert!(
                    lo[i] <= x[i] && x[i] <= hi[i],
                    "coordinate {i} outside the box"
                );
                if !PINNED_FREE.contains(&i) {
                    assert_eq!(x[i], lo[i], "pinned coordinate {i} moved");
                }
            }
            assert!(point_objective(&full, &x, &e0) <= f64::from(t));
        }
        LowDimOutcome::Unproved(..) => {}
        LowDimOutcome::Declined(r) => panic!("declined: {r}"),
    }
    let t = (min_seen - 0.2 * spread) as f32;
    match prove_first_output_witnessed(&g, &input, t, false, deadline, &cfg, &mut None) {
        LowDimOutcome::Verified(_) => {}
        other => panic!("expected a proof of y0 > {t} (min seen {min_seen}): {other:?}"),
    }
}

// ------------------------------------------------------ (2) boundary alpha

/// One chunk of 24 random sub-boxes of the ReLU fixture, its objective, and a
/// threshold that puts the chunk's domains in the alpha band.
fn alpha_fixture(seed: u64) -> (Program, Vec<f64>, Vec<f64>, usize, Vec<Vec<f64>>, f64) {
    let d = 3;
    let g = relu_graph(seed, d, 16);
    let (lo, hi) = root_box(d);
    let prog = lower_graph(&g, &boxed(&lo, &hi)).expect("lowers");
    let mut rng = Lcg(seed * 13 + 1);
    let (mut blo, mut bhi) = (Vec::new(), Vec::new());
    for i in 0..24 {
        let c = sample(&mut rng, &lo, &hi);
        let r = [0.6, 0.3, 0.1, 0.03][i % 4];
        for k in 0..d {
            blo.push((c[k] - r).max(lo[k]));
            bhi.push((c[k] + r).min(hi[k]));
        }
    }
    let objs = vec![vec![1.0, -0.5]];
    let ch = Chunk::new(24, d, &blo, &bhi);
    let base = bound_chunk(&prog, &ch, &objs, false, None);
    let top = base
        .iter()
        .map(|db| db.lbs[0])
        .fold(f64::NEG_INFINITY, f64::max);
    let thr = top + 0.5;
    (prog, blo, bhi, 24, objs, thr)
}

/// Alpha bounds are the MAXIMUM of rigorous bounds: never below the
/// unrefined bound, never above a sampled value; every plane (the alpha
/// pass's included) holds at sampled points of its box; the band engaged.
#[test]
fn alpha_bounds_never_exceed_a_sampled_value_and_never_fall_below_the_base() {
    let d = 3;
    let mut ran_total = 0usize;
    let mut improved_total = 0u32;
    for seed in [1u64, 2, 3, 4] {
        let (prog, blo, bhi, m, objs, thr) = alpha_fixture(seed);
        let ch = Chunk::new(m, d, &blo, &bhi);
        let base = bound_chunk(&prog, &ch, &objs, true, None);
        let spec = AlphaSpec {
            iters: 20,
            thrs: std::slice::from_ref(&thr),
        };
        let armed = bound_chunk(&prog, &ch, &objs, true, Some(&spec));
        assert_eq!(armed.len(), m);
        let mut rng = Lcg(seed + 100);
        for b in 0..m {
            let (l, h) = (&blo[b * d..(b + 1) * d], &bhi[b * d..(b + 1) * d]);
            let (lb0, lb) = (base[b].lbs[0], armed[b].lbs[0]);
            assert!(
                lb >= lb0,
                "seed {seed} box {b}: alpha lowered the bound {lb0} -> {lb}"
            );
            ran_total += usize::from(armed[b].alpha.ran);
            improved_total += armed[b].alpha.improved;
            let mut min_seen = f64::INFINITY;
            let mut pts: Vec<Vec<f64>> = (0..300).map(|_| sample(&mut rng, l, h)).collect();
            for corner in 0..(1 << d) {
                pts.push(
                    (0..d)
                        .map(|k| if corner >> k & 1 == 1 { h[k] } else { l[k] })
                        .collect(),
                );
            }
            for x in &pts {
                let v = point_objective(&prog, x, &objs[0]);
                min_seen = min_seen.min(v);
                for (i, a, c) in &armed[b].planes {
                    let pv: f64 = a.iter().zip(x).map(|(ak, xk)| ak * xk).sum::<f64>() + c;
                    let yv = point_objective(&prog, x, &objs[*i]);
                    assert!(
                        pv <= yv + 1e-9 * (1.0 + yv.abs()),
                        "SOUNDNESS: seed {seed} box {b}: plane value {pv} above {yv}"
                    );
                }
            }
            assert!(
                lb <= min_seen + 1e-9,
                "SOUNDNESS: seed {seed} box {b}: alpha bound {lb} exceeds attained {min_seen}"
            );
        }
    }
    assert!(ran_total > 0, "VOID: no domain was in the alpha band");
    eprintln!("[lowdim-bundle-test] alpha ran={ran_total} improved={improved_total}");
}

/// Zero iterations is the historical pass, bit for bit, and so is no spec.
#[test]
fn zero_alpha_iterations_is_the_historical_pass() {
    assert_eq!(LowDimConfig::default().alpha_iters, 0);
    let d = 3;
    let (prog, blo, bhi, m, objs, thr) = alpha_fixture(9);
    let ch = Chunk::new(m, d, &blo, &bhi);
    let plain = bound_chunk(&prog, &ch, &objs, true, None);
    let spec = AlphaSpec {
        iters: 0,
        thrs: std::slice::from_ref(&thr),
    };
    let zero = bound_chunk(&prog, &ch, &objs, true, Some(&spec));
    for (p, z) in plain.iter().zip(&zero) {
        assert_eq!(
            p.lbs.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            z.lbs.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(
            p.grads.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            z.grads.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(p.planes, z.planes);
        assert_eq!(z.alpha, search::AlphaNote::default());
    }
}

/// End to end with alpha armed: a sampled-violated threshold never verifies;
/// a true margin does.
#[test]
fn the_lane_with_alpha_never_verifies_a_violated_threshold_and_proves_a_true_margin() {
    let d = 3;
    let g = relu_graph(3, d, 12);
    let (lo, hi) = root_box(d);
    let input = boxed(&lo, &hi);
    let prog = lower_graph(&g, &input).expect("lowers");
    let o = [1.0f64, 0.0];
    let mut rng = Lcg(1234);
    let mut min_seen = f64::INFINITY;
    for _ in 0..20_000 {
        min_seen = min_seen.min(point_objective(&prog, &sample(&mut rng, &lo, &hi), &o));
    }
    let deadline = Instant::now() + Duration::from_secs(60);
    let cfg = LowDimConfig {
        alpha_iters: 20,
        ..plain_cfg()
    };
    let false_thr = (min_seen + 0.05) as f32;
    match prove_objective_lower_bound(&g, &input, &[1.0, 0.0], false_thr, deadline, &cfg) {
        LowDimOutcome::Verified(s) => {
            panic!("SOUNDNESS: verified lower > {false_thr} but a point attains {min_seen} ({s:?})")
        }
        LowDimOutcome::Unproved(..) => {}
        LowDimOutcome::Declined(r) => panic!("declined: {r}"),
    }
    let true_thr = (min_seen - 0.25) as f32;
    match prove_objective_lower_bound(&g, &input, &[1.0, 0.0], true_thr, deadline, &cfg) {
        LowDimOutcome::Verified(s) => assert!(s.closed >= 1),
        other => panic!("expected a proof of lower > {true_thr} (min seen {min_seen}): {other:?}"),
    }
}

// --------------------------------------------------------- (3) leaf attack

/// A clock start old enough that the attack's wall share is not the limit.
fn long_ago() -> Instant {
    Instant::now()
        .checked_sub(Duration::from_secs(120))
        .expect("the monotonic clock is older than two minutes")
}

/// A hit is inside its leaf box and violates on the lowering by the lane's
/// candidate margin; a true margin never yields one; counters record the
/// round.
#[test]
fn the_leaf_attack_finds_an_in_box_violation_and_never_a_false_one() {
    let d = 3;
    let g = relu_graph(3, d, 12);
    let (lo, hi) = root_box(d);
    let prog = lower_graph(&g, &boxed(&lo, &hi)).expect("lowers");
    let o = vec![1.0f64, 0.0];
    let mut rng = Lcg(4321);
    let mut min_seen = f64::INFINITY;
    for _ in 0..20_000 {
        min_seen = min_seen.min(point_objective(&prog, &sample(&mut rng, &lo, &hi), &o));
    }
    let cfg = plain_cfg();
    let deadline = Instant::now() + Duration::from_secs(60);

    // Violated threshold: the attack must find a point the centre test uses.
    let thr = min_seen + 0.05;
    let mut stats = LowDimStats::default();
    let hit = attack::round(
        &prog,
        std::slice::from_ref(&o),
        &[thr],
        &cfg,
        vec![(-1.0, lo.clone(), hi.clone())],
        &mut stats,
        long_ago(),
        deadline,
    )
    .expect("the attack finds a sampled-reachable violation");
    assert!(hit.len() == d && (0..d).all(|k| lo[k] <= hit[k] && hit[k] <= hi[k]));
    assert!(point_objective(&prog, &hit, &o) <= thr - candidate_margin(thr));
    assert_eq!((stats.attack_rounds, stats.attack_hits), (1, 1));
    assert!(stats.attack_evals > 0);

    // True margin: no candidate can exist, so none may be returned.
    let thr = min_seen - 0.25;
    let mut stats = LowDimStats::default();
    let none = attack::round(
        &prog,
        std::slice::from_ref(&o),
        &[thr],
        &cfg,
        vec![(-1.0, lo.clone(), hi.clone())],
        &mut stats,
        long_ago(),
        deadline,
    );
    assert!(
        none.is_none(),
        "SOUNDNESS: a candidate below a true margin: {none:?}"
    );
    assert_eq!(stats.attack_hits, 0);

    // No wall share: a due round is skipped and counted, nothing runs.
    let mut stats = LowDimStats::default();
    let skipped = attack::round(
        &prog,
        std::slice::from_ref(&o),
        &[min_seen + 0.05],
        &cfg,
        vec![(-1.0, lo.clone(), hi.clone())],
        &mut stats,
        // A lane that "started" in the future has spent no wall: no share.
        Instant::now() + Duration::from_secs(60),
        deadline,
    );
    assert!(skipped.is_none());
    assert_eq!((stats.attack_rounds, stats.attack_skipped), (0, 1));
}

/// End to end with the attack armed on every wave: a true margin still
/// verifies (the attack never interferes with a proof); a violated threshold
/// ends as a Candidate whose point violates on the lowering.
#[test]
fn the_lane_with_the_leaf_attack_proves_true_margins_and_hands_back_violations() {
    assert_eq!(LowDimConfig::default().leaf_attack_every, 0);
    let d = 3;
    let g = relu_graph(8, d, 12);
    let (lo, hi) = root_box(d);
    let input = boxed(&lo, &hi);
    let prog = lower_graph(&g, &input).expect("lowers");
    let o2 = [1.0f64, 0.0];
    let mut rng = Lcg(99);
    let mut min_seen = f64::INFINITY;
    for _ in 0..20_000 {
        min_seen = min_seen.min(point_objective(&prog, &sample(&mut rng, &lo, &hi), &o2));
    }
    let deadline = Instant::now() + Duration::from_secs(60);
    let cfg = LowDimConfig {
        leaf_attack_every: 1,
        ..plain_cfg()
    };
    match prove_objective_lower_bound(
        &g,
        &input,
        &[1.0, 0.0],
        (min_seen - 0.25) as f32,
        deadline,
        &cfg,
    ) {
        LowDimOutcome::Verified(_) => {}
        other => panic!("expected a proof with the attack armed: {other:?}"),
    }
    let t = (min_seen + 0.05) as f32;
    let mut w = None;
    match prove_first_output_witnessed(&g, &input, t, false, deadline, &cfg, &mut w) {
        LowDimOutcome::Verified(s) => panic!("SOUNDNESS: verified a violated threshold ({s:?})"),
        LowDimOutcome::Unproved(_, LowDimStop::Candidate) => {
            let x = w.expect("a Candidate stop hands back its point");
            assert!(x.len() == d && (0..d).all(|k| lo[k] <= x[k] && x[k] <= hi[k]));
            assert!(point_objective(&prog, &x, &o2) <= f64::from(t));
        }
        other => panic!("expected a Candidate stop: {other:?}"),
    }
}

/// Every bundle key is dark in the default lane config.
#[test]
fn every_bundle_key_is_dark_by_default() {
    let cfg = LowDimConfig::default();
    assert!(!cfg.varying_dims);
    assert_eq!(cfg.alpha_iters, 0);
    assert_eq!(cfg.leaf_attack_every, 0);
    assert!(!cfg.shared_worklist);
    assert_eq!(LowerOpts::of(&cfg), LowerOpts::default());
}

// ------------------------------------------- #m8-cgan-leaf-batch: dist_shift

/// A dist_shift-shaped chain (`mnist_concat` in miniature): `Linear(d -> 10)
/// -> ReLU -> Linear(10 -> 16) -> Sigmoid -> Reshape [1, 4, 4] -> Reshape [16]
/// -> Linear(16 -> 8) -> ReLU -> Linear(8 -> 3)`. The first reshape's output
/// is an IMAGE, not a vector (the W10B decline: "node `layer_4` produces
/// [1, 28, 28], not a width-784 vector").
fn image_chain_graph(seed: u64, d: usize) -> GraphNetwork {
    let mut rng = Lcg(seed);
    let mut g = GraphNetwork::new();
    let chain: Vec<(&str, Layer)> = vec![
        ("l1", linear(&mut rng, 10, d, 0.8)),
        ("r1", Layer::ReLU(ReLULayer::new())),
        ("l2", linear(&mut rng, 16, 10, 1.0)),
        ("s", Layer::Sigmoid(SigmoidLayer::new())),
        ("img", Layer::Reshape(ReshapeLayer::new(vec![1, 4, 4]))),
        ("vec", Layer::Reshape(ReshapeLayer::new(vec![16]))),
        ("l3", linear(&mut rng, 8, 16, 1.2)),
        ("r3", Layer::ReLU(ReLULayer::new())),
        ("out", linear(&mut rng, 3, 8, 1.5)),
    ];
    let mut prev = NETWORK_INPUT.to_string();
    for (name, layer) in chain {
        g.add_node(GraphNode::new(name, layer, vec![prev.clone()]));
        prev = name.to_string();
    }
    g.set_output("out");
    g
}

const DIST_D: usize = 20;
/// The free (latent) coordinates sit at the END, as in dist_shift.
const DIST_FREE: [usize; 4] = [16, 17, 18, 19];
/// The one EXACT pin (every other pinned coordinate is loader-widened).
const DIST_EXACT: usize = 5;

/// The VNN-LIB loader's widening of a declared point (`split_input_bounds_f32`
/// in ny-onnx, which this crate cannot see; `dist_shift_entry_box_widening_is_ulp_scale`
/// in the CLI pins the real one): one f32 ulp outward on each side, then a
/// subnormal endpoint snapped outward to `0` / `+-f32::MIN_POSITIVE`.
fn loader_widen(v: f32) -> (f64, f64) {
    let snap_lo = |l: f32| {
        if l.is_subnormal() {
            if l < 0.0 {
                -f32::MIN_POSITIVE
            } else {
                0.0
            }
        } else {
            l
        }
    };
    let snap_hi = |h: f32| {
        if h.is_subnormal() {
            if h > 0.0 {
                f32::MIN_POSITIVE
            } else {
                0.0
            }
        } else {
            h
        }
    };
    (
        f64::from(snap_lo(ny_tensor::next_down_f32(v))),
        f64::from(snap_hi(ny_tensor::next_up_f32(v))),
    )
}

/// The dist_shift-shaped root box: 16 declared "pixels" (zeros, ones, a
/// binade edge and ordinary values) as the loader delivers them (one of them
/// exact, the rest widened), then 4 free latent intervals.
fn dist_box() -> (Vec<f64>, Vec<f64>) {
    let mut rng = Lcg(71);
    let (mut lo, mut hi) = (vec![0.0f64; DIST_D], vec![0.0f64; DIST_D]);
    for i in 0..16 {
        let v: f32 = match i % 4 {
            0 => 0.0,
            1 => 1.0,
            2 => 0.5,
            // Pixels up to 8 make each pin's one-ulp width (up to ~1e-6)
            // visible at the output without saturating the sigmoid, so a
            // point fold cannot hide.
            _ => ((rng.next() + 1.0) * 4.0) as f32,
        };
        let (l, h) = if i == DIST_EXACT {
            (f64::from(v), f64::from(v))
        } else {
            loader_widen(v)
        };
        lo[i] = l;
        hi[i] = h;
    }
    for (k, &i) in DIST_FREE.iter().enumerate() {
        lo[i] = f64::from(-1.25f32 - 0.25 * k as f32);
        hi[i] = f64::from(1.0f32 + 0.125 * k as f32);
    }
    (lo, hi)
}

/// A full-dimension point of the dist_shift box: the free coordinates from
/// `x_free`, every pinned coordinate at a random point of ITS interval (its
/// endpoints half the time), so a fold that treated a narrow pin as a point
/// would be caught.
fn dist_point(rng: &mut Lcg, lo: &[f64], hi: &[f64], x_free: &[f64]) -> Vec<f64> {
    let mut x = vec![0.0f64; lo.len()];
    for i in 0..lo.len() {
        let u = rng.next();
        x[i] = if u < -0.5 {
            lo[i]
        } else if u < 0.0 {
            hi[i]
        } else {
            lo[i] + (hi[i] - lo[i]) * u
        };
    }
    for (k, &i) in DIST_FREE.iter().enumerate() {
        x[i] = x_free[k];
    }
    x
}

fn armed_lowering() -> LowerOpts {
    LowerOpts {
        sigmoid: true,
        flat_reshape: true,
    }
}

/// The graph's own f32 forward at `x` (a point box through IBP), as
/// `(lower, upper)` per output.
fn graph_point(g: &GraphNetwork, x: &[f64]) -> Vec<(f64, f64)> {
    let x32: Vec<f32> = x.iter().map(|&v| v as f32).collect();
    let point = ArrayD::from_shape_vec(IxDyn(&[1, x.len()]), x32).expect("shape");
    let y = g
        .propagate_ibp(&BoundedTensor::new(point.clone(), point).expect("point box"))
        .expect("graph forward");
    y.lower()
        .iter()
        .zip(y.upper().iter())
        .map(|(&l, &h)| (f64::from(l), f64::from(h)))
        .collect()
}

/// The classification: exact, narrow (every shape the loader's widening
/// takes, including a declared zero and a binade edge) or free.
#[test]
fn pinned_coordinates_are_classified_exact_narrow_or_free() {
    assert_eq!(pin_kind(0.25, 0.25), PinKind::Exact);
    assert_eq!(pin_kind(0.0, 0.0), PinKind::Exact);
    let mut rng = Lcg(3);
    let mut values: Vec<f32> = vec![0.0, 1.0, 0.5, 2.0, -1.0, 52.088_51, -76.635_78, 1e-30, 3e38];
    for _ in 0..200 {
        values.push((rng.next() * 100.0) as f32);
    }
    for v in values {
        let (l, h) = loader_widen(v);
        assert!(l <= f64::from(v) && f64::from(v) <= h, "{v}: not outward");
        assert!(l < h, "{v}: the widening vanished");
        assert_eq!(pin_kind(l, h), PinKind::Narrow, "{v}: [{l:e}, {h:e}]");
        // Twice the tolerance is free: the classification is not a blanket
        // "small interval" rule.
        let wide = h + 2.0 * pin_tolerance(l, h);
        assert_eq!(pin_kind(l, wide), PinKind::Free, "{v}: [{l:e}, {wide:e}]");
    }
    // The snapped zero is exactly [-MIN_POSITIVE, MIN_POSITIVE].
    let (l0, h0) = loader_widen(0.0);
    assert_eq!(
        (l0, h0),
        (-f64::from(f32::MIN_POSITIVE), f64::from(f32::MIN_POSITIVE))
    );
    // Real dist_shift latent intervals are free.
    assert_eq!(
        pin_kind(-55.467_694_988_250_734, -52.088_510_761_260_984),
        PinKind::Free
    );
    assert_eq!(pin_kind(-1.0, 1.0), PinKind::Free);
    // Malformed coordinates are never pinned.
    assert_eq!(pin_kind(1.0, 0.0), PinKind::Free);
    assert_eq!(pin_kind(f64::NEG_INFINITY, 0.0), PinKind::Free);
    assert_eq!(pin_kind(f64::NAN, 0.0), PinKind::Free);
    let (lo, hi) = dist_box();
    let counts = pin_counts(&lo, &hi);
    assert_eq!(
        counts,
        PinCounts {
            exact: 1,
            narrow: 15,
            free: DIST_FREE.len()
        }
    );
}

/// SOUNDNESS: the dist_shift-shaped box is admitted over its 4 free
/// coordinates with every loader-widened pin folded as its INTERVAL: bounds
/// over sub-boxes (points included) of the free coordinates never exceed the
/// FULL program's value at any point of the pinned intervals. The point
/// sub-boxes are the bite: there the relaxation slack vanishes, so a fold
/// that used one point of a narrow interval would be exceeded by the
/// sampled endpoints (their spread is asserted non-zero, else VOID).
#[test]
fn a_narrow_pinned_box_folds_as_intervals_and_contains_sampled_values() {
    let g = image_chain_graph(11, DIST_D);
    let (lo, hi) = dist_box();
    let input = boxed(&lo, &hi);
    let cfg = armed_varying();
    // Unarmed the box declines as before (20 dims, nothing classified).
    assert!(admit_with(&g, &input, &plain_cfg()).is_err());
    let adm = admit_with(&g, &input, &cfg).expect("admitted armed");
    assert_eq!(adm.input_dim(), DIST_FREE.len());
    let full = lower_graph_opts(&g, &input, armed_lowering()).expect("full lowering");
    let fold = pinned::PinnedFold::of(&lo, &hi)
        .expect("admissible")
        .expect("folds");
    assert_eq!(fold.free_dim(), DIST_FREE.len());
    assert_eq!(fold.counts().narrow, 15);
    let (flo, fhi) = (fold.restrict(&lo), fold.restrict(&hi));
    let d = flo.len();
    let mut rng = Lcg(29);
    let radii = [0.5, 0.01, 1e-4, 0.0];
    let (mut blo, mut bhi) = (Vec::new(), Vec::new());
    for i in 0..16 {
        let c = sample(&mut rng, &flo, &fhi);
        let r = radii[i % radii.len()];
        for k in 0..d {
            blo.push((c[k] - r).max(flo[k]));
            bhi.push((c[k] + r).min(fhi[k]));
        }
    }
    let objectives = vec![vec![1.0, 0.0, 0.0], vec![-0.5, 1.0, 0.25]];
    let lbs = bound_boxes(&adm, &blo, &bhi, &objectives, &cfg).expect("bounds");
    assert_eq!(lbs.len(), 16);
    let mut point_spread = 0.0f64;
    for (b, row) in lbs.iter().enumerate() {
        let (l, h) = (&blo[b * d..(b + 1) * d], &bhi[b * d..(b + 1) * d]);
        for (o, lb) in objectives.iter().zip(row) {
            assert!(lb.is_finite(), "box {b}: non-finite bound");
            let (mut min_seen, mut max_seen) = (f64::INFINITY, f64::NEG_INFINITY);
            for _ in 0..300 {
                let xf = sample(&mut rng, l, h);
                let x = dist_point(&mut rng, &lo, &hi, &xf);
                let v = point_objective(&full, &x, o);
                min_seen = min_seen.min(v);
                max_seen = max_seen.max(v);
            }
            assert!(
                *lb <= min_seen + 1e-9,
                "SOUNDNESS: box {b}: folded bound {lb:e} exceeds the full program's {min_seen:e}"
            );
            if radii[b % radii.len()] == 0.0 {
                point_spread = point_spread.max(max_seen - min_seen);
            }
        }
    }
    assert!(
        point_spread > 1e-7,
        "the narrow pins never moved the output at a free point: the test cannot see a point fold (VOID)"
    );
}

/// Armed, an equal-count reshape to an image lowers as a flat relabel and
/// the lowering is the graph's own function; unarmed (flat relabel off) it
/// declines exactly as W10B's dist_shift row did.
#[test]
fn a_flat_reshape_to_an_image_lowers_armed_and_declines_unarmed() {
    let d = 6;
    let g = image_chain_graph(13, d);
    let (lo, hi) = root_box(d);
    let input = boxed(&lo, &hi);
    let unarmed = lower_graph_opts(
        &g,
        &input,
        LowerOpts {
            sigmoid: true,
            flat_reshape: false,
        },
    );
    match unarmed {
        Err(r) => assert!(
            r.contains("produces [1, 4, 4], not a width-16 vector"),
            "unexpected decline: {r}"
        ),
        Ok(_) => panic!("an image reshape lowered with flat relabels off"),
    }
    assert!(lower_graph_opts(&g, &input, LowerOpts::default()).is_err());
    let prog = lower_graph_opts(&g, &input, armed_lowering()).expect("armed lowering");
    let mut rng = Lcg(19);
    for _ in 0..40 {
        let x = sample(&mut rng, &lo, &hi);
        // The graph forward runs in f32: evaluate the lowering at the same
        // f32-rounded point.
        let x: Vec<f64> = x.iter().map(|&v| f64::from(v as f32)).collect();
        let mine = point_values(&prog, &x);
        let theirs = graph_point(&g, &x);
        assert_eq!(mine.len(), theirs.len());
        for (j, (v, (l, h))) in mine.iter().zip(&theirs).enumerate() {
            let tol = 1e-4 * (1.0 + v.abs());
            assert!(
                *l - tol <= *v && *v <= *h + tol,
                "output {j}: lowering {v:e} outside the graph's [{l:e}, {h:e}]"
            );
        }
    }
    // End to end over the (within-cap) box: a sampled-violated threshold
    // never verifies, a true margin does.
    let cfg = armed_varying();
    let e0 = [1.0f64, 0.0, 0.0];
    let (mut min_seen, mut max_seen) = (f64::INFINITY, f64::NEG_INFINITY);
    for _ in 0..20_000 {
        let v = point_objective(&prog, &sample(&mut rng, &lo, &hi), &e0);
        min_seen = min_seen.min(v);
        max_seen = max_seen.max(v);
    }
    let spread = (max_seen - min_seen).max(1e-3);
    let deadline = Instant::now() + Duration::from_secs(60);
    let t = (min_seen + 0.05 * spread) as f32;
    if let LowDimOutcome::Verified(s) =
        prove_first_output_witnessed(&g, &input, t, false, deadline, &cfg, &mut None)
    {
        panic!("SOUNDNESS: verified y0 > {t} but a point attains {min_seen} ({s:?})");
    }
    let t = (min_seen - 0.2 * spread) as f32;
    match prove_first_output_witnessed(&g, &input, t, false, deadline, &cfg, &mut None) {
        LowDimOutcome::Verified(_) => {}
        other => panic!("expected a proof of y0 > {t} (min seen {min_seen}): {other:?}"),
    }
}

/// Two branches reshaped to `[2, 4]` and `[1 or 2, ...]`, concatenated, then
/// flattened: along axis 0 the concat is a flat concatenation (lowers armed,
/// agrees with the graph), along axis 1 it interleaves (declines armed).
fn concat_graph(seed: u64, d: usize, outer: bool) -> GraphNetwork {
    let mut rng = Lcg(seed);
    let mut g = GraphNetwork::new();
    let inp = NETWORK_INPUT.to_string();
    g.add_node(GraphNode::new(
        "la",
        linear(&mut rng, 8, d, 1.0),
        vec![inp.clone()],
    ));
    g.add_node(GraphNode::new(
        "ra",
        Layer::Reshape(ReshapeLayer::new(vec![2, 4])),
        vec!["la".into()],
    ));
    g.add_node(GraphNode::new("lb", linear(&mut rng, 4, d, 1.0), vec![inp]));
    let (rb_shape, axis, out_n): (Vec<i64>, i64, usize) = if outer {
        (vec![1, 4], 0, 12)
    } else {
        (vec![2, 2], 1, 12)
    };
    g.add_node(GraphNode::new(
        "rb",
        Layer::Reshape(ReshapeLayer::new(rb_shape)),
        vec!["lb".into()],
    ));
    g.add_node(GraphNode::new(
        "cat",
        Layer::Concat(ConcatLayer::new(axis)),
        vec!["ra".into(), "rb".into()],
    ));
    g.add_node(GraphNode::new(
        "flat",
        Layer::Reshape(ReshapeLayer::new(vec![out_n as i64])),
        vec!["cat".into()],
    ));
    g.add_node(GraphNode::new(
        "out",
        linear(&mut rng, 2, out_n, 0.7),
        vec!["flat".into()],
    ));
    g.set_output("out");
    g
}

#[test]
fn an_outer_axis_concat_lowers_flat_and_an_inner_axis_concat_declines() {
    let d = 6;
    let (lo, hi) = root_box(d);
    let input = boxed(&lo, &hi);
    let outer = concat_graph(31, d, true);
    let off = lower_graph_opts(
        &outer,
        &input,
        LowerOpts {
            sigmoid: true,
            flat_reshape: false,
        },
    );
    assert!(
        off.is_err(),
        "a [2, 4] reshape lowered with flat relabels off"
    );
    let prog = lower_graph_opts(&outer, &input, armed_lowering()).expect("outer concat lowers");
    assert!(
        prog.ops.iter().any(|op| matches!(op, Op::Concat { .. })),
        "the lowering has no Concat: VOID"
    );
    let mut rng = Lcg(37);
    for _ in 0..40 {
        let x: Vec<f64> = sample(&mut rng, &lo, &hi)
            .iter()
            .map(|&v| f64::from(v as f32))
            .collect();
        let mine = point_values(&prog, &x);
        let theirs = graph_point(&outer, &x);
        assert_eq!(mine.len(), theirs.len());
        for (j, (v, (l, h))) in mine.iter().zip(&theirs).enumerate() {
            let tol = 1e-4 * (1.0 + v.abs());
            assert!(
                *l - tol <= *v && *v <= *h + tol,
                "output {j}: lowering {v:e} outside the graph's [{l:e}, {h:e}]"
            );
        }
    }
    let inner = concat_graph(31, d, false);
    match lower_graph_opts(&inner, &input, armed_lowering()) {
        Err(r) => assert!(r.contains("interleaves"), "unexpected decline: {r}"),
        Ok(_) => panic!("an inner-axis concat lowered as a flat concatenation"),
    }
}

/// End to end on the dist_shift-shaped widened box, armed: a
/// sampled-violated threshold (sampled over the pinned intervals too) never
/// verifies, and a Candidate comes back full-dimension INSIDE the widened
/// box, exact pins at their value and narrow pins at their midpoint; a true
/// margin verifies.
#[test]
fn the_dist_shift_shaped_lane_never_verifies_a_violation_and_proves_a_true_margin() {
    let g = image_chain_graph(17, DIST_D);
    let (lo, hi) = dist_box();
    let input = boxed(&lo, &hi);
    let full = lower_graph_opts(&g, &input, armed_lowering()).expect("full lowering");
    let fold = pinned::PinnedFold::of(&lo, &hi)
        .expect("admissible")
        .expect("folds");
    let (flo, fhi) = (fold.restrict(&lo), fold.restrict(&hi));
    let e0 = [1.0f64, 0.0, 0.0];
    let mut rng = Lcg(41);
    let (mut min_seen, mut max_seen) = (f64::INFINITY, f64::NEG_INFINITY);
    for _ in 0..20_000 {
        let xf = sample(&mut rng, &flo, &fhi);
        let v = point_objective(&full, &dist_point(&mut rng, &lo, &hi, &xf), &e0);
        min_seen = min_seen.min(v);
        max_seen = max_seen.max(v);
    }
    let spread = (max_seen - min_seen).max(1e-3);
    let deadline = Instant::now() + Duration::from_secs(60);
    let cfg = armed_varying();
    let t = (min_seen + 0.05 * spread) as f32;
    let mut w = None;
    match prove_first_output_witnessed(&g, &input, t, false, deadline, &cfg, &mut w) {
        LowDimOutcome::Verified(s) => {
            panic!("SOUNDNESS: verified y0 > {t} but a point attains {min_seen} ({s:?})")
        }
        LowDimOutcome::Unproved(_, LowDimStop::Candidate) => {
            let x = w.expect("a Candidate stop must hand back its point");
            assert_eq!(x.len(), DIST_D, "the candidate must be full-dimension");
            for i in 0..DIST_D {
                assert!(
                    lo[i] <= x[i] && x[i] <= hi[i],
                    "coordinate {i} outside the box"
                );
                if !DIST_FREE.contains(&i) {
                    assert_eq!(
                        x[i],
                        lo[i] + 0.5 * (hi[i] - lo[i]),
                        "pinned coordinate {i} not at its midpoint"
                    );
                }
            }
            assert_eq!(x[DIST_EXACT], lo[DIST_EXACT]);
            assert!(point_objective(&full, &x, &e0) <= f64::from(t) + 1e-6);
        }
        LowDimOutcome::Unproved(..) => {}
        LowDimOutcome::Declined(r) => panic!("declined: {r}"),
    }
    let t = (min_seen - 0.2 * spread) as f32;
    match prove_first_output_witnessed(&g, &input, t, false, deadline, &cfg, &mut None) {
        LowDimOutcome::Verified(_) => {}
        other => panic!("expected a proof of y0 > {t} (min seen {min_seen}): {other:?}"),
    }
}

// ------------------------------------ review(m8b): narrow pins through the embed

/// review(m8b): a dist_shift-shaped graph that reads its PINNED coordinates
/// ONLY through non-Linear readers (`Slice` -> `Concat` -> the output
/// `Linear`), so every narrow pin reaches the objective through the fold's
/// EMBED row (an `Op::Sparse` constant `[lo_j, hi_j]`), never through the
/// input-Linear fold that `image_chain_graph` exercises. The free (latent)
/// coordinates are sliced into their own nonlinear branch.
fn embed_only_graph(seed: u64) -> GraphNetwork {
    let mut rng = Lcg(seed);
    let mut g = GraphNetwork::new();
    let inp = NETWORK_INPUT.to_string();
    g.add_node(GraphNode::new(
        "zf",
        Layer::Slice(SliceLayer::new(1, 16, 20)),
        vec![inp.clone()],
    ));
    g.add_node(GraphNode::new(
        "l1",
        linear(&mut rng, 6, 4, 1.0),
        vec!["zf".into()],
    ));
    g.add_node(GraphNode::new(
        "r1",
        Layer::ReLU(ReLULayer::new()),
        vec!["l1".into()],
    ));
    g.add_node(GraphNode::new(
        "s",
        Layer::Sigmoid(SigmoidLayer::new()),
        vec!["r1".into()],
    ));
    g.add_node(GraphNode::new(
        "px",
        Layer::Slice(SliceLayer::new(1, 0, 16)),
        vec![inp],
    ));
    g.add_node(GraphNode::new(
        "cat",
        Layer::Concat(ConcatLayer::new(1)),
        vec!["s".into(), "px".into()],
    ));
    g.add_node(GraphNode::new(
        "out",
        linear(&mut rng, 3, 22, 1.2),
        vec!["cat".into()],
    ));
    g.set_output("out");
    g
}

/// review(m8b) SOUNDNESS of the EMBED path under NARROW pins. The pins enter
/// the objective linearly (`Slice -> Concat -> Linear`), so the full
/// program's exact minimum over the pinned intervals at a free point is
/// attained with each pin at the endpoint its own coefficient prefers; the
/// test finds that corner coordinate-wise (both endpoints evaluated) and
/// asserts the folded bound over the POINT free box never exceeds it. A fold
/// whose embed carried a narrow pin as a point would exceed it by
/// `sum_j |c_j| (hi_j - lo_j) / 2`, which is asserted visible (else VOID).
/// The builder's narrow-pin tests only reach pins through the input-Linear
/// fold, so this is the embed's only narrow-pin pin.
#[test]
fn review_narrow_pins_read_through_the_embed_fold_as_intervals() {
    let g = embed_only_graph(23);
    let (lo, hi) = dist_box();
    let input = boxed(&lo, &hi);
    let cfg = armed_varying();
    let adm = admit_with(&g, &input, &cfg).expect("admitted armed");
    assert_eq!(adm.input_dim(), DIST_FREE.len());
    let full = lower_graph_opts(&g, &input, armed_lowering()).expect("full lowering");
    let fold = pinned::PinnedFold::of(&lo, &hi)
        .expect("admissible")
        .expect("folds");
    assert_eq!(fold.counts().narrow, 15);
    // No input Linear exists, so the pins can only be carried by the embed.
    assert!(
        !full
            .ops
            .iter()
            .any(|op| matches!(op, Op::Linear { src: 0, .. })),
        "a Linear reads the input directly: the fold path, not the embed (VOID)"
    );
    let (flo, fhi) = (fold.restrict(&lo), fold.restrict(&hi));
    let mut rng = Lcg(43);
    let objectives = vec![vec![1.0, 0.0, 0.0], vec![-0.5, 1.0, 0.25]];
    let (mut blo, mut bhi, mut centres) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..12 {
        let c = sample(&mut rng, &flo, &fhi);
        blo.extend_from_slice(&c);
        bhi.extend_from_slice(&c);
        centres.push(c);
    }
    let lbs = bound_boxes(&adm, &blo, &bhi, &objectives, &cfg).expect("bounds");
    assert_eq!(lbs.len(), centres.len());
    let mut min_excess = f64::INFINITY;
    for (b, c) in centres.iter().enumerate() {
        for (o, lb) in objectives.iter().zip(&lbs[b]) {
            assert!(lb.is_finite(), "box {b}: non-finite bound");
            let mut x = fold.expand(c);
            let at_mid = point_objective(&full, &x, o);
            for j in 0..DIST_D {
                if DIST_FREE.contains(&j) {
                    continue;
                }
                x[j] = lo[j];
                let at_lo = point_objective(&full, &x, o);
                x[j] = hi[j];
                let at_hi = point_objective(&full, &x, o);
                x[j] = if at_lo <= at_hi { lo[j] } else { hi[j] };
            }
            let worst = point_objective(&full, &x, o);
            assert!(
                *lb <= worst + 1e-12 * (1.0 + worst.abs()),
                "SOUNDNESS: box {b}: folded bound {lb:e} exceeds the full program's minimum \
                 {worst:e} over the pinned intervals"
            );
            min_excess = min_excess.min(at_mid - worst);
        }
    }
    assert!(
        min_excess > 1e-8,
        "the pinned intervals are too narrow to expose a point fold ({min_excess:e}): VOID"
    );
}
