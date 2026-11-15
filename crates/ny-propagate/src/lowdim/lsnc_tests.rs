// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #lsnc-lane soundness tests for the Lyapunov-certificate lowerings:
//! `MulBinary` (McCormick), `ReduceSum`, `Gather`, `DivConstant` and both
//! `IntervalConst` shapes, plus the lane-twin binding.
//!
//! The load-bearing assertion everywhere is CONTAINMENT: a published lower
//! bound must never exceed a value the network attains. For an interval
//! constant "the network" is EVERY member of the enclosed set, so the
//! reference below evaluates the function at extreme members too — a lowering
//! that dropped the radius passes a centre-only check and fails this one.
//! The reference is an independent f64 evaluation written out by hand, not the
//! lowered program.

use std::time::{Duration, Instant};

use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;

use super::*;
use crate::layers::{
    ConcatLayer, DivConstantLayer, GatherLayer, IntervalConstLayer, LinearLayer, MulBinaryLayer,
    OpaqueSkipLayer, ReLULayer, ReduceSumLayer,
};
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
    fn f32(&mut self, scale: f64) -> f32 {
        (self.next() * scale) as f32
    }
}

fn boxed(lo: &[f64], hi: &[f64]) -> BoundedTensor {
    let d = lo.len();
    let l = ArrayD::from_shape_vec(IxDyn(&[1, d]), lo.iter().map(|&v| v as f32).collect()).unwrap();
    let h = ArrayD::from_shape_vec(IxDyn(&[1, d]), hi.iter().map(|&v| v as f32).collect()).unwrap();
    BoundedTensor::new(l, h).unwrap()
}

fn sample(rng: &mut Lcg, lo: &[f64], hi: &[f64]) -> Vec<f64> {
    lo.iter()
        .zip(hi)
        .map(|(l, h)| l + (h - l) * 0.5 * (rng.next() + 1.0))
        .collect()
}

/// Sub-boxes of the root box `[lo, hi]` at assorted widths (including point
/// boxes), `[count][d]`.
fn sub_boxes(rng: &mut Lcg, lo: &[f64], hi: &[f64], count: usize) -> (Vec<f64>, Vec<f64>) {
    let d = lo.len();
    let (mut blo, mut bhi) = (Vec::new(), Vec::new());
    for i in 0..count {
        let c = sample(rng, lo, hi);
        let frac = [1.0, 0.3, 0.05, 1e-3, 0.0][i % 5];
        for k in 0..d {
            let r = 0.5 * frac * (hi[k] - lo[k]);
            blo.push((c[k] - r).max(lo[k]));
            bhi.push((c[k] + r).min(hi[k]));
        }
    }
    (blo, bhi)
}

/// The Lyapunov-shaped fragment, with every constant in f64 (exact images of
/// the f32 values the graph stores).
struct Lyap {
    w1: Vec<Vec<f64>>,
    b1: Vec<f64>,
    w2: Vec<Vec<f64>>,
    b2: Vec<f64>,
    off_c: Vec<f64>,
    off_r: Vec<f64>,
    q_c: Vec<Vec<f64>>,
    q_r: Vec<Vec<f64>>,
}

/// One member of the interval-constant set: every scalar at its centre, or
/// (for an extreme member) at a randomly chosen end of its enclosure.
struct Member {
    off: Vec<f64>,
    q: Vec<Vec<f64>>,
}

impl Lyap {
    fn centre(&self) -> Member {
        Member {
            off: self.off_c.clone(),
            q: self.q_c.clone(),
        }
    }

    fn extreme(&self, rng: &mut Lcg) -> Member {
        let pick = |rng: &mut Lcg, c: f64, r: f64| {
            if rng.next() >= 0.0 {
                c + r
            } else {
                c - r
            }
        };
        Member {
            off: self
                .off_c
                .iter()
                .zip(&self.off_r)
                .map(|(&c, &r)| pick(rng, c, r))
                .collect(),
            q: self
                .q_c
                .iter()
                .zip(&self.q_r)
                .map(|(cr, rr)| cr.iter().zip(rr).map(|(&c, &r)| pick(rng, c, r)).collect())
                .collect(),
        }
    }

    /// `[V / 3, V, z / 2]` with `z = relu-net(x) + off`, `cat = [x2, x0, z]`,
    /// `V = cat . (Q cat)`.
    fn eval(&self, x: &[f64], mem: &Member) -> Vec<f64> {
        let h: Vec<f64> = self
            .w1
            .iter()
            .zip(&self.b1)
            .map(|(row, b)| (row.iter().zip(x).map(|(w, v)| w * v).sum::<f64>() + b).max(0.0))
            .collect();
        let u: Vec<f64> = self
            .w2
            .iter()
            .zip(&self.b2)
            .map(|(row, b)| row.iter().zip(&h).map(|(w, v)| w * v).sum::<f64>() + b)
            .collect();
        let z: Vec<f64> = u.iter().zip(&mem.off).map(|(a, b)| a + b).collect();
        let cat = [x[2], x[0], z[0], z[1], z[2]];
        let qc: Vec<f64> = mem
            .q
            .iter()
            .map(|row| row.iter().zip(&cat).map(|(w, v)| w * v).sum::<f64>())
            .collect();
        let v: f64 = cat.iter().zip(&qc).map(|(a, b)| a * b).sum();
        vec![v / 3.0, v, z[0] / 2.0, z[1] / 2.0, z[2] / 2.0]
    }
}

fn mat(rows: &[Vec<f64>]) -> Array2<f32> {
    let (r, c) = (rows.len(), rows[0].len());
    Array2::from_shape_vec((r, c), rows.iter().flatten().map(|&v| v as f32).collect()).unwrap()
}

fn round_f32(v: &mut [Vec<f64>]) {
    for row in v.iter_mut() {
        for x in row.iter_mut() {
            *x = f64::from(*x as f32);
        }
    }
}

/// `x` (3) -> Linear(4) -> ReLU -> Linear(3) -> IntervalOffset -> z;
/// `cat = Concat(Gather(x, [2, 0]), z)`; `V = ReduceSum(cat * IntervalLinear(cat))`;
/// output `Concat(V / 3, V, z / 2)`.
fn lyap(seed: u64, radius: f64) -> (GraphNetwork, Lyap) {
    let mut rng = Lcg(seed);
    let grid = |rng: &mut Lcg, r: usize, c: usize, s: f64| -> Vec<Vec<f64>> {
        (0..r)
            .map(|_| (0..c).map(|_| f64::from(rng.f32(s))).collect())
            .collect()
    };
    let mut spec = Lyap {
        w1: grid(&mut rng, 4, 3, 1.2),
        b1: (0..4).map(|_| f64::from(rng.f32(0.4))).collect(),
        w2: grid(&mut rng, 3, 4, 0.8),
        b2: (0..3).map(|_| f64::from(rng.f32(0.3))).collect(),
        off_c: (0..3).map(|_| f64::from(rng.f32(0.5))).collect(),
        off_r: (0..3)
            .map(|_| f64::from((radius * (0.5 + 0.5 * rng.next().abs())) as f32))
            .collect(),
        q_c: grid(&mut rng, 5, 5, 1.0),
        q_r: (0..5)
            .map(|_| {
                (0..5)
                    .map(|_| f64::from((radius * (0.5 + 0.5 * rng.next().abs())) as f32))
                    .collect()
            })
            .collect(),
    };
    round_f32(&mut spec.q_c);
    round_f32(&mut spec.q_r);
    let inp = NETWORK_INPUT.to_string();
    let mut g = GraphNetwork::new();
    let b1 = Array1::from_vec(spec.b1.iter().map(|&v| v as f32).collect());
    let b2 = Array1::from_vec(spec.b2.iter().map(|&v| v as f32).collect());
    g.add_node(GraphNode::new(
        "l1",
        Layer::Linear(LinearLayer::new(mat(&spec.w1), Some(b1)).unwrap()),
        vec![inp.clone()],
    ));
    g.add_node(GraphNode::new(
        "r1",
        Layer::ReLU(ReLULayer::new()),
        vec!["l1".into()],
    ));
    g.add_node(GraphNode::new(
        "u",
        Layer::Linear(LinearLayer::new(mat(&spec.w2), Some(b2)).unwrap()),
        vec!["r1".into()],
    ));
    let arr = |v: &[f64]| {
        ArrayD::from_shape_vec(IxDyn(&[v.len()]), v.iter().map(|&x| x as f32).collect()).unwrap()
    };
    g.add_node(GraphNode::new(
        "z",
        Layer::IntervalConst(
            IntervalConstLayer::try_new_offset(arr(&spec.off_c[..]), arr(&spec.off_r[..]), 3)
                .unwrap(),
        ),
        vec!["u".into()],
    ));
    let idx = ArrayD::from_shape_vec(IxDyn(&[2]), vec![2i64, 0]).unwrap();
    g.add_node(GraphNode::new(
        "g",
        Layer::Gather(GatherLayer::new(1, Some(idx), vec![])),
        vec![inp],
    ));
    g.add_node(GraphNode::new(
        "cat",
        Layer::Concat(ConcatLayer::new(1)),
        vec!["g".into(), "z".into()],
    ));
    g.add_node(GraphNode::new(
        "q",
        Layer::IntervalConst(
            IntervalConstLayer::try_new_linear(
                LinearLayer::new(mat(&spec.q_c), None).unwrap(),
                mat(&spec.q_r),
            )
            .unwrap(),
        ),
        vec!["cat".into()],
    ));
    g.add_node(GraphNode::new(
        "m",
        Layer::MulBinary(MulBinaryLayer),
        vec!["cat".into(), "q".into()],
    ));
    g.add_node(GraphNode::new(
        "v",
        Layer::ReduceSum(ReduceSumLayer::new(vec![-1], true)),
        vec!["m".into()],
    ));
    g.add_node(GraphNode::new(
        "dv",
        Layer::DivConstant(DivConstantLayer::scalar(3.0)),
        vec!["v".into()],
    ));
    g.add_node(GraphNode::new(
        "hz",
        Layer::DivConstant(DivConstantLayer::scalar(2.0)),
        vec!["z".into()],
    ));
    g.add_node(GraphNode::new(
        "out",
        Layer::Concat(ConcatLayer::new(1)),
        vec!["dv".into(), "v".into(), "hz".into()],
    ));
    g.set_output("out");
    (g, spec)
}

const LO: [f64; 3] = [-0.75, -0.5, -1.0];
const HI: [f64; 3] = [0.5, 1.0, 0.75];

#[test]
fn the_lyapunov_fragment_is_admitted_and_counted() {
    let (g, _) = lyap(3, 1e-3);
    let adm = admit(&g, &boxed(&LO, &HI)).expect("the fragment must lower");
    assert_eq!(adm.lowered_ops(), 12, "every graph node lowers to one op");
    assert_eq!(adm.interval_consts(), 2, "IntervalOffset + IntervalLinear");
    assert_eq!(adm.bilinear_ops(), 1, "one MulBinary");
    assert!(!adm.from_twin());
    assert_eq!((adm.input_dim(), adm.output_width()), (3, 5));
}

#[test]
fn the_lowered_program_matches_the_graph_at_points() {
    let (g, spec) = lyap(5, 1e-3);
    let adm = admit(&g, &boxed(&LO, &HI)).expect("lowers");
    let mut rng = Lcg(41);
    for _ in 0..60 {
        let x: Vec<f64> = sample(&mut rng, &LO, &HI)
            .into_iter()
            .map(|v| f64::from(v as f32))
            .collect();
        let mine = lowered_point_values(&adm, &x).expect("point");
        let reference = spec.eval(&x, &spec.centre());
        let y = g.propagate_ibp(&boxed(&x, &x)).expect("ibp");
        for j in 0..5 {
            let (gl, gh) = (
                f64::from(y.lower()[IxDyn(&[0, j])]),
                f64::from(y.upper()[IxDyn(&[0, j])]),
            );
            let tol = 1e-3 * (1.0 + gl.abs().max(gh.abs()));
            assert!(
                mine[j] >= gl - tol && mine[j] <= gh + tol,
                "lowered y{j}={} outside the graph's point enclosure [{gl}, {gh}]",
                mine[j]
            );
            assert!(
                (mine[j] - reference[j]).abs() <= 1e-9 * (1.0 + reference[j].abs()),
                "lowered y{j}={} vs independent f64 reference {}",
                mine[j],
                reference[j]
            );
        }
    }
}

/// THE soundness test: on boxes of every width, the published lower bound of
/// random objectives never exceeds the value of ANY member of the enclosed
/// constant set at any sampled point (centre member plus extreme members).
#[test]
fn published_bounds_hold_for_every_member_of_the_interval_constants() {
    for (seed, radius) in [(7u64, 1e-3), (8, 2e-2), (9, 0.0)] {
        // radius 0 is the plain-constant control (the constructors refuse an
        // all-zero radius, so it is the smallest positive instead).
        let radius = if radius == 0.0 { 1e-9 } else { radius };
        let (g, spec) = lyap(seed, radius);
        let adm = admit(&g, &boxed(&LO, &HI)).expect("lowers");
        let mut rng = Lcg(seed * 13 + 1);
        let (blo, bhi) = sub_boxes(&mut rng, &LO, &HI, 30);
        let mut objectives: Vec<Vec<f64>> = Vec::new();
        for j in 0..5 {
            let mut e = vec![0.0; 5];
            e[j] = 1.0;
            objectives.push(e.clone());
            e[j] = -1.0;
            objectives.push(e);
        }
        for _ in 0..4 {
            objectives.push((0..5).map(|_| rng.next()).collect());
        }
        let members: Vec<Member> = std::iter::once(spec.centre())
            .chain((0..6).map(|_| spec.extreme(&mut rng)))
            .collect();
        for cfg in [
            LowDimConfig::default(),
            LowDimConfig {
                refine_max_width: 8,
                chunk: 7,
                ..LowDimConfig::default()
            },
            // #lsnc-lane stage 2: the joint quadratic bound at `V`'s site
            // (`ReduceSum(cat * IntervalLinear(cat))`), with and without
            // refinement — its operand certificates and planes must hold for
            // every member of the interval set too.
            LowDimConfig {
                joint_quadratic: true,
                chunk: 5,
                ..LowDimConfig::default()
            },
            LowDimConfig {
                joint_quadratic: true,
                refine_max_width: 8,
                chunk: 7,
                ..LowDimConfig::default()
            },
        ] {
            let lbs = bound_boxes(&adm, &blo, &bhi, &objectives, &cfg).expect("bounds");
            assert_eq!(lbs.len(), 30);
            for (b, lb) in lbs.iter().enumerate() {
                let (l, h) = (&blo[b * 3..(b + 1) * 3], &bhi[b * 3..(b + 1) * 3]);
                let mut points: Vec<Vec<f64>> = (0..120).map(|_| sample(&mut rng, l, h)).collect();
                for corner in 0..8 {
                    points.push(
                        (0..3)
                            .map(|k| if corner >> k & 1 == 1 { h[k] } else { l[k] })
                            .collect(),
                    );
                }
                for (o, bound) in objectives.iter().zip(lb) {
                    assert!(!bound.is_nan(), "seed {seed} box {b}: NaN bound");
                    let mut min_seen = f64::INFINITY;
                    for x in &points {
                        for mem in &members {
                            let y = spec.eval(x, mem);
                            min_seen = min_seen.min(o.iter().zip(&y).map(|(a, v)| a * v).sum());
                        }
                    }
                    assert!(
                        *bound <= min_seen + 1e-12 * (1.0 + min_seen.abs()),
                        "seed {seed} radius {radius} box {b}: bound {bound} exceeds a value {min_seen} \
                         some member of the constant set attains"
                    );
                }
            }
        }
    }
}

/// The radius is CHARGED: on a point box the bound of `V` sits below every
/// member's value, including the minimising extreme member, and strictly below
/// the centre member's value by at least a radius-sized gap. A lowering that
/// carried only the midpoint would publish (a hair under) the centre value.
#[test]
fn the_interval_radius_is_charged_not_dropped() {
    let (g, spec) = lyap(21, 5e-2);
    let adm = admit(&g, &boxed(&LO, &HI)).expect("lowers");
    let mut rng = Lcg(5);
    let mut o = vec![0.0; 5];
    o[1] = 1.0; // V
    for _ in 0..10 {
        let x: Vec<f64> = sample(&mut rng, &LO, &HI);
        let lb = bound_boxes(
            &adm,
            &x,
            &x,
            std::slice::from_ref(&o),
            &LowDimConfig::default(),
        )
        .expect("bounds")[0][0];
        let centre = spec.eval(&x, &spec.centre())[1];
        let mut worst = centre;
        for _ in 0..64 {
            worst = worst.min(spec.eval(&x, &spec.extreme(&mut rng))[1]);
        }
        assert!(
            lb <= worst,
            "point bound {lb} above a member's value {worst}"
        );
        assert!(
            centre - lb > 1e-4,
            "the bound {lb} hugs the centre member's value {centre}: the radius was dropped"
        );
    }
}

/// McCormick on its own: products of operands that straddle zero, on many
/// boxes; both lower planes must actually be chosen somewhere (the selection
/// is exercised, not just plane 0), and every bound stays below the product.
#[test]
fn mccormick_bounds_contain_the_product_and_both_planes_are_used() {
    let mut rng = Lcg(77);
    let wa = mat(&[vec![1.0, -0.5, 0.25], vec![-0.75, 0.5, 1.0]]);
    let wb = mat(&[vec![0.5, 1.0, -1.0], vec![1.25, -0.25, 0.5]]);
    let mut g = GraphNetwork::new();
    g.add_node(GraphNode::new(
        "a",
        Layer::Linear(LinearLayer::new(wa.clone(), None).unwrap()),
        vec![NETWORK_INPUT.into()],
    ));
    g.add_node(GraphNode::new(
        "b",
        Layer::Linear(LinearLayer::new(wb.clone(), None).unwrap()),
        vec![NETWORK_INPUT.into()],
    ));
    g.add_node(GraphNode::new(
        "m",
        Layer::MulBinary(MulBinaryLayer),
        vec!["a".into(), "b".into()],
    ));
    g.set_output("m");
    let (lo, hi) = ([-1.0f64, -1.0, -1.0], [1.0f64, 1.0, 1.0]);
    let prog = lower_graph(&g, &boxed(&lo, &hi)).expect("lowers");
    let eval = |x: &[f64]| -> Vec<f64> {
        (0..2)
            .map(|i| {
                let a: f64 = (0..3).map(|k| f64::from(wa[(i, k)]) * x[k]).sum();
                let b: f64 = (0..3).map(|k| f64::from(wb[(i, k)]) * x[k]).sum();
                a * b
            })
            .collect()
    };
    let (blo, bhi) = sub_boxes(&mut rng, &lo, &hi, 40);
    let ch = Chunk::new(40, 3, &blo, &bhi);
    let one = [&prog];
    let (vals, _) = forward(Fam::single(&one), &ch);
    let mul = vals.last().and_then(Option::as_ref).expect("mul value");
    assert_eq!(mul.pick.len(), 2 * 40, "one pick per neuron and domain");
    let mut picks_seen = [0usize; 4];
    for &p in &mul.pick {
        picks_seen[usize::from(p & 1)] += 1;
        picks_seen[2 + usize::from((p >> 1) & 1)] += 1;
    }
    // Diagnostic: how often each plane (lower 0/1, upper 0/1) was kept.
    eprintln!(
        "mccormick picks lower0={} lower1={} upper0={} upper1={}",
        picks_seen[0], picks_seen[1], picks_seen[2], picks_seen[3]
    );
    assert!(
        picks_seen[1] + picks_seen[3] > 0,
        "the plane selection never left plane 0: it is not exercised ({picks_seen:?})"
    );
    for o in [
        vec![1.0, 0.0],
        vec![-1.0, 0.0],
        vec![0.0, 1.0],
        vec![0.0, -1.0],
        vec![0.7, -0.3],
    ] {
        let res = bound_batch(&prog, &blo, &bhi, 40, &o, 3);
        for (b, (lb, _)) in res.iter().enumerate() {
            let (l, h) = (&blo[b * 3..(b + 1) * 3], &bhi[b * 3..(b + 1) * 3]);
            let mut min_seen = f64::INFINITY;
            for _ in 0..300 {
                let y = eval(sample(&mut rng, l, h).as_slice());
                min_seen = min_seen.min(o[0] * y[0] + o[1] * y[1]);
            }
            assert!(
                *lb <= min_seen + 1e-12,
                "box {b} objective {o:?}: bound {lb} exceeds attained {min_seen}"
            );
        }
    }
}

/// A product that FEEDS a ReLU is refined by a backward pass seeded at the
/// product itself, before its own value is stored: that path must use a valid
/// plane (not panic) and stay sound.
#[test]
fn refinement_seeded_at_a_product_stays_sound() {
    let mut rng = Lcg(9);
    let mut g = GraphNetwork::new();
    let wa = mat(&[vec![1.0, -0.5], vec![0.5, 0.75]]);
    let wb = mat(&[vec![-0.25, 1.0], vec![1.0, 0.5]]);
    let wo = mat(&[vec![1.0, -1.0]]);
    g.add_node(GraphNode::new(
        "a",
        Layer::Linear(LinearLayer::new(wa.clone(), None).unwrap()),
        vec![NETWORK_INPUT.into()],
    ));
    g.add_node(GraphNode::new(
        "b",
        Layer::Linear(LinearLayer::new(wb.clone(), None).unwrap()),
        vec![NETWORK_INPUT.into()],
    ));
    g.add_node(GraphNode::new(
        "m",
        Layer::MulBinary(MulBinaryLayer),
        vec!["a".into(), "b".into()],
    ));
    g.add_node(GraphNode::new(
        "r",
        Layer::ReLU(ReLULayer::new()),
        vec!["m".into()],
    ));
    g.add_node(GraphNode::new(
        "o",
        Layer::Linear(LinearLayer::new(wo, None).unwrap()),
        vec!["r".into()],
    ));
    g.set_output("o");
    let (lo, hi) = ([-1.0f64, -1.0], [1.0f64, 1.0]);
    let adm = admit(&g, &boxed(&lo, &hi)).expect("lowers");
    let eval = |x: &[f64]| -> f64 {
        let m: Vec<f64> = (0..2)
            .map(|i| {
                let a: f64 = (0..2).map(|k| f64::from(wa[(i, k)]) * x[k]).sum();
                let b: f64 = (0..2).map(|k| f64::from(wb[(i, k)]) * x[k]).sum();
                (a * b).max(0.0)
            })
            .collect();
        m[0] - m[1]
    };
    let (blo, bhi) = sub_boxes(&mut rng, &lo, &hi, 25);
    let cfg = LowDimConfig {
        refine_max_width: 4,
        ..LowDimConfig::default()
    };
    for sign in [1.0, -1.0] {
        let lbs = bound_boxes(&adm, &blo, &bhi, &[vec![sign]], &cfg).expect("bounds");
        for (b, lb) in lbs.iter().enumerate() {
            let (l, h) = (&blo[b * 2..(b + 1) * 2], &bhi[b * 2..(b + 1) * 2]);
            let mut min_seen = f64::INFINITY;
            for _ in 0..300 {
                min_seen = min_seen.min(sign * eval(sample(&mut rng, l, h).as_slice()));
            }
            assert!(
                lb[0] <= min_seen + 1e-12,
                "box {b} sign {sign}: bound {} exceeds attained {min_seen}",
                lb[0]
            );
        }
    }
}

#[test]
fn div_constant_is_exact_for_powers_of_two_and_outward_otherwise() {
    assert_eq!(exact_reciprocal(1.0), Some(1.0));
    assert_eq!(exact_reciprocal(2.0), Some(0.5));
    assert_eq!(exact_reciprocal(-4.0), Some(-0.25));
    assert_eq!(exact_reciprocal(0.125), Some(8.0));
    for k in [3.0, 0.1, -7.0, 1.5] {
        assert_eq!(exact_reciprocal(k), None, "{k} has no exact reciprocal");
    }
    assert_eq!(exact_reciprocal(f64::from_bits(1)), None, "subnormal");
    assert_eq!(exact_reciprocal(0.0), None);

    match lower_div_constant(0, &[2.0, 4.0], "t").expect("lowers") {
        Op::MulConst { k, .. } => assert_eq!(k, vec![0.5, 0.25]),
        other => panic!("powers of two must lower to an exact MulConst: {other:?}"),
    }
    match lower_div_constant(0, &[3.0, 2.0], "t").expect("lowers") {
        Op::Sparse {
            terms, k_lo, k_hi, ..
        } => {
            assert_eq!(k_lo, vec![0.0, 0.0]);
            assert_eq!(k_hi, vec![0.0, 0.0]);
            for (row, kj) in terms.iter().zip([3.0f64, 2.0]) {
                let [t] = row.as_slice() else {
                    panic!("one term per element")
                };
                let q = 1.0 / kj;
                // The coefficient interval must contain the true 1/k: it
                // contains both neighbours of the rounded quotient.
                assert!(t.coef - t.err <= next_down(q) && t.coef + t.err >= next_up(q));
                assert!(t.err > 0.0 && t.err < 1e-15);
            }
        }
        other => panic!("a non power of two must lower to an interval row: {other:?}"),
    }
    assert!(lower_div_constant(0, &[0.0], "t").is_err());
}

/// The twin is lowered ONLY while it is bound to the graph it was attached
/// to: a structural mutation makes it stale and the lane falls back to the
/// graph itself (which declines on its OpaqueSkip). And a twin is refused
/// outright while it still carries an OpaqueSkip or carries nothing enclosed.
#[test]
fn a_twin_is_lowered_only_while_bound_to_its_graph() {
    let dark = |out: &str| {
        let mut g = GraphNetwork::new();
        g.add_node(GraphNode::new(
            "l1",
            Layer::Linear(LinearLayer::new(mat(&[vec![1.0, 0.5], vec![-0.5, 1.0]]), None).unwrap()),
            vec![NETWORK_INPUT.into()],
        ));
        g.add_node(GraphNode::new(
            "skip",
            Layer::OpaqueSkip(OpaqueSkipLayer::new()),
            vec!["l1".into()],
        ));
        g.set_output(out);
        g
    };
    let twin_graph = || {
        let mut g = GraphNetwork::new();
        g.add_node(GraphNode::new(
            "l1",
            Layer::Linear(LinearLayer::new(mat(&[vec![1.0, 0.5], vec![-0.5, 1.0]]), None).unwrap()),
            vec![NETWORK_INPUT.into()],
        ));
        g.add_node(GraphNode::new(
            "skip",
            Layer::IntervalConst(
                IntervalConstLayer::try_new_linear(
                    LinearLayer::new(mat(&[vec![0.5, 0.25], vec![0.0, 1.0]]), None).unwrap(),
                    mat(&[vec![1e-6, 1e-6], vec![0.0, 1e-6]]),
                )
                .unwrap(),
            ),
            vec!["l1".into()],
        ));
        g.set_output("skip");
        g
    };
    let input = boxed(&[-1.0, -1.0], &[1.0, 1.0]);

    // Refusals.
    assert!(
        LaneTwin::new(dark("skip"), 1).is_err(),
        "a twin with an OpaqueSkip"
    );
    let mut plain = GraphNetwork::new();
    plain.add_node(GraphNode::new(
        "l1",
        Layer::Linear(LinearLayer::new(mat(&[vec![1.0, 0.5]]), None).unwrap()),
        vec![NETWORK_INPUT.into()],
    ));
    plain.set_output("l1");
    assert!(
        LaneTwin::new(plain, 0).is_err(),
        "a twin with nothing enclosed"
    );

    let mut g = dark("skip");
    assert!(admit(&g, &input).is_err(), "the dark graph must decline");
    assert_eq!(g.lowdim_twin_state(), "none");
    g.attach_lowdim_twin(LaneTwin::new(twin_graph(), 1).expect("twin"));
    assert_eq!(g.lowdim_twin_state(), "bound");
    let adm = admit(&g, &input).expect("the bound twin lowers");
    assert!(adm.from_twin());
    assert_eq!(adm.interval_consts(), 1);
    // A clone is the same model: still bound.
    let cloned = g.clone();
    assert_eq!(cloned.lowdim_twin_state(), "bound");
    assert!(admit(&cloned, &input).is_ok_and(|a| a.from_twin()));
    // A structural mutation (output retarget) makes the binding stale.
    g.set_output("skip");
    assert_eq!(g.lowdim_twin_state(), "stale");
    assert!(
        admit(&g, &input).is_err(),
        "a stale twin must not be lowered; the dark graph declines"
    );
}

// ------------------------------------------------------------------------
// #lsnc-lane stage 2: the joint quadratic bound on lowered programs
// ------------------------------------------------------------------------

/// The Lyapunov fragment's `V = ReduceSum(cat * IntervalLinear(cat))` is ONE
/// joint site, found structurally and named after its graph node.
#[test]
fn the_lyapunov_fragment_has_one_named_joint_site() {
    let (g, _) = lyap(3, 1e-3);
    let adm = admit(&g, &boxed(&LO, &HI)).expect("lowers");
    assert_eq!(adm.joint_sites(), 1);
    assert_eq!(adm.joint_site_names(), vec!["v".to_string()]);
}

/// `x` (3) -> Linear(5) -> ReLU -> Linear(4) = `a`; `b = Q a`;
/// `v = ReduceSum(a * b) = a'Q a`: both factors are NON-affine functions of
/// the same input (the McCormick worst case the joint bound exists for).
struct Corr {
    w1: Vec<Vec<f64>>,
    b1: Vec<f64>,
    w2: Vec<Vec<f64>>,
    b2: Vec<f64>,
    q: Vec<Vec<f64>>,
}

impl Corr {
    fn eval(&self, x: &[f64]) -> f64 {
        let h: Vec<f64> = self
            .w1
            .iter()
            .zip(&self.b1)
            .map(|(row, b)| (row.iter().zip(x).map(|(w, v)| w * v).sum::<f64>() + b).max(0.0))
            .collect();
        let a: Vec<f64> = self
            .w2
            .iter()
            .zip(&self.b2)
            .map(|(row, b)| row.iter().zip(&h).map(|(w, v)| w * v).sum::<f64>() + b)
            .collect();
        let qa: Vec<f64> = self
            .q
            .iter()
            .map(|row| row.iter().zip(&a).map(|(w, v)| w * v).sum::<f64>())
            .collect();
        a.iter().zip(&qa).map(|(p, r)| p * r).sum()
    }
}

fn corr(seed: u64) -> (GraphNetwork, Corr) {
    let mut rng = Lcg(seed);
    let grid = |rng: &mut Lcg, r: usize, c: usize, s: f64| -> Vec<Vec<f64>> {
        (0..r)
            .map(|_| (0..c).map(|_| f64::from(rng.f32(s))).collect())
            .collect()
    };
    let spec = Corr {
        w1: grid(&mut rng, 5, 3, 1.0),
        b1: (0..5).map(|_| f64::from(rng.f32(0.3))).collect(),
        w2: grid(&mut rng, 4, 5, 0.8),
        b2: (0..4).map(|_| f64::from(rng.f32(0.2))).collect(),
        q: grid(&mut rng, 4, 4, 1.0),
    };
    let vec32 = |v: &[f64]| Array1::from_vec(v.iter().map(|&x| x as f32).collect());
    let mut g = GraphNetwork::new();
    g.add_node(GraphNode::new(
        "l1",
        Layer::Linear(LinearLayer::new(mat(&spec.w1), Some(vec32(&spec.b1))).unwrap()),
        vec![NETWORK_INPUT.into()],
    ));
    g.add_node(GraphNode::new(
        "r1",
        Layer::ReLU(ReLULayer::new()),
        vec!["l1".into()],
    ));
    g.add_node(GraphNode::new(
        "a",
        Layer::Linear(LinearLayer::new(mat(&spec.w2), Some(vec32(&spec.b2))).unwrap()),
        vec!["r1".into()],
    ));
    g.add_node(GraphNode::new(
        "b",
        Layer::Linear(LinearLayer::new(mat(&spec.q), None).unwrap()),
        vec!["a".into()],
    ));
    g.add_node(GraphNode::new(
        "m",
        Layer::MulBinary(MulBinaryLayer),
        vec!["a".into(), "b".into()],
    ));
    g.add_node(GraphNode::new(
        "v",
        Layer::ReduceSum(ReduceSumLayer::new(vec![-1], true)),
        vec!["m".into()],
    ));
    g.set_output("v");
    (g, spec)
}

const CLO: [f64; 3] = [-1.0, -0.75, -0.5];
const CHI: [f64; 3] = [1.0, 0.5, 1.0];

/// SOUNDNESS of the joint bound on a correlated site: on boxes of every
/// width, (a) the published bounds of `+v` and `-v` contain every sampled
/// value, with the joint armed, with and without refinement; (b) the joint
/// PLANES of the same pass bound `v` from both sides at every sample of their
/// own box; (c) the site interval contains every sample. And NON-VACUITY: the
/// joint engages (planes on every box) and is strictly tighter than the
/// unarmed lane on the root box.
#[test]
fn a_correlated_site_is_contained_by_the_joint_bound_and_its_planes() {
    for seed in [11u64, 12, 13] {
        let (g, spec) = corr(seed);
        let adm = admit(&g, &boxed(&CLO, &CHI)).expect("lowers");
        assert_eq!(adm.joint_sites(), 1, "seed {seed}: one site");
        let mut rng = Lcg(seed * 31 + 7);
        let (blo, bhi) = sub_boxes(&mut rng, &CLO, &CHI, 40);
        let objectives = vec![vec![1.0], vec![-1.0]];
        let off = LowDimConfig::default();
        for cfg in [
            LowDimConfig {
                joint_quadratic: true,
                ..LowDimConfig::default()
            },
            LowDimConfig {
                joint_quadratic: true,
                refine_max_width: 8,
                chunk: 3,
                ..LowDimConfig::default()
            },
        ] {
            let lbs = bound_boxes(&adm, &blo, &bhi, &objectives, &cfg).expect("bounds");
            let sites = joint_site_boxes(&adm, &blo, &bhi, &cfg).expect("site boxes");
            assert_eq!((lbs.len(), sites.len()), (40, 40));
            for b in 0..40 {
                let (l, h) = (&blo[b * 3..(b + 1) * 3], &bhi[b * 3..(b + 1) * 3]);
                let site = &sites[b][0];
                let (Some((gl, cl)), Some((gh, chh))) = (&site.lower, &site.upper) else {
                    panic!("seed {seed} box {b}: the armed joint produced no planes");
                };
                let mut xs: Vec<Vec<f64>> = (0..300).map(|_| sample(&mut rng, l, h)).collect();
                for corner in 0..8 {
                    xs.push(
                        (0..3)
                            .map(|k| if corner >> k & 1 == 1 { h[k] } else { l[k] })
                            .collect(),
                    );
                }
                for x in &xs {
                    let v = spec.eval(x);
                    let tol = 1e-12 * (1.0 + v.abs());
                    assert!(
                        lbs[b][0] <= v + tol && -lbs[b][1] >= v - tol,
                        "seed {seed} box {b}: v = {v} escapes the published [{}, {}]",
                        lbs[b][0],
                        -lbs[b][1]
                    );
                    assert!(
                        site.lo <= v + tol && v <= site.hi + tol,
                        "seed {seed} box {b}: v = {v} escapes the site interval [{}, {}]",
                        site.lo,
                        site.hi
                    );
                    let at =
                        |gg: &[f64], c: f64| gg.iter().zip(x).map(|(p, q)| p * q).sum::<f64>() + c;
                    let (pl, ph) = (at(gl.as_slice(), *cl), at(gh.as_slice(), *chh));
                    assert!(
                        pl <= v + tol && v <= ph + tol,
                        "seed {seed} box {b}: v = {v} escapes the joint planes [{pl}, {ph}]"
                    );
                }
            }
        }
        // Diagnostic only: on a ReLU-fed site the joint bound is USUALLY much
        // tighter, but not provably so (the operands' residual radii can
        // exceed McCormick's gap). Tightness is pinned on a pure quadratic
        // below and on the real lsnc row-2 root in ny-onnx.
        let (root_lo, root_hi) = (CLO.to_vec(), CHI.to_vec());
        let on = bound_boxes(
            &adm,
            &root_lo,
            &root_hi,
            &objectives,
            &LowDimConfig {
                joint_quadratic: true,
                ..LowDimConfig::default()
            },
        )
        .expect("bounds")[0]
            .clone();
        let plain =
            bound_boxes(&adm, &root_lo, &root_hi, &objectives, &off).expect("bounds")[0].clone();
        eprintln!("[lsnc-joint-test] seed {seed} root: joint {on:?} vs plain {plain:?}");
    }
}

/// NON-VACUITY, pinned: on a PURE quadratic site `v = a'Qa`, `a = W x`, `Q`
/// positive definite, both operands are exactly affine (radius ~0), so the
/// joint upper bound is the exact vertex maximum up to rounding — while the
/// unarmed lane (McCormick per product) is strictly looser.
#[test]
fn the_joint_bound_is_exact_on_a_pure_quadratic_site() {
    let w: Vec<Vec<f64>> = vec![
        vec![1.0, -0.5, 0.25],
        vec![0.5, 1.0, -0.75],
        vec![-0.25, 0.5, 1.0],
    ];
    // Q = B'B + 0.125 I, stored as f32 (every entry here is exact in f32).
    let bm = [[1.0, 0.5, 0.0], [0.0, 1.0, -0.5], [0.25, 0.0, 1.0]];
    let mut q = vec![vec![0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            q[i][j] =
                (0..3).map(|k| bm[k][i] * bm[k][j]).sum::<f64>() + if i == j { 0.125 } else { 0.0 };
            q[i][j] = f64::from(q[i][j] as f32);
        }
    }
    let mut g = GraphNetwork::new();
    g.add_node(GraphNode::new(
        "a",
        Layer::Linear(LinearLayer::new(mat(&w), None).unwrap()),
        vec![NETWORK_INPUT.into()],
    ));
    g.add_node(GraphNode::new(
        "b",
        Layer::Linear(LinearLayer::new(mat(&q), None).unwrap()),
        vec!["a".into()],
    ));
    g.add_node(GraphNode::new(
        "m",
        Layer::MulBinary(MulBinaryLayer),
        vec!["a".into(), "b".into()],
    ));
    g.add_node(GraphNode::new(
        "v",
        Layer::ReduceSum(ReduceSumLayer::new(vec![-1], true)),
        vec!["m".into()],
    ));
    g.set_output("v");
    let eval = |x: &[f64]| -> f64 {
        let a: Vec<f64> = w
            .iter()
            .map(|r| r.iter().zip(x).map(|(p, t)| p * t).sum())
            .collect();
        let qa: Vec<f64> = q
            .iter()
            .map(|r| r.iter().zip(&a).map(|(p, t)| p * t).sum())
            .collect();
        a.iter().zip(&qa).map(|(p, t)| p * t).sum()
    };
    let adm = admit(&g, &boxed(&CLO, &CHI)).expect("lowers");
    // v is convex in x: its maximum over the box is at a vertex.
    let vmax = (0..8usize)
        .map(|c| {
            let x: Vec<f64> = (0..3)
                .map(|k| if c >> k & 1 == 1 { CHI[k] } else { CLO[k] })
                .collect();
            eval(&x)
        })
        .fold(f64::NEG_INFINITY, f64::max);
    let armed = LowDimConfig {
        joint_quadratic: true,
        ..LowDimConfig::default()
    };
    let site = &joint_site_boxes(&adm, &CLO, &CHI, &armed).expect("boxes")[0][0];
    assert!(site.upper.is_some(), "the joint ran");
    assert!(
        site.hi >= vmax && site.hi <= vmax + 1e-9 * (1.0 + vmax.abs()),
        "joint site upper bound {} vs the exact vertex maximum {vmax}",
        site.hi
    );
    let ub =
        |cfg: &LowDimConfig| -bound_boxes(&adm, &CLO, &CHI, &[vec![-1.0]], cfg).expect("b")[0][0];
    let (joint_ub, plain_ub) = (ub(&armed), ub(&LowDimConfig::default()));
    eprintln!("[lsnc-joint-test] pure quadratic: max {vmax} joint {joint_ub} plain {plain_ub}");
    assert!(joint_ub >= vmax && joint_ub <= vmax + 1e-9 * (1.0 + vmax.abs()));
    assert!(
        plain_ub > vmax + 1e-6,
        "McCormick is expected to be strictly looser here ({plain_ub} vs {vmax})"
    );
}

/// Unarmed, the site is found but the joint pass never runs: no planes, and
/// the published bounds are the stage-1 lane's.
#[test]
fn an_unarmed_lane_publishes_no_joint_planes() {
    let (g, _) = corr(11);
    let adm = admit(&g, &boxed(&CLO, &CHI)).expect("lowers");
    let sites = joint_site_boxes(&adm, &CLO, &CHI, &LowDimConfig::default()).expect("boxes");
    assert_eq!(sites.len(), 1);
    assert!(sites[0][0].lower.is_none() && sites[0][0].upper.is_none());
    let report = joint_root_report(&adm, &LowDimConfig::default());
    assert!(!report.armed);
    assert_eq!(report.sites.len(), 1);
    // A width-1 output has no `v_lb` slot: the report says so with NaN.
    assert!(report.dv_ub.is_nan() && report.v_lb.is_nan());
}

/// End to end through the branch and bound, joint armed: a threshold the
/// function EXCEEDS somewhere must never verify; one comfortably above its
/// maximum must.
#[test]
fn the_joint_lane_never_proves_a_false_bound_and_proves_a_true_one() {
    let (g, spec) = corr(12);
    let adm = admit(&g, &boxed(&CLO, &CHI)).expect("lowers");
    let mut rng = Lcg(99);
    let mut vmax = f64::NEG_INFINITY;
    let mut vmin = f64::INFINITY;
    for _ in 0..20_000 {
        let v = spec.eval(&sample(&mut rng, &CLO, &CHI));
        vmax = vmax.max(v);
        vmin = vmin.min(v);
    }
    let cfg = LowDimConfig {
        joint_quadratic: true,
        refine_max_width: 8,
        batch: 256,
        chunk: 8,
        ..LowDimConfig::default()
    };
    let deadline = || std::time::Instant::now() + Duration::from_secs(20);
    // "v < t" is lower(-v) > -t.
    let claim = |t: f64| {
        let mut witness = None;
        prove_conjunction_admitted(
            &adm,
            &[vec![-1.0f32]],
            &[-(t as f32)],
            deadline(),
            &cfg,
            &mut witness,
        )
    };
    let false_t = vmax - 0.05 * (vmax - vmin).max(1e-3);
    match claim(false_t) {
        LowDimOutcome::Verified(s) => {
            panic!("PROVED v < {false_t} although a sample reaches {vmax}: a FALSE PROOF ({s:?})")
        }
        other => eprintln!("[lsnc-joint-test] false claim v < {false_t}: {other:?}"),
    }
    let true_t = vmax + 0.25 * (vmax - vmin).max(1e-3);
    match claim(true_t) {
        LowDimOutcome::Verified(s) => {
            eprintln!("[lsnc-joint-test] true claim v < {true_t}: verified {s:?}");
        }
        other => panic!("the armed lane failed to prove v < {true_t} (max {vmax}): {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// #lsnc-lane stage 3: clip, smart split, parallel waves
// ---------------------------------------------------------------------------

/// The `allmax` stack of the numpy gate as a lane config (joint, refinement,
/// clip, `sbwr`), in the given parallel mode, with small waves and chunks so a
/// small tree still spans several waves and chunks.
fn allmax(parallel: LowDimParallel) -> LowDimConfig {
    LowDimConfig {
        joint_quadratic: true,
        refine_max_width: 8,
        clip: true,
        split: LowDimSplit::SbWr,
        parallel,
        batch: 64,
        chunk: 4,
        ..LowDimConfig::default()
    }
}

/// The `q`-quantile of `v` (nearest rank).
fn quantile(v: &[f64], q: f64) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    s[((s.len() - 1) as f64 * q).round() as usize]
}

/// CLIP SOUNDNESS on a lane program, with an INDEPENDENT f64 reference: on
/// boxes of every width, (a) every clip plane the lane computes for a box
/// (each backward pass's own `a . x + c <= objective . y(x)`) holds at every
/// sample of the box, for EVERY member of the interval-constant set (centre
/// and extreme members); (b) with synthetic thresholds at the 0.8 quantile of
/// each objective over the box (so a real share of the samples satisfies the
/// whole conjunction), every sample that satisfies every conjunct — for any
/// member — is still inside the clipped box, and an emptied box holds none.
/// Non-vacuity: thousands of such points, and the clip shrinks some boxes.
#[test]
fn the_clip_keeps_every_sampled_counterexample_of_every_member() {
    for (seed, radius) in [(21u64, 1e-3), (22, 2e-2)] {
        let (g, spec) = lyap(seed, radius);
        let adm = admit(&g, &boxed(&LO, &HI)).expect("lowers");
        let mut rng = Lcg(seed * 17 + 5);
        let (blo, bhi) = sub_boxes(&mut rng, &LO, &HI, 30);
        // A clause over V (= Y_1) and z_0 / 2 (= Y_2), in both directions.
        let objectives: Vec<Vec<f64>> = vec![
            vec![0.0, -1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, -1.0, 0.0, 0.0],
        ];
        let members: Vec<Member> = std::iter::once(spec.centre())
            .chain((0..4).map(|_| spec.extreme(&mut rng)))
            .collect();
        let cfg = allmax(LowDimParallel::Serial);
        let (mut tested, mut shrunk, mut planes_seen) = (0usize, 0usize, 0usize);
        for b in 0..30 {
            let (l, h) = (&blo[b * 3..(b + 1) * 3], &bhi[b * 3..(b + 1) * 3]);
            let mut points: Vec<Vec<f64>> = (0..400).map(|_| sample(&mut rng, l, h)).collect();
            for corner in 0..8 {
                points.push(
                    (0..3)
                        .map(|k| if corner >> k & 1 == 1 { h[k] } else { l[k] })
                        .collect(),
                );
            }
            // values[p][member][objective]
            let values: Vec<Vec<Vec<f64>>> = points
                .iter()
                .map(|x| {
                    members
                        .iter()
                        .map(|mem| {
                            let y = spec.eval(x, mem);
                            objectives
                                .iter()
                                .map(|o| o.iter().zip(&y).map(|(a, v)| a * v).sum())
                                .collect()
                        })
                        .collect()
                })
                .collect();
            // V's two-sided band at the centre member's [0.2, 0.8] quantiles;
            // the one-sided z conjunct at its median, so its plane cuts about
            // half of each box (a dominant coordinate then moves a face).
            let thresholds: Vec<f64> = [0.8, 0.8, 0.5]
                .iter()
                .enumerate()
                .map(|(i, &qi)| {
                    let col: Vec<f64> = values.iter().map(|pv| pv[0][i]).collect();
                    quantile(&col, qi)
                })
                .collect();
            let report = clip_report(&adm, l, h, &objectives, &thresholds, &cfg)
                .expect("clip report")
                .pop()
                .expect("one box");
            planes_seen += report.planes.len();
            for (p, x) in points.iter().enumerate() {
                for (mi, mv) in values[p].iter().enumerate() {
                    for (i, a, c) in &report.planes {
                        let pv = a.iter().zip(x).map(|(u, w)| u * w).sum::<f64>() + c;
                        let tol = 1e-12 * (1.0 + mv[*i].abs());
                        assert!(
                            pv <= mv[*i] + tol,
                            "seed {seed} box {b} member {mi}: plane of objective {i} gives {pv} \
                             above the value {} at {x:?} — a FALSE PLANE",
                            mv[*i]
                        );
                    }
                    let counterexample = mv.iter().zip(&thresholds).all(|(v, t)| v <= t);
                    if !counterexample {
                        continue;
                    }
                    tested += 1;
                    match &report.clipped {
                        None => panic!(
                            "seed {seed} box {b} member {mi}: the clip EMPTIED a box holding the \
                             counterexample {x:?}"
                        ),
                        Some((cl, ch)) => assert!(
                            (0..3).all(|k| cl[k] <= x[k] && x[k] <= ch[k]),
                            "seed {seed} box {b} member {mi}: the clip cut the counterexample \
                             {x:?} out of [{cl:?}, {ch:?}]"
                        ),
                    }
                }
            }
            if let Some((cl, ch)) = &report.clipped {
                shrunk += usize::from((0..3).any(|k| cl[k] > l[k] || ch[k] < h[k]));
            }
        }
        eprintln!(
            "[lsnc-clip-test] seed {seed}: counterexamples_tested={tested} boxes_shrunk={shrunk} \
             planes={planes_seen}"
        );
        assert!(
            tested >= 3000,
            "VOID: only {tested} counterexample points tested"
        );
        assert!(planes_seen >= 30, "VOID: only {planes_seen} planes");
        assert!(shrunk > 0, "VOID: the clip never shrank a box");
    }
}

/// Outcome equality up to wall time: the variant, the stop reason and every
/// counter except `secs`.
fn same_outcome(a: &LowDimOutcome, b: &LowDimOutcome) -> bool {
    let key = |s: &LowDimStats| {
        (
            s.domains,
            s.closed,
            s.max_depth,
            s.waves,
            s.clip_closed,
            s.clip_shrunk,
            s.joint_hits,
            s.joint_sites,
        )
    };
    match (a, b) {
        (LowDimOutcome::Verified(x), LowDimOutcome::Verified(y)) => key(x) == key(y),
        (LowDimOutcome::Unproved(x, p), LowDimOutcome::Unproved(y, q)) => {
            p == q && key(x) == key(y)
        }
        (LowDimOutcome::Declined(x), LowDimOutcome::Declined(y)) => x == y,
        _ => false,
    }
}

/// PARALLEL == SERIAL: the full stack (joint, refinement, clip, `sbwr`) in
/// every wave mode — Serial (one thread), Bound (the historical rayon bound
/// with a serial decision) and Full (bound and decide in the rayon tasks) —
/// gives the SAME outcome, the same counters and the same candidate point.
/// Also for the historical Widest split without the clip, so the default
/// lane's `Bound` path is pinned to the serial reference too. Three
/// two-conjunct clauses (both conjuncts' planes clip): far above the sampled
/// maximum (impossible), just above it (a deeper tree, whatever its verdict)
/// and below it (attained: never proved). Whether each is provable does not
/// matter for the equality — only that no run stops on the wall clock, the
/// one stop that is not a function of the domains. Non-vacuity: some run
/// spans at least 3 waves.
#[test]
fn every_wave_mode_gives_the_same_verdict_counters_and_witness() {
    let (g, spec) = corr(12);
    let adm = admit(&g, &boxed(&CLO, &CHI)).expect("lowers");
    let mut rng = Lcg(77);
    let (mut vmax, mut vmin) = (f64::NEG_INFINITY, f64::INFINITY);
    for _ in 0..20_000 {
        let v = spec.eval(&sample(&mut rng, &CLO, &CHI));
        vmax = vmax.max(v);
        vmin = vmin.min(v);
    }
    let range = (vmax - vmin).max(1e-3);
    // "v >= lo_t AND v <= hi_t": conjuncts `-v <= -lo_t` and `v <= hi_t`.
    let clause = |lo_t: f64, hi_t: f64| {
        (
            vec![vec![-1.0f32], vec![1.0f32]],
            vec![-(lo_t as f32), hi_t as f32],
        )
    };
    let claims = [
        ("far", clause(vmax + 0.25 * range, vmax + 2.0 * range)),
        ("near", clause(vmax + 0.02 * range, vmax + 2.0 * range)),
        ("attained", clause(vmax - 0.05 * range, vmax + 1.0)),
    ];
    let deadline = || Instant::now() + Duration::from_secs(120);
    let mut max_waves = 0u64;
    for (label, base) in [
        ("allmax", allmax(LowDimParallel::Serial)),
        (
            "widest",
            LowDimConfig {
                refine_max_width: 8,
                batch: 64,
                chunk: 4,
                ..LowDimConfig::default()
            },
        ),
    ] {
        for (what, (objs, thrs)) in &claims {
            let run = |parallel: LowDimParallel| {
                let cfg = LowDimConfig {
                    parallel,
                    ..base.clone()
                };
                let mut witness = None;
                let out =
                    prove_conjunction_admitted(&adm, objs, thrs, deadline(), &cfg, &mut witness);
                (out, witness)
            };
            let (serial, w_serial) = run(LowDimParallel::Serial);
            let (bound, w_bound) = run(LowDimParallel::Bound);
            let (full, w_full) = run(LowDimParallel::Full);
            eprintln!("[lsnc-par-test] {label} {what}: serial {serial:?} | full {full:?}");
            assert!(
                !matches!(serial, LowDimOutcome::Unproved(_, LowDimStop::Deadline)),
                "{label} {what}: the reference run hit the deadline; the comparison is VOID"
            );
            assert!(
                same_outcome(&serial, &bound),
                "{label} {what}: Bound {bound:?} != Serial {serial:?}"
            );
            assert!(
                same_outcome(&serial, &full),
                "{label} {what}: Full {full:?} != Serial {serial:?}"
            );
            assert_eq!(w_serial, w_bound, "{label} {what}: Bound witness differs");
            assert_eq!(w_serial, w_full, "{label} {what}: Full witness differs");
            if let LowDimOutcome::Verified(s) | LowDimOutcome::Unproved(s, _) = &serial {
                max_waves = max_waves.max(s.waves);
            }
            if *what == "attained" {
                assert!(
                    !matches!(serial, LowDimOutcome::Verified(_)),
                    "{label}: PROVED a clause the sampled maximiser satisfies — a FALSE PROOF"
                );
            }
        }
    }
    assert!(
        max_waves >= 3,
        "VOID: no run spanned 3 waves (max {max_waves})"
    );
}

/// END TO END, full stack in the parallel mode the measurement arms: a clause
/// the network satisfies somewhere is never proved — on the correlated site,
/// and on the interval-constant Lyapunov fragment for its CENTRE member (a
/// dropped radius or a false clip plane would prove it) — and an impossible
/// clause is proved.
#[test]
fn the_full_stack_never_proves_an_attained_clause_and_proves_an_impossible_one() {
    let cfg = allmax(LowDimParallel::Full);
    let deadline = || Instant::now() + Duration::from_secs(60);
    // Correlated site: "v <= t" for t just below the sampled maximum ...
    let (g, spec) = corr(13);
    let adm = admit(&g, &boxed(&CLO, &CHI)).expect("lowers");
    let mut rng = Lcg(91);
    let (mut vmax, mut vmin) = (f64::NEG_INFINITY, f64::INFINITY);
    for _ in 0..20_000 {
        let v = spec.eval(&sample(&mut rng, &CLO, &CHI));
        vmax = vmax.max(v);
        vmin = vmin.min(v);
    }
    let range = (vmax - vmin).max(1e-3);
    // "v >= t" conjunct is `-v <= -t`; claim "no x has v >= t".
    let claim = |adm: &LaneAdmission, t: f64| {
        let mut w = None;
        prove_conjunction_admitted(
            adm,
            &[vec![-1.0f32]],
            &[-(t as f32)],
            deadline(),
            &cfg,
            &mut w,
        )
    };
    match claim(&adm, vmax - 0.05 * range) {
        LowDimOutcome::Verified(s) => {
            panic!("corr: PROVED no x reaches {vmax} - 5%, a sample does — FALSE PROOF ({s:?})")
        }
        other => eprintln!("[lsnc-stack-test] corr attained clause: {other:?}"),
    }
    match claim(&adm, vmax + 0.25 * range) {
        LowDimOutcome::Verified(s) => eprintln!("[lsnc-stack-test] corr impossible clause: {s:?}"),
        other => panic!("corr: failed to prove an impossible clause: {other:?}"),
    }
    // Lyapunov fragment with interval constants: "V >= t" for t just below
    // the CENTRE member's sampled maximum of V must never be proved.
    let (g, spec) = lyap(31, 2e-2);
    let adm = admit(&g, &boxed(&LO, &HI)).expect("lowers");
    let centre = spec.centre();
    let vmax = (0..20_000)
        .map(|_| spec.eval(&sample(&mut rng, &LO, &HI), &centre)[1])
        .fold(f64::NEG_INFINITY, f64::max);
    let mut w = None;
    let out = prove_conjunction_admitted(
        &adm,
        &[vec![0.0f32, -1.0, 0.0, 0.0, 0.0]],
        &[-((vmax - 1e-3 * (1.0 + vmax.abs())) as f32)],
        deadline(),
        &cfg,
        &mut w,
    );
    assert!(
        !matches!(out, LowDimOutcome::Verified(_)),
        "lyap: PROVED V < {vmax} although the centre member reaches it — FALSE PROOF ({out:?})"
    );
}

/// The family core (`prove_family_core`, the `#lowdim-fold` batch) takes the
/// same split rules: with `sbwr` it proves an impossible clause and never an
/// attained one, in one family run.
#[test]
fn the_family_core_with_the_smart_split_proves_only_impossible_clauses() {
    let (g, spec) = corr(11);
    let adm = admit(&g, &boxed(&CLO, &CHI)).expect("lowers");
    let mut rng = Lcg(5);
    let (mut vmax, mut vmin) = (f64::NEG_INFINITY, f64::INFINITY);
    for _ in 0..20_000 {
        let v = spec.eval(&sample(&mut rng, &CLO, &CHI));
        vmax = vmax.max(v);
        vmin = vmin.min(v);
    }
    let range = (vmax - vmin).max(1e-3);
    let fam = |t: f64| FamClause {
        prog: adm.prog.clone(),
        lo: CLO.to_vec(),
        hi: CHI.to_vec(),
        objs: vec![vec![-1.0]],
        thrs: vec![-t],
        combine: Combine::Each,
    };
    let cfg = LowDimConfig {
        split: LowDimSplit::SbWr,
        joint_quadratic: true,
        refine_max_width: 8,
        batch: 64,
        chunk: 4,
        ..LowDimConfig::default()
    };
    let out = prove_family_core(
        vec![fam(vmax + 0.25 * range), fam(vmax - 0.05 * range)],
        Instant::now() + Duration::from_secs(60),
        &cfg,
    );
    eprintln!("[lsnc-family-test] {out:?}");
    assert!(
        matches!(out[0], LowDimOutcome::Verified(_)),
        "the impossible clause must be proved: {:?}",
        out[0]
    );
    assert!(
        !matches!(out[1], LowDimOutcome::Verified(_)),
        "the attained clause was PROVED — FALSE PROOF: {:?}",
        out[1]
    );
}
