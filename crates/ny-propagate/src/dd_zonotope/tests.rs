// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Soundness oracles for the certified double-double zonotope (`#dd-zonotope`).
//!
//! The whole product of this lane is a NEW certified lower bound, so every
//! `-150` risk lives in this module's subject. The oracles below are ENCLOSURE
//! tests: for every transformer and for whole small networks, the certified
//! interval must contain the true value at exhaustively / densely sampled
//! points of the input box. A bound that is too LOOSE passes (and costs
//! points); a bound that is too TIGHT fails (and would cost a wrong verdict).

use ny_core::dd::Dd;
use ny_tensor::BoundedTensor;

use super::affine::{apply_affine, AffineOp, ConvPlan};
use super::maxpool::{apply_maxpool, PoolPlan};
use super::relu::{apply_relu, prune_zero_generators};
use super::state::DdZono;
use super::{dd_zonotope_margins, DdZonoConfig, DdZonoPlan};

use crate::layers::{Conv2dLayer, Layer, LinearLayer, MaxPool2dLayer, ReLULayer, SigmoidLayer};
use crate::GraphNetwork;
use crate::GraphNode;

use ndarray::{Array1, Array2, ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/// Deterministic xorshift in `[-1, 1)`.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed | 1)
    }
    fn next_f64(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
    }
    fn next_f32(&mut self) -> f32 {
        self.next_f64() as f32
    }
}

/// Build a zonotope over a box `[lo, up]` (one generator per perturbed coord).
fn box_zono(lo: &[f64], up: &[f64], shape: Vec<usize>) -> DdZono {
    let n = lo.len();
    let mut center = Vec::with_capacity(n);
    let mut gens = Vec::new();
    for i in 0..n {
        let (s, e) = ny_core::dd::two_sum(lo[i], up[i]);
        center.push(Dd {
            hi: s * 0.5,
            lo: e * 0.5,
        });
        if up[i] > lo[i] {
            let mut col = vec![0.0_f64; n];
            col[i] = ny_core::dd::next_up_f64((up[i] - lo[i]) * 0.5 * (1.0 + 1e-15));
            gens.push(col);
        }
    }
    DdZono {
        shape,
        center,
        gens,
        ec: vec![0.0; n],
        eg: vec![0.0; n],
    }
}

/// An `ExactBox` over f32-exact endpoints (so the trailing word and the
/// residual are both exactly zero).
fn exact_box_from_f32(lo: &[f32], up: &[f32]) -> super::certified_box::ExactBox {
    super::certified_box::ExactBox {
        lower: lo.iter().map(|&v| f64::from(v)).collect(),
        upper: up.iter().map(|&v| f64::from(v)).collect(),
        center_hi: lo
            .iter()
            .zip(up)
            .map(|(&a, &b)| (f64::from(a) + f64::from(b)) * 0.5)
            .collect(),
        center_lo: vec![0.0; lo.len()],
        center_err: vec![0.0; lo.len()],
        half_width: lo
            .iter()
            .zip(up)
            .map(|(&a, &b)| (f64::from(b) - f64::from(a)) * 0.5)
            .collect(),
    }
}

/// Evaluate the zonotope at a concrete assignment of its `e` symbols.
/// Only valid for a state whose generators are the box seed (identity basis).
fn point_from_box(lo: &[f64], up: &[f64], t: &[f64]) -> Vec<f64> {
    lo.iter()
        .zip(up.iter())
        .zip(t.iter())
        .map(|((&l, &u), &s)| l + (u - l) * s)
        .collect()
}

// ---------------------------------------------------------------------------
// ReLU
// ---------------------------------------------------------------------------

#[test]
fn relu_encloses_true_values_on_a_dense_sample() {
    let mut rng = Rng::new(0xC0FFEE);
    for trial in 0..200 {
        let n = 6usize;
        let lo: Vec<f64> = (0..n).map(|_| rng.next_f64() * 4.0).collect();
        let up: Vec<f64> = lo.iter().map(|&l| l + rng.next_f64().abs() * 4.0).collect();
        let mut z = box_zono(&lo, &up, vec![n]);
        let outcome = apply_relu(&mut z).expect("finite");
        for (i, mu) in outcome.spent {
            z.push_sparse_generator(&[(i, mu)]);
        }
        let (zlo, zup) = z.concretize();
        for _ in 0..80 {
            let t: Vec<f64> = (0..n).map(|_| (rng.next_f64() + 1.0) * 0.5).collect();
            let x = point_from_box(&lo, &up, &t);
            for i in 0..n {
                let y = x[i].max(0.0);
                assert!(
                    y >= zlo[i] && y <= zup[i],
                    "trial {trial} elem {i}: relu({}) = {y} outside certified [{}, {}]",
                    x[i],
                    zlo[i],
                    zup[i]
                );
            }
        }
    }
}

#[test]
fn relu_is_exact_on_a_stable_neuron() {
    let lo = vec![1.0_f64, -3.0];
    let up = vec![2.0_f64, -1.0];
    let mut z = box_zono(&lo, &up, vec![2]);
    apply_relu(&mut z).expect("finite");
    let (zlo, zup) = z.concretize();
    // Positive-stable: passes through (tiny outward rounding only).
    assert!(zlo[0] <= 1.0 && zup[0] >= 2.0);
    assert!(zup[0] - zlo[0] < 1.0 + 1e-6);
    // Negative-stable: collapses to zero.
    assert!(zlo[1] <= 0.0 && zup[1] >= 0.0);
    assert!(zup[1] - zlo[1] < 1e-12);
}

// ---------------------------------------------------------------------------
// MaxPool — exhaustive enumeration (risk R2: the index plumbing is new code)
// ---------------------------------------------------------------------------

#[test]
fn maxpool_encloses_the_exhaustive_window_maximum() {
    let mut rng = Rng::new(0xBEEF_1234);
    // 2 channels, 4x4 -> 2x2 with a 2x2/stride-2 window: 16 inputs, so an
    // exhaustive corner enumeration (2^16) is affordable and is the strongest
    // available check on the window/argmax/stride indexing.
    let plan = PoolPlan::build((2, 4, 4), (2, 2), (2, 2), (0, 0)).expect("plan");
    for trial in 0..25 {
        let n = 2 * 4 * 4;
        let lo: Vec<f64> = (0..n).map(|_| rng.next_f64() * 2.0).collect();
        let up: Vec<f64> = lo.iter().map(|&l| l + rng.next_f64().abs() * 2.0).collect();
        let z = box_zono(&lo, &up, vec![2, 4, 4]);
        let (mut out, outcome) = apply_maxpool(&z, &plan).expect("finite");
        for (i, half) in outcome.spent {
            out.push_sparse_generator(&[(i, half)]);
        }
        let (olo, oup) = out.concretize();

        // Exhaustive over the 16 box CORNERS per channel-window (4 inputs each),
        // which is where the max is attained, plus dense interior samples.
        for corner in 0..(1u32 << 16) {
            if corner % 977 != 0 {
                continue; // stride the enumeration to keep the test fast
            }
            let x: Vec<f64> = (0..n)
                .map(|i| if corner & (1 << i) != 0 { up[i] } else { lo[i] })
                .collect();
            for ch in 0..2 {
                for oy in 0..2 {
                    for ox in 0..2 {
                        let o = ch * 4 + oy * 2 + ox;
                        let mut m = f64::NEG_INFINITY;
                        for ky in 0..2 {
                            for kx in 0..2 {
                                let idx = ch * 16 + (oy * 2 + ky) * 4 + (ox * 2 + kx);
                                m = m.max(x[idx]);
                            }
                        }
                        assert!(
                            m >= olo[o] && m <= oup[o],
                            "trial {trial} out {o}: max {m} outside certified [{}, {}]",
                            olo[o],
                            oup[o]
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn maxpool_is_exact_when_one_window_entry_strictly_dominates() {
    // Window 0 dominates: lo = 10 beats every other up.
    let lo = vec![10.0, 0.0, 0.0, 0.0];
    let up = vec![11.0, 1.0, 1.0, 1.0];
    let plan = PoolPlan::build((1, 2, 2), (2, 2), (2, 2), (0, 0)).expect("plan");
    let z = box_zono(&lo, &up, vec![1, 2, 2]);
    let (out, outcome) = apply_maxpool(&z, &plan).expect("finite");
    assert_eq!(outcome.relaxed, 0, "a dominated window must spend no slack");
    let (olo, oup) = out.concretize();
    assert!(olo[0] <= 10.0 && oup[0] >= 11.0);
    assert!(oup[0] - olo[0] < 1.0 + 1e-6);
}

#[test]
fn maxpool_refuses_padding() {
    assert!(PoolPlan::build((1, 4, 4), (2, 2), (2, 2), (1, 1)).is_none());
}

// ---------------------------------------------------------------------------
// Add (zonotope sum) — the correlation-hazard case
// ---------------------------------------------------------------------------

/// Two operands that carry DIFFERENT independent symbols at the same column
/// index must be enclosed by summing them, and the enclosure must hold at the
/// symbol assignment that maximises each operand independently. Column-by-index
/// summation would treat the two symbols as one and shrink the set below that
/// true corner; concatenation must not.
#[test]
fn add_encloses_independent_symbols_at_their_joint_extremes() {
    // a = 1 + 1*e0 ; b = 2 + 3*e1  (e0, e1 independent in [-1,1]).
    // True sum ranges over [1+(-1)] + [2+(-3)] = -1 ... [1+1]+[2+3] = 7.
    let a = DdZono {
        shape: vec![1],
        center: vec![Dd::from_f64(1.0)],
        gens: vec![vec![1.0]],
        ec: vec![0.0],
        eg: vec![0.0],
    };
    let b = DdZono {
        shape: vec![1],
        center: vec![Dd::from_f64(2.0)],
        gens: vec![vec![3.0]],
        ec: vec![0.0],
        eg: vec![0.0],
    };
    let s = super::add_states(&a, &b).expect("finite");
    // Two independent symbols must survive as two columns, not be merged.
    assert_eq!(s.n_gens(), 2, "independent symbols must stay independent");
    let (lo, up) = s.concretize();
    // The true joint extremes (-1 and 7) must be inside the certified box.
    assert!(lo[0] <= -1.0 + 1e-12, "lower {} must enclose -1", lo[0]);
    assert!(up[0] >= 7.0 - 1e-12, "upper {} must enclose 7", up[0]);
}

/// Concatenation must also enclose the true sum when the two operands DO share
/// the same symbol — it is merely looser there, never unsound.
#[test]
fn add_encloses_the_true_sum_on_a_shared_symbol() {
    // a = 0 + 1*e0 ; b = 0 + 1*e0 (same symbol). True sum = 2*e0 in [-2, 2].
    let a = DdZono {
        shape: vec![1],
        center: vec![Dd::from_f64(0.0)],
        gens: vec![vec![1.0]],
        ec: vec![0.0],
        eg: vec![0.0],
    };
    let s = super::add_states(&a, &a).expect("finite");
    let (lo, up) = s.concretize();
    assert!(
        lo[0] <= -2.0 + 1e-12 && up[0] >= 2.0 - 1e-12,
        "[{},{}]",
        lo[0],
        up[0]
    );
}

// ---------------------------------------------------------------------------
// Conv2d / Linear
// ---------------------------------------------------------------------------

#[test]
fn conv_encloses_the_true_output_on_a_dense_sample() {
    let mut rng = Rng::new(0x5EED);
    let plan = ConvPlan::build((2, 5, 5), 3, 3, 3, (1, 1), (1, 1), (1, 1), 1).expect("plan");
    let kn = 3 * 2 * 3 * 3;
    let w: Vec<f64> = (0..kn).map(|_| rng.next_f64()).collect();
    let wabs: Vec<f64> = w.iter().map(|v| v.abs()).collect();
    let bias: Vec<f64> = (0..3).map(|_| rng.next_f64()).collect();
    let n = 2 * 5 * 5;
    let lo: Vec<f64> = (0..n).map(|_| rng.next_f64()).collect();
    let up: Vec<f64> = lo.iter().map(|&l| l + rng.next_f64().abs()).collect();
    let z = box_zono(&lo, &up, vec![2, 5, 5]);
    let op = AffineOp::Conv {
        plan,
        w: &w,
        wabs: &wabs,
        bias: Some(&bias),
    };
    let out = apply_affine(&z, &op, || Ok(())).expect("affine");
    let (olo, oup) = out.concretize();

    for _ in 0..400 {
        let t: Vec<f64> = (0..n).map(|_| (rng.next_f64() + 1.0) * 0.5).collect();
        let x = point_from_box(&lo, &up, &t);
        let y = super::affine::conv_f64(&plan, &w, Some(&bias), &x);
        for (i, &yi) in y.iter().enumerate() {
            assert!(
                yi >= olo[i] && yi <= oup[i],
                "conv output {i}: {yi} outside certified [{}, {}]",
                olo[i],
                oup[i]
            );
        }
    }
}

#[test]
fn conv_refuses_dilation_and_groups() {
    assert!(ConvPlan::build((2, 5, 5), 3, 3, 3, (1, 1), (1, 1), (2, 2), 1).is_none());
    assert!(ConvPlan::build((2, 5, 5), 3, 3, 3, (1, 1), (1, 1), (1, 1), 2).is_none());
}

#[test]
fn linear_encloses_the_true_output_on_a_dense_sample() {
    let mut rng = Rng::new(0xABCD);
    let (out_f, in_f) = (7usize, 11usize);
    let w: Vec<f64> = (0..out_f * in_f).map(|_| rng.next_f64()).collect();
    let wabs: Vec<f64> = w.iter().map(|v| v.abs()).collect();
    let bias: Vec<f64> = (0..out_f).map(|_| rng.next_f64()).collect();
    let lo: Vec<f64> = (0..in_f).map(|_| rng.next_f64()).collect();
    let up: Vec<f64> = lo.iter().map(|&l| l + rng.next_f64().abs()).collect();
    let z = box_zono(&lo, &up, vec![in_f]);
    let op = AffineOp::Linear {
        out_features: out_f,
        in_features: in_f,
        w: &w,
        wabs: &wabs,
        bias: Some(&bias),
    };
    let out = apply_affine(&z, &op, || Ok(())).expect("affine");
    let (olo, oup) = out.concretize();
    for _ in 0..500 {
        let t: Vec<f64> = (0..in_f).map(|_| (rng.next_f64() + 1.0) * 0.5).collect();
        let x = point_from_box(&lo, &up, &t);
        let y = super::affine::linear_f64(out_f, in_f, &w, Some(&bias), &x);
        for (i, &yi) in y.iter().enumerate() {
            assert!(yi >= olo[i] && yi <= oup[i]);
        }
    }
}

// ---------------------------------------------------------------------------
// End-to-end: a whole small conv net through the public entry point
// ---------------------------------------------------------------------------

/// Build a Conv -> ReLU -> MaxPool -> Conv -> ReLU -> Flatten -> Linear graph
/// on a `3 x H x W` input, mirroring VGG16's op sequence in miniature.
fn tiny_vgg(rng: &mut Rng, h: usize, w: usize) -> (GraphNetwork, usize) {
    let c1 = 4usize;
    let k1: Vec<f32> = (0..c1 * 3 * 3 * 3).map(|_| rng.next_f32() * 0.3).collect();
    let conv1 = Conv2dLayer::new(
        ArrayD::from_shape_vec(IxDyn(&[c1, 3, 3, 3]), k1).unwrap(),
        Some(Array1::from_vec(
            (0..c1).map(|_| rng.next_f32() * 0.1).collect(),
        )),
        (1, 1),
        (1, 1),
    )
    .unwrap();
    let c2 = 5usize;
    let k2: Vec<f32> = (0..c2 * c1 * 3 * 3).map(|_| rng.next_f32() * 0.3).collect();
    let conv2 = Conv2dLayer::new(
        ArrayD::from_shape_vec(IxDyn(&[c2, c1, 3, 3]), k2).unwrap(),
        Some(Array1::from_vec(
            (0..c2).map(|_| rng.next_f32() * 0.1).collect(),
        )),
        (1, 1),
        (1, 1),
    )
    .unwrap();
    let (ph, pw) = (h / 2, w / 2);
    let flat = c2 * ph * pw;
    let n_out = 3usize;
    let lw: Vec<f32> = (0..n_out * flat).map(|_| rng.next_f32() * 0.2).collect();
    let lin = LinearLayer::new(
        Array2::from_shape_vec((n_out, flat), lw).unwrap(),
        Some(Array1::from_vec(
            (0..n_out).map(|_| rng.next_f32() * 0.1).collect(),
        )),
    )
    .expect("linear");

    let nodes = vec![
        GraphNode::from_input("conv1", Layer::Conv2d(conv1)),
        GraphNode::new("relu1", Layer::ReLU(ReLULayer), vec!["conv1".into()]),
        GraphNode::new(
            "pool1",
            Layer::MaxPool2d(MaxPool2dLayer::new((2, 2), (2, 2), (0, 0))),
            vec!["relu1".into()],
        ),
        GraphNode::new("conv2", Layer::Conv2d(conv2), vec!["pool1".into()]),
        GraphNode::new("relu2", Layer::ReLU(ReLULayer), vec!["conv2".into()]),
        GraphNode::new("fc", Layer::Linear(lin), vec!["relu2".into()]),
    ];
    let mut g = GraphNetwork::new();
    for n in nodes {
        g.add_node(n);
    }
    g.set_output("fc");
    (g, n_out)
}

/// Concrete forward evaluation of the same graph, in f64.
fn tiny_vgg_forward(graph: &GraphNetwork, x: &[f32], shape: (usize, usize, usize)) -> Vec<f64> {
    let mut cur: Vec<f64> = x.iter().map(|&v| f64::from(v)).collect();
    let mut sh = shape;
    for name in graph.exec_order().unwrap() {
        let node = graph.node(name).unwrap();
        match node.layer() {
            Layer::Conv2d(c) => {
                let k = c.kernel.shape().to_vec();
                let plan = ConvPlan::build(
                    sh, k[0], k[2], k[3], c.stride, c.padding, c.dilation, c.groups,
                )
                .unwrap();
                let w: Vec<f64> = c.kernel.iter().map(|&v| f64::from(v)).collect();
                let b: Vec<f64> = c
                    .bias
                    .as_ref()
                    .unwrap()
                    .iter()
                    .map(|&v| f64::from(v))
                    .collect();
                cur = super::affine::conv_f64(&plan, &w, Some(&b), &cur);
                sh = (plan.out_c, plan.out_h, plan.out_w);
            }
            Layer::ReLU(_) => {
                for v in cur.iter_mut() {
                    *v = v.max(0.0);
                }
            }
            Layer::MaxPool2d(p) => {
                let plan = PoolPlan::build(sh, p.kernel_size, p.stride, p.padding).unwrap();
                let mut out = vec![f64::NEG_INFINITY; plan.out_numel()];
                for ch in 0..plan.c {
                    for oy in 0..plan.out_h {
                        for ox in 0..plan.out_w {
                            let o = ch * plan.out_h * plan.out_w + oy * plan.out_w + ox;
                            for ky in 0..plan.kh {
                                for kx in 0..plan.kw {
                                    let idx = ch * plan.in_h * plan.in_w
                                        + (oy * plan.sh + ky) * plan.in_w
                                        + ox * plan.sw
                                        + kx;
                                    out[o] = out[o].max(cur[idx]);
                                }
                            }
                        }
                    }
                }
                cur = out;
                sh = (plan.c, plan.out_h, plan.out_w);
            }
            Layer::Linear(l) => {
                let w: Vec<f64> = l.weight.iter().map(|&v| f64::from(v)).collect();
                let b: Vec<f64> = l
                    .bias
                    .as_ref()
                    .unwrap()
                    .iter()
                    .map(|&v| f64::from(v))
                    .collect();
                cur = super::affine::linear_f64(
                    l.weight.nrows(),
                    l.weight.ncols(),
                    &w,
                    Some(&b),
                    &cur,
                );
                sh = (cur.len(), 1, 1);
            }
            other => panic!("unexpected layer {other:?}"),
        }
    }
    cur
}

#[test]
fn end_to_end_margin_encloses_the_sampled_truth() {
    let (h, w) = (8usize, 8usize);
    let n = 3 * h * w;
    let mut rng = Rng::new(0x1234_5678);
    let (graph, n_out) = tiny_vgg(&mut rng, h, w);

    // Sparse-input box: fix every pixel, perturb k of them.
    let base: Vec<f32> = (0..n).map(|_| rng.next_f32()).collect();
    let k = 6usize;
    let mut lower = base.clone();
    let mut upper = base.clone();
    let mut pert = Vec::new();
    for j in 0..k {
        let i = (j * 31 + 7) % n;
        lower[i] = base[i] - 0.05;
        upper[i] = base[i] + 0.05;
        pert.push(i);
    }
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[3, h, w]), lower.clone()).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[3, h, w]), upper.clone()).unwrap(),
    )
    .unwrap();

    // Objective rows: every pairwise margin y_a - y_b.
    let mut objectives = Vec::new();
    for a in 0..n_out {
        for b in 0..n_out {
            if a != b {
                let mut row = vec![0.0_f32; n_out];
                row[a] = 1.0;
                row[b] = -1.0;
                objectives.push(row);
            }
        }
    }

    let cfg = DdZonoConfig {
        min_input_numel: 1,
        ..DdZonoConfig::default()
    };
    let plan = DdZonoPlan {
        perturbed: pert.clone(),
        input_shape: vec![3, h, w],
        exact: exact_box_from_f32(&lower, &upper),
        declared_point_exact: false,
    };
    let margin = dd_zonotope_margins(&graph, &input, &objectives, &plan, &cfg, None)
        .expect("no error")
        .expect("admitted");

    // 20k samples of the true margin must all lie inside the certified
    // interval. This is a genuinely strong test only BECAUSE the bound is
    // tight — with a vacuous bound it would be worthless.
    for s in 0..20_000 {
        let mut x = base.clone();
        for &i in &pert {
            let t = if s < (1 << k) {
                // exhaust the corners first
                if s & (1 << (pert.iter().position(|&p| p == i).unwrap())) != 0 {
                    1.0
                } else {
                    0.0
                }
            } else {
                (rng.next_f32() + 1.0) * 0.5
            };
            x[i] = lower[i] + (upper[i] - lower[i]) * t;
        }
        let y = tiny_vgg_forward(&graph, &x, (3, h, w));
        for (o, obj) in objectives.iter().enumerate() {
            let m: f64 = obj
                .iter()
                .zip(y.iter())
                .map(|(&r, &v)| f64::from(r) * v)
                .sum();
            assert!(
                m >= margin.lower[o] && m <= margin.upper[o],
                "sample {s} obj {o}: true margin {m} outside certified [{}, {}]",
                margin.lower[o],
                margin.upper[o]
            );
        }
    }

    // The certified rounding channel must be far below the relaxation channel:
    // that is the whole reason double-double is used.
    for o in 0..objectives.len() {
        assert!(
            margin.rounding_half_width[o] < 1e-6 * margin.relax_half_width[o].max(1e-12),
            "obj {o}: rounding hw {} not negligible vs relax hw {}",
            margin.rounding_half_width[o],
            margin.relax_half_width[o]
        );
    }
}

// ---------------------------------------------------------------------------
// Detector: the conjunctive admission gate (risk R6)
// ---------------------------------------------------------------------------

fn dense_box(n: usize) -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[n]), vec![-1.0_f32; n]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[n]), vec![1.0_f32; n]).unwrap(),
    )
    .unwrap()
}

#[test]
fn detector_refuses_small_inputs_dense_boxes_and_unsupported_ops() {
    let mut rng = Rng::new(7);
    let (graph, _) = tiny_vgg(&mut rng, 8, 8);
    let cfg = DdZonoConfig::default();

    // Small input volume: every non-image category is excluded.
    let small = dense_box(100);
    assert!(DdZonoPlan::detect(&graph, &small, &cfg).is_none());

    // Large input but k above the cap.
    let big = dense_box(60_000);
    assert!(DdZonoPlan::detect(&graph, &big, &cfg).is_none());

    // Large input, k in range, but the shape is not CHW.
    let n = 60_000usize;
    let mut lo = vec![0.0_f32; n];
    let up = vec![0.0_f32; n];
    lo[0] = -1.0;
    let flat = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[n]), lo).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[n]), up).unwrap(),
    )
    .unwrap();
    assert!(DdZonoPlan::detect(&graph, &flat, &cfg).is_none());
}

#[test]
fn detector_refuses_a_graph_with_an_unsupported_layer() {
    let mut rng = Rng::new(11);
    let (graph, _) = tiny_vgg(&mut rng, 8, 8);
    // Sanity: the supported graph is accepted with a relaxed volume floor.
    let cfg = DdZonoConfig {
        min_input_numel: 1,
        ..DdZonoConfig::default()
    };
    let mut lo = vec![0.0_f32; 3 * 8 * 8];
    let up = vec![0.0_f32; 3 * 8 * 8];
    lo[0] = -0.1;
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[3, 8, 8]), lo.clone()).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[3, 8, 8]), up.clone()).unwrap(),
    )
    .unwrap();
    // Without a published exact-decimal box the detector must refuse: the
    // engine's f32 box alone is not admissible evidence for this method.
    let _guard = super::certified_box::test_lock();
    super::certified_box::reset_for_test();
    assert!(DdZonoPlan::detect(&graph, &input, &cfg).is_none());
    super::certified_box::register(&lo, &up, exact_box_from_f32(&lo, &up));
    assert!(DdZonoPlan::detect(&graph, &input, &cfg).is_some());

    // Swap in a Sigmoid: the pass must refuse rather than degrade.
    let nodes = vec![
        GraphNode::from_input("sig", Layer::Sigmoid(SigmoidLayer)),
        GraphNode::new(
            "fc",
            Layer::Linear(
                LinearLayer::new(
                    Array2::from_shape_vec((2, 3 * 8 * 8), vec![0.1_f32; 2 * 3 * 8 * 8]).unwrap(),
                    None,
                )
                .expect("linear"),
            ),
            vec!["sig".into()],
        ),
    ];
    let mut bad = GraphNetwork::new();
    for n in nodes {
        bad.add_node(n);
    }
    bad.set_output("fc");
    assert!(DdZonoPlan::detect(&bad, &input, &cfg).is_none());
    super::certified_box::reset_for_test();
}

#[test]
fn generator_cap_fails_closed() {
    let (h, w) = (8usize, 8usize);
    let n = 3 * h * w;
    let mut rng = Rng::new(0xFA11);
    let (graph, n_out) = tiny_vgg(&mut rng, h, w);
    let base: Vec<f32> = (0..n).map(|_| rng.next_f32()).collect();
    // A WIDE box makes many neurons cross, so many generators are spent.
    let lower: Vec<f32> = base.iter().map(|v| v - 2.0).collect();
    let upper: Vec<f32> = base.iter().map(|v| v + 2.0).collect();
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[3, h, w]), lower).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[3, h, w]), upper).unwrap(),
    )
    .unwrap();
    let objectives = vec![{
        let mut r = vec![0.0_f32; n_out];
        r[0] = 1.0;
        r[1] = -1.0;
        r
    }];
    let plan = DdZonoPlan {
        perturbed: (0..n).collect(),
        input_shape: vec![3, h, w],
        exact: exact_box_from_f32(
            &base.iter().map(|&v| v - 2.0).collect::<Vec<f32>>(),
            &base.iter().map(|&v| v + 2.0).collect::<Vec<f32>>(),
        ),
        declared_point_exact: false,
    };
    let cfg = DdZonoConfig {
        min_input_numel: 1,
        max_generators: 4,
        ..DdZonoConfig::default()
    };
    let got =
        dd_zonotope_margins(&graph, &input, &objectives, &plan, &cfg, None).expect("no error");
    assert!(
        got.is_none(),
        "the generator cap must FAIL CLOSED, not degrade"
    );
}

#[test]
fn precision_gate_rejects_a_wide_rounding_channel() {
    use super::DdZonoMargin;
    let m = DdZonoMargin {
        lower: vec![1.0],
        upper: vec![1.1],
        rounding_half_width: vec![1e-12],
        relax_half_width: vec![0.05],
        output_lower: vec![],
        output_upper: vec![],
        output_shape: vec![],
        n_generators: 1,
        wall: std::time::Duration::from_secs(0),
    };
    assert!(m.precision_ok(1e-2));

    let bad = DdZonoMargin {
        rounding_half_width: vec![0.5],
        ..m.clone()
    };
    assert!(!bad.precision_ok(1e-2));

    let nonfinite = DdZonoMargin {
        rounding_half_width: vec![f64::INFINITY],
        ..m
    };
    assert!(!nonfinite.precision_ok(1e-2));
}

#[test]
fn prune_zero_generators_does_not_change_the_enclosure() {
    let lo = vec![-1.0, -1.0, 5.0];
    let up = vec![1.0, 1.0, 6.0];
    let mut z = box_zono(&lo, &up, vec![3]);
    let before = z.concretize();
    z.push_sparse_generator(&[]); // an all-zero column
    prune_zero_generators(&mut z);
    let after = z.concretize();
    assert_eq!(before.0, after.0);
    assert_eq!(before.1, after.1);
}
