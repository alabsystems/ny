// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Soundness and dark-by-default tests for `#pwl-lookup`.
//!
//! EXACTNESS DISCIPLINE. Every "exact" value below is computed in plain f64
//! on DYADIC data chosen so that f64 arithmetic is exact: breakpoints are
//! multiples of 2^-6 in [-16, 16], weights multiples of 2^-8 in [-2, 2] (a
//! saturating table's last weight, the negated sum of the others, stays below
//! 2^7), the bias a multiple of 2^-8, and every evaluation point a multiple of
//! 2^-12 in [-32, 32]. Then `t − b` needs <= 18 significant bits, `w·(t − b)`
//! <= 33 and a sum of <= 64 such terms <= 39 — all exact in binary64 (53
//! bits). So `exact_f` IS the
//! exact rational value, and every containment assertion is a comparison
//! against the truth, not against another rounded computation.

use std::collections::HashMap;

use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_tensor::{next_up_f32, BoundedTensor};

use super::{
    check_lookup_shapes, pwl_point_enclosure, pwl_range_f32, tighten_lookup_output, PwlLookupForm,
    PwlLookupTable, PwlTightenOutcome,
};
use crate::layers::{
    Layer, LinearLayer, MulConstantLayer, ReLULayer, ReduceSumLayer, SigmoidLayer, SqueezeLayer,
    SubConstantLayer, UnsqueezeLayer,
};
use crate::network::{GraphNetwork, GraphNode};

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// Deterministic xorshift64* stream (no `rand` dependency in the fixture).
struct TestRng(u64);

impl TestRng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform integer in `[lo, hi]` (inclusive).
    fn int(&mut self, lo: i64, hi: i64) -> i64 {
        let span = (hi - lo + 1) as u64;
        lo + (self.next_u64() % span) as i64
    }

    /// A multiple of `2^-bits` with integer numerator in `[lo, hi]`.
    fn dyadic(&mut self, lo: i64, hi: i64, bits: i32) -> f32 {
        self.int(lo, hi) as f32 * 2f32.powi(-bits)
    }
}

/// The exact value of `bias + Σ_k w_k·max(0, t − b_k)` for dyadic data
/// (see the module docs for why plain f64 is exact here).
fn exact_f(t: f64, thresholds: &[f32], weights: &[f32], bias: f32) -> f64 {
    thresholds
        .iter()
        .zip(weights)
        .fold(f64::from(bias), |acc, (&b, &w)| {
            acc + f64::from(w) * (t - f64::from(b)).max(0.0)
        })
}

/// Magnitude scale for tightness tolerances at `t`.
fn scale_at(t: f64, thresholds: &[f32], weights: &[f32], bias: f32) -> f64 {
    thresholds
        .iter()
        .zip(weights)
        .map(|(&b, &w)| f64::from(w).abs() * (t.abs() + f64::from(b).abs()))
        .sum::<f64>()
        + f64::from(bias).abs()
        + 1.0
}

fn ulp_f32(x: f64) -> f64 {
    let a = (x as f32).abs();
    f64::from(next_up_f32(a) - a)
}

/// A random dyadic table: K breakpoints (multiples of 2^-6 in [-16, 16]) and
/// weights (multiples of 2^-8 in [-2, 2]). With `saturating`, the last weight
/// is set so that `Σ w_k = 0`: the table is flat beyond its last breakpoint,
/// like the ml4acopf sigmoid (whose tail slope is 4.4e-11).
fn random_table(rng: &mut TestRng, k: usize, saturating: bool) -> (Vec<f32>, Vec<f32>, f32) {
    let mut thresholds: Vec<f32> = (0..k).map(|_| rng.dyadic(-1024, 1024, 6)).collect();
    thresholds.sort_by(f32::total_cmp);
    let mut weights: Vec<f32> = (0..k).map(|_| rng.dyadic(-512, 512, 8)).collect();
    if saturating && k >= 2 {
        let head: f32 = weights[..k - 1].iter().sum();
        // `head` is a multiple of 2^-8 of magnitude < 2^11: exact in f32.
        weights[k - 1] = -head;
    }
    let bias = rng.dyadic(-256, 256, 8);
    (thresholds, weights, bias)
}

fn node(name: &str, layer: Layer, inputs: &[&str]) -> GraphNode {
    GraphNode::new(
        name,
        layer,
        inputs.iter().map(|input| (*input).to_string()).collect(),
    )
}

fn vector_constant(values: &[f32]) -> ArrayD<f32> {
    ArrayD::from_shape_vec(IxDyn(&[values.len()]), values.to_vec()).expect("1-D constant")
}

/// Which lookup spelling a fixture graph uses.
#[derive(Clone, Copy)]
enum Spelling {
    /// `Unsqueeze → SubConstant → ReLU → Linear(1×K) → Squeeze(-1)`.
    MatMul,
    /// `Unsqueeze → SubConstant → ReLU → MulConstant(w) → ReduceSum(-1)`.
    ReduceSum,
}

/// The table end of a fixture graph.
enum Tail {
    /// `→ Linear(weight, bias) [named "table"] → Squeeze(-1) [named "out"]`.
    Row(Array2<f32>, Option<Array1<f32>>),
    /// `→ MulConstant(w) [named "scale"] → ReduceSum(axes, keepdims) ["out"]`.
    Reduce(ArrayD<f32>, Vec<i64>, bool),
}

/// `x[n] → Linear(I) ["z"] → Unsqueeze(-1) ["col"] → <shift> ["shift"] →
/// <hinge> ["hinge"] → <tail>`. Every part is a parameter so the decline
/// tests can swap exactly one of them.
fn build_graph(n: usize, shift: Layer, hinge: Layer, tail: Tail) -> GraphNetwork {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "z",
        Layer::Linear(LinearLayer::new(Array2::eye(n), None).expect("identity linear")),
    ));
    graph.add_node(node(
        "col",
        Layer::Unsqueeze(UnsqueezeLayer::new(-1)),
        &["z"],
    ));
    graph.add_node(node("shift", shift, &["col"]));
    graph.add_node(node("hinge", hinge, &["shift"]));
    match tail {
        Tail::Row(weight, bias) => {
            graph.add_node(node(
                "table",
                Layer::Linear(LinearLayer::new(weight, bias).expect("table linear")),
                &["hinge"],
            ));
            graph.add_node(node(
                "out",
                Layer::Squeeze(SqueezeLayer::new(-1)),
                &["table"],
            ));
        }
        Tail::Reduce(weights, axes, keepdims) => {
            graph.add_node(node(
                "scale",
                Layer::MulConstant(MulConstantLayer::new(weights)),
                &["hinge"],
            ));
            graph.add_node(node(
                "out",
                Layer::ReduceSum(ReduceSumLayer::new(axes, keepdims)),
                &["scale"],
            ));
        }
    }
    graph.set_output("out");
    graph
}

fn row_tail(weights: &[f32], bias: f32) -> Tail {
    let w = Array2::from_shape_vec((1, weights.len()), weights.to_vec()).expect("w row");
    Tail::Row(w, (bias != 0.0).then(|| Array1::from_vec(vec![bias])))
}

fn shift_layer(thresholds: &[f32]) -> Layer {
    Layer::SubConstant(SubConstantLayer::new(vector_constant(thresholds)))
}

/// The authored lookup table in either spelling. The identity Linear makes
/// the scalar enclosure an (outward-rounded) copy of the input box, so every
/// output element is exactly `f(x_i)`.
fn lookup_graph(
    n: usize,
    thresholds: &[f32],
    weights: &[f32],
    bias: f32,
    spelling: Spelling,
) -> GraphNetwork {
    let tail = match spelling {
        Spelling::MatMul => row_tail(weights, bias),
        Spelling::ReduceSum => {
            assert_eq!(bias, 0.0, "the ReduceSum spelling carries no bias");
            Tail::Reduce(vector_constant(weights), vec![-1], false)
        }
    };
    build_graph(n, shift_layer(thresholds), Layer::ReLU(ReLULayer), tail)
}

fn input_box(lower: &[f32], upper: &[f32]) -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[lower.len()]), lower.to_vec()).expect("lower"),
        ArrayD::from_shape_vec(IxDyn(&[upper.len()]), upper.to_vec()).expect("upper"),
    )
    .expect("valid input box")
}

fn output_of(graph: &GraphNetwork, input: &BoundedTensor) -> BoundedTensor {
    graph
        .collect_node_bounds(input)
        .expect("node-bound collection")
        .remove(graph.output_name())
        .expect("output node bounds")
}

fn bits(tensor: &BoundedTensor) -> Vec<(u32, u32)> {
    tensor
        .lower()
        .iter()
        .zip(tensor.upper().iter())
        .map(|(l, u)| (l.to_bits(), u.to_bits()))
        .collect()
}

fn table(thresholds: &[f32], weights: &[f32], bias: f32, form: PwlLookupForm) -> PwlLookupTable {
    PwlLookupTable {
        scalar_input: "col".to_string(),
        sub_node: "shift".to_string(),
        mul_node: matches!(form, PwlLookupForm::ReduceSumMul { .. }).then(|| "scale".to_string()),
        thresholds: thresholds.to_vec(),
        weights: weights.to_vec(),
        bias,
        form,
    }
}

// ---------------------------------------------------------------------------
// Scalar kernel: exactness and containment
// ---------------------------------------------------------------------------

/// Exact rational comparison at breakpoints, endpoints and degenerate
/// intervals: every point enclosure contains the exact value, and every range
/// contains the exact extremes (attained at a candidate) while being no looser
/// than 2 f32 ULPs plus the f64 per-operation widening.
#[test]
fn pwl_lookup_range_exact_at_breakpoints() {
    let mut rng = TestRng::new(0x5057_4C31);
    for case in 0..400 {
        let k = 1 + (case % 60);
        let (thresholds, weights, bias) = random_table(&mut rng, k, case % 3 == 0);

        // Point enclosures at every breakpoint and a few off-grid points.
        let mut points: Vec<f64> = thresholds.iter().map(|&b| f64::from(b)).collect();
        for _ in 0..8 {
            points.push(f64::from(rng.dyadic(-131_072, 131_072, 12)));
        }
        for &t in &points {
            let exact = exact_f(t, &thresholds, &weights, bias);
            let (lo, hi) = pwl_point_enclosure(t, &thresholds, &weights, bias);
            assert!(
                lo <= exact && exact <= hi,
                "case {case}: point enclosure [{lo:e}, {hi:e}] misses exact f({t}) = {exact:e}"
            );
            let slack = 1e-12 * scale_at(t, &thresholds, &weights, bias);
            assert!(
                hi - lo <= slack,
                "case {case}: point enclosure at {t} is {:e} wide (> {slack:e})",
                hi - lo
            );
        }

        // Ranges over random dyadic intervals (including degenerate ones and
        // intervals that start or end exactly ON a breakpoint).
        for sub in 0..6 {
            let (l, u) = match sub {
                0 => {
                    let b = thresholds[rng.int(0, k as i64 - 1) as usize];
                    (b, b)
                }
                1 => {
                    let b = thresholds[rng.int(0, k as i64 - 1) as usize];
                    (b, b + rng.dyadic(0, 8192, 12))
                }
                _ => {
                    let a = rng.dyadic(-131_072, 131_072, 12);
                    let c = rng.dyadic(-131_072, 131_072, 12);
                    (a.min(c), a.max(c))
                }
            };
            let (lf, uf) = (f64::from(l), f64::from(u));
            let mut candidates = vec![lf, uf];
            candidates.extend(
                thresholds
                    .iter()
                    .map(|&b| f64::from(b))
                    .filter(|&b| b > lf && b < uf),
            );
            let values: Vec<f64> = candidates
                .iter()
                .map(|&t| exact_f(t, &thresholds, &weights, bias))
                .collect();
            let exact_min = values.iter().copied().fold(f64::INFINITY, f64::min);
            let exact_max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let (lo, hi) = pwl_range_f32(l, u, &thresholds, &weights, bias)
                .unwrap_or_else(|| panic!("case {case}: finite interval must yield a range"));
            let (lo, hi) = (f64::from(lo), f64::from(hi));
            assert!(
                lo <= exact_min && exact_max <= hi,
                "case {case}: range [{lo:e}, {hi:e}] misses the exact range \
                 [{exact_min:e}, {exact_max:e}] over [{l}, {u}]"
            );
            let slack = 2.0 * ulp_f32(exact_min.abs().max(exact_max.abs()))
                + 1e-12 * scale_at(lf.abs().max(uf.abs()), &thresholds, &weights, bias);
            assert!(
                exact_min - lo <= slack && hi - exact_max <= slack,
                "case {case}: range [{lo:e}, {hi:e}] is looser than the exact \
                 [{exact_min:e}, {exact_max:e}] by more than {slack:e}"
            );
        }
    }
}

/// 1e5 samples per random interval, with 0, 1, 2 and 3 interior breakpoints
/// and saturating tails on both sides: every exact sample lies inside the
/// published f32 range, and the range is no wider than the sampled hull by
/// more than the certified rounding slack.
#[test]
fn pwl_lookup_range_contains_dense_samples() {
    const SAMPLES: usize = 100_000;
    let mut rng = TestRng::new(0x5057_4C32);
    let mut intervals_checked = 0usize;
    for case in 0..6 {
        let k = [4usize, 7, 12, 18, 33, 54][case];
        let (thresholds, weights, bias) = random_table(&mut rng, k, true);
        let lo_b = thresholds[0];
        let hi_b = thresholds[k - 1];
        let mut intervals = Vec::new();
        // Exactly m interior breakpoints (thresholds are sorted; pick a gap
        // start and extend m breakpoints to the right, stopping strictly
        // between neighbours). Duplicated breakpoints only add candidates.
        for m in 0..=3usize {
            if m + 1 >= k {
                continue;
            }
            let start = rng.int(0, (k - m - 2) as i64) as usize;
            let left = f64::from(thresholds[start]);
            let right = f64::from(thresholds[start + m + 1]);
            let l = (left + (right - left) * 0.25) as f32;
            let u = (right - (right - left) * 0.25) as f32;
            if l < u {
                intervals.push((l, u));
            }
        }
        // Saturating tails: wholly left of, wholly right of, and across the
        // whole table.
        intervals.push((lo_b - 8.0, lo_b - 0.5));
        intervals.push((hi_b + 0.5, hi_b + 8.0));
        intervals.push((lo_b - 4.0, hi_b + 4.0));
        for &(l, u) in &intervals {
            // Snap to the 2^-12 grid so samples and endpoints are exact.
            let snap = |v: f32| (f64::from(v) * 4096.0).round() / 4096.0;
            let (lf, uf) = (snap(l), snap(u));
            if lf >= uf {
                continue;
            }
            let (l, u) = (lf as f32, uf as f32);
            let (range_lo, range_hi) = pwl_range_f32(l, u, &thresholds, &weights, bias)
                .expect("finite interval must yield a range");
            let (range_lo, range_hi) = (f64::from(range_lo), f64::from(range_hi));
            let steps = ((uf - lf) * 4096.0) as i64;
            let mut sample_min = f64::INFINITY;
            let mut sample_max = f64::NEG_INFINITY;
            let mut check = |t: f64| {
                let value = exact_f(t, &thresholds, &weights, bias);
                assert!(
                    range_lo <= value && value <= range_hi,
                    "case {case}: exact f({t}) = {value:e} escapes [{range_lo:e}, {range_hi:e}]"
                );
                sample_min = sample_min.min(value);
                sample_max = sample_max.max(value);
            };
            for _ in 0..SAMPLES {
                check(lf + rng.int(0, steps) as f64 / 4096.0);
            }
            // The candidates themselves (the range is ATTAINED there).
            check(lf);
            check(uf);
            for &b in &thresholds {
                let b = f64::from(b);
                if b > lf && b < uf {
                    check(b);
                }
            }
            let slack = 4.0 * ulp_f32(sample_min.abs().max(sample_max.abs()))
                + 1e-12 * scale_at(lf.abs().max(uf.abs()), &thresholds, &weights, bias);
            assert!(
                sample_min - range_lo <= slack && range_hi - sample_max <= slack,
                "case {case}: range [{range_lo:e}, {range_hi:e}] is not the attained hull \
                 [{sample_min:e}, {sample_max:e}]"
            );
            intervals_checked += 1;
        }
    }
    assert!(
        intervals_checked >= 30,
        "too few intervals were checked ({intervals_checked})"
    );
}

// ---------------------------------------------------------------------------
// Graph level: engagement, containment through the IBP entry points
// ---------------------------------------------------------------------------

/// Armed, BOTH IBP entry points (the node-bound collector and the fast-path
/// forward) publish an output box that (a) contains the exact value at dense
/// samples, (b) lies inside the unarmed box, and (c) is strictly tighter on at
/// least one element — for both authored spellings.
#[test]
fn pwl_lookup_graph_ibp_is_contained_shrink_only_and_engaged() {
    let mut rng = TestRng::new(0x5057_4C33);
    for spelling in [Spelling::MatMul, Spelling::ReduceSum] {
        let (thresholds, weights, bias) = random_table(&mut rng, 24, true);
        let bias = match spelling {
            Spelling::MatMul => bias,
            Spelling::ReduceSum => 0.0,
        };
        let n = 6;
        let span = f64::from(thresholds[23] - thresholds[0]).max(1.0);
        let lower: Vec<f32> = (0..n)
            .map(|i| thresholds[0] - 2.0 + (i as f32) * (span as f32) / 8.0)
            .map(|v| ((f64::from(v) * 4096.0).round() / 4096.0) as f32)
            .collect();
        let upper: Vec<f32> = lower.iter().map(|&l| l + (span as f32) / 2.0).collect();
        let upper: Vec<f32> = upper
            .iter()
            .map(|&v| ((f64::from(v) * 4096.0).round() / 4096.0) as f32)
            .collect();
        let input = input_box(&lower, &upper);

        let mut graph = lookup_graph(n, &thresholds, &weights, bias, spelling);
        assert_eq!(
            graph.pwl_lookup_census().len(),
            1,
            "one table in the fixture"
        );
        let plain_collect = output_of(&graph, &input);
        let plain_forward = graph.propagate_ibp(&input).expect("plain forward");
        graph.set_pwl_lookup_exact_range(true);
        let armed_collect = output_of(&graph, &input);
        let armed_forward = graph.propagate_ibp(&input).expect("armed forward");

        for (armed, plain, route) in [
            (&armed_collect, &plain_collect, "collect_node_bounds"),
            (&armed_forward, &plain_forward, "propagate_ibp"),
        ] {
            let mut strictly_tighter = 0usize;
            for i in 0..n {
                let (al, au) = (armed.lower()[[i]], armed.upper()[[i]]);
                let (pl, pu) = (plain.lower()[[i]], plain.upper()[[i]]);
                assert!(
                    pl <= al && au <= pu,
                    "{route}: armed [{al}, {au}] is not inside plain [{pl}, {pu}] at {i}"
                );
                if pl < al || au < pu {
                    strictly_tighter += 1;
                }
                let (lf, uf) = (f64::from(lower[i]), f64::from(upper[i]));
                let steps = ((uf - lf) * 4096.0) as i64;
                for _ in 0..2_000 {
                    let t = lf + rng.int(0, steps) as f64 / 4096.0;
                    let value = exact_f(t, &thresholds, &weights, bias);
                    assert!(
                        f64::from(al) <= value && value <= f64::from(au),
                        "{route}: exact f({t}) = {value:e} escapes armed [{al}, {au}] at {i}"
                    );
                }
            }
            assert!(
                strictly_tighter > 0,
                "{route}: the armed lookup range never tightened a hinge-by-hinge box"
            );
        }
    }
}

/// Every structural deviation declines: the structural matcher returns `None`
/// (or the runtime layout check refuses), and the ARMED output is
/// bit-identical to the unarmed one.
#[test]
fn pwl_lookup_declines_on_shape_mismatch() {
    let thresholds = [-1.0f32, 0.0, 1.0];
    let weights = [0.5f32, 0.5, -1.0];
    let input = input_box(&[-3.0, -2.0, 0.25], &[3.0, 0.5, 2.0]);

    let assert_declined = |mut graph: GraphNetwork, input: &BoundedTensor, label: &str| {
        let plain = output_of(&graph, input);
        let plain_forward = graph.propagate_ibp(input).expect("plain forward");
        graph.set_pwl_lookup_exact_range(true);
        assert_eq!(
            bits(&output_of(&graph, input)),
            bits(&plain),
            "{label}: an armed decline must publish the plain box byte-for-byte"
        );
        assert_eq!(
            bits(&graph.propagate_ibp(input).expect("armed forward")),
            bits(&plain_forward),
            "{label}: an armed decline must publish the plain forward byte-for-byte"
        );
    };

    // (a) reversed Sub: b − x is a different table.
    let graph = build_graph(
        3,
        Layer::SubConstant(SubConstantLayer::new_reverse(vector_constant(&thresholds))),
        Layer::ReLU(ReLULayer),
        row_tail(&weights, 0.0),
    );
    assert!(graph.pwl_lookup_census().is_empty(), "reversed Sub");
    assert_declined(graph, &input, "reversed Sub");

    // (b) a weight that is not a single row (two outputs per scalar).
    // Structural decline only: this graph's trailing Squeeze cannot run.
    let graph = build_graph(
        3,
        shift_layer(&thresholds),
        Layer::ReLU(ReLULayer),
        Tail::Row(
            Array2::from_shape_vec((2, 3), vec![0.5, 0.5, -1.0, 1.0, 0.0, 0.0]).expect("rows"),
            None,
        ),
    );
    assert!(graph.pwl_lookup_census().is_empty(), "two-row weight");

    // (c) breakpoints NOT on the last axis (shape [K, 1]): elementwise, not a
    // lookup. Structural decline only.
    let graph = build_graph(
        3,
        Layer::SubConstant(SubConstantLayer::new(
            ArrayD::from_shape_vec(IxDyn(&[3, 1]), thresholds.to_vec()).expect("column"),
        )),
        Layer::ReLU(ReLULayer),
        row_tail(&weights, 0.0),
    );
    assert!(graph.pwl_lookup_census().is_empty(), "column thresholds");

    // (d) K mismatch between breakpoints and weights. Structural decline only.
    let graph = build_graph(
        3,
        shift_layer(&[-1.0, 0.0, 1.0, 2.0]),
        Layer::ReLU(ReLULayer),
        row_tail(&weights, 0.0),
    );
    assert!(graph.pwl_lookup_census().is_empty(), "K mismatch");

    // (e) a non-ReLU activation between the Sub and the table.
    let graph = build_graph(
        3,
        shift_layer(&thresholds),
        Layer::Sigmoid(SigmoidLayer::new()),
        row_tail(&weights, 0.0),
    );
    assert!(graph.pwl_lookup_census().is_empty(), "sigmoid hinge");
    assert_declined(graph, &input, "sigmoid hinge");

    // (f) a ReduceSum over a non-last axis (and the positive spelling of an
    // axis, which is ambiguous between ONNX and internal conventions).
    for axes in [vec![0i64], vec![1]] {
        let graph = build_graph(
            3,
            shift_layer(&thresholds),
            Layer::ReLU(ReLULayer),
            Tail::Reduce(vector_constant(&weights), axes.clone(), false),
        );
        assert!(
            graph.pwl_lookup_census().is_empty(),
            "ReduceSum axes {axes:?}"
        );
        assert_declined(graph, &input, "ReduceSum non-negative axis");
    }

    // (f') a scalar MulConstant (K=1 broadcast weight) is not a K-row table.
    let graph = build_graph(
        3,
        shift_layer(&thresholds),
        Layer::ReLU(ReLULayer),
        Tail::Reduce(ArrayD::from_elem(IxDyn(&[]), 0.5), vec![-1], false),
    );
    assert!(graph.pwl_lookup_census().is_empty(), "scalar MulConstant");
    assert_declined(graph, &input, "scalar MulConstant");

    // (g) structurally a table, but the scalar input's last axis is NOT 1:
    // `x[3] − b[3]` is an elementwise Sub, not a broadcast lookup. The
    // runtime layout check must refuse it.
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "z",
        Layer::Linear(LinearLayer::new(Array2::eye(3), None).expect("identity")),
    ));
    graph.add_node(node(
        "shift",
        Layer::SubConstant(SubConstantLayer::new(vector_constant(&thresholds))),
        &["z"],
    ));
    graph.add_node(node("hinge", Layer::ReLU(ReLULayer), &["shift"]));
    graph.add_node(node(
        "out",
        Layer::Linear(
            LinearLayer::new(
                Array2::from_shape_vec((1, 3), weights.to_vec()).expect("row"),
                None,
            )
            .expect("row linear"),
        ),
        &["hinge"],
    ));
    graph.set_output("out");
    assert_eq!(
        graph.pwl_lookup_census().len(),
        1,
        "the elementwise spelling matches STRUCTURALLY; only the runtime check refuses it"
    );
    assert_declined(graph, &input, "elementwise Sub");

    // The pure layout check, on the shapes the fixtures produce.
    let row = table(&thresholds, &weights, 0.0, PwlLookupForm::LinearRow);
    assert!(check_lookup_shapes(&row, &[4, 1], &[4, 3], None, &[4, 1]).is_ok());
    assert!(check_lookup_shapes(&row, &[3], &[3], None, &[1]).is_err());
    assert!(check_lookup_shapes(&row, &[4, 1], &[4, 5], None, &[4, 1]).is_err());
    assert!(check_lookup_shapes(&row, &[4, 1], &[2, 3], None, &[4, 1]).is_err());
    assert!(check_lookup_shapes(&row, &[4, 1], &[4, 3], None, &[4]).is_err());
    assert!(check_lookup_shapes(&row, &[], &[3], None, &[1]).is_err());
    let sum = table(
        &thresholds,
        &weights,
        0.0,
        PwlLookupForm::ReduceSumMul { keepdims: false },
    );
    assert!(check_lookup_shapes(&sum, &[4, 1], &[4, 3], Some(&[4, 3][..]), &[4]).is_ok());
    assert!(check_lookup_shapes(&sum, &[4, 1], &[4, 3], Some(&[2, 4, 3][..]), &[4]).is_err());
    assert!(check_lookup_shapes(&sum, &[4, 1], &[4, 3], Some(&[4, 3][..]), &[4, 1]).is_err());
    let kept = table(
        &thresholds,
        &weights,
        0.0,
        PwlLookupForm::ReduceSumMul { keepdims: true },
    );
    assert!(check_lookup_shapes(&kept, &[4, 1], &[4, 3], Some(&[4, 3][..]), &[4, 1]).is_ok());
}

/// A plain box DISJOINT from the exact range (impossible for two sound
/// enclosures, so it can only mean a defect) is reported as a contradiction
/// by the kernel, and the graph hook then publishes the plain box untouched —
/// never an empty or "repaired" one.
#[test]
fn pwl_lookup_contradiction_keeps_plain_box() {
    let thresholds = [-1.0f32, 0.0, 1.0];
    let weights = [0.5f32, 0.5, -1.0];
    // Exact range of f over [-3, 3] is [0, 1.5]; claim [5, 6] instead.
    let scalar = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![-3.0]).expect("l"),
        ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![3.0]).expect("u"),
    )
    .expect("scalar");
    let bogus = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![5.0]).expect("l"),
        ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![6.0]).expect("u"),
    )
    .expect("bogus plain box");
    let row = table(&thresholds, &weights, 0.0, PwlLookupForm::LinearRow);
    assert!(matches!(
        tighten_lookup_output(&row, &scalar, &bogus, None),
        PwlTightenOutcome::Contradiction { element: 0, .. }
    ));

    let mut graph = lookup_graph(1, &thresholds, &weights, 0.0, Spelling::MatMul);
    graph.set_pwl_lookup_exact_range(true);
    let sub = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1, 3]), vec![-2.0, -3.0, -4.0]).expect("l"),
        ArrayD::from_shape_vec(IxDyn(&[1, 3]), vec![4.0, 3.0, 2.0]).expect("u"),
    )
    .expect("sub");
    let cache: HashMap<String, BoundedTensor> =
        [("col".to_string(), scalar), ("shift".to_string(), sub)]
            .into_iter()
            .collect();
    let table_node = graph.node("table").expect("table node").clone();
    let network_input = input_box(&[-3.0], &[3.0]);
    let published = graph.pwl_lookup_tighten_node(
        "table",
        &table_node,
        bogus.clone(),
        &network_input,
        &cache,
        None,
    );
    assert_eq!(
        bits(&published),
        bits(&bogus),
        "a contradiction must publish the plain box byte-for-byte"
    );
}

/// An expired deadline declines (the plain box is always a valid answer).
#[test]
fn pwl_lookup_expired_deadline_declines() {
    let thresholds = [-1.0f32, 0.0, 1.0];
    let weights = [0.5f32, 0.5, -1.0];
    let scalar = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![-3.0]).expect("l"),
        ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![3.0]).expect("u"),
    )
    .expect("scalar");
    let plain = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![-2.0]).expect("l"),
        ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![3.5]).expect("u"),
    )
    .expect("plain");
    let row = table(&thresholds, &weights, 0.0, PwlLookupForm::LinearRow);
    let expired = std::time::Instant::now()
        .checked_sub(std::time::Duration::from_secs(1))
        .expect("an instant one second ago");
    assert!(matches!(
        tighten_lookup_output(&row, &scalar, &plain, Some(expired)),
        PwlTightenOutcome::Declined("deadline")
    ));
    match tighten_lookup_output(&row, &scalar, &plain, None) {
        PwlTightenOutcome::Applied {
            bounds,
            elems,
            tightened,
        } => {
            assert_eq!((elems, tightened), (1, 1));
            assert!(bounds.lower()[[0, 0]] <= 0.0 && bounds.lower()[[0, 0]] > -1e-6);
            assert!(bounds.upper()[[0, 0]] >= 1.5 && bounds.upper()[[0, 0]] < 1.5 + 1e-6);
        }
        other => panic!("an unexpired kernel must apply, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Dark by default
// ---------------------------------------------------------------------------

/// Unarmed is the default at every layer (graph, config, serde), unarmed
/// output is byte-identical before and after an arm/disarm cycle, and the
/// flag is the ONLY switch (arming changes the box on the fixture).
#[test]
fn pwl_lookup_dark_by_default_is_bit_identical() {
    assert!(!GraphNetwork::new().pwl_lookup_exact_range());
    assert!(!crate::BetaCrownConfig::default().pwl_lookup_exact_range);
    let absent: crate::BetaCrownConfig =
        serde_json::from_str("{}").expect("empty config deserializes");
    assert!(!absent.pwl_lookup_exact_range);
    let armed: crate::BetaCrownConfig =
        serde_json::from_str(r#"{"pwl_lookup_exact_range": true}"#).expect("typed bool");
    assert!(armed.pwl_lookup_exact_range);

    let thresholds = [-1.0f32, 0.0, 1.0];
    let weights = [0.5f32, 0.5, -1.0];
    let input = input_box(&[-3.0, -2.0, 0.25], &[3.0, 0.5, 2.0]);
    let mut graph = lookup_graph(3, &thresholds, &weights, 0.0, Spelling::MatMul);
    let unarmed_collect = output_of(&graph, &input);
    let unarmed_forward = graph.propagate_ibp(&input).expect("forward");
    let unarmed_clone = graph.clone();

    graph.set_pwl_lookup_exact_range(true);
    assert!(graph.clone().pwl_lookup_exact_range(), "clones inherit");
    let armed_collect = output_of(&graph, &input);
    assert_ne!(
        bits(&armed_collect),
        bits(&unarmed_collect),
        "arming must change the fixture's box (the flag is a live switch)"
    );

    graph.set_pwl_lookup_exact_range(false);
    assert_eq!(bits(&output_of(&graph, &input)), bits(&unarmed_collect));
    assert_eq!(
        bits(&graph.propagate_ibp(&input).expect("forward")),
        bits(&unarmed_forward)
    );
    assert_eq!(
        bits(&output_of(&unarmed_clone, &input)),
        bits(&unarmed_collect)
    );
}

/// The census names each table with its hinge count, independent of arming.
#[test]
fn pwl_lookup_census_counts_tables() {
    let graph = lookup_graph(
        2,
        &[-1.0, 0.0, 1.0],
        &[0.5, 0.5, -1.0],
        0.25,
        Spelling::MatMul,
    );
    assert_eq!(graph.pwl_lookup_census(), vec![("table".to_string(), 3)]);
    let graph = lookup_graph(
        2,
        &[-1.0, 0.0, 1.0],
        &[0.5, 0.5, -1.0],
        0.0,
        Spelling::ReduceSum,
    );
    assert_eq!(graph.pwl_lookup_census(), vec![("out".to_string(), 3)]);
    let table = graph
        .match_pwl_lookup(graph.node("out").expect("out"))
        .expect("ReduceSum spelling matches");
    assert_eq!(table.mul_node.as_deref(), Some("scale"));
    assert_eq!(table.form, PwlLookupForm::ReduceSumMul { keepdims: false });
}
