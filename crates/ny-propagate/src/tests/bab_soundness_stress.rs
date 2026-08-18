// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! End-to-end BaB soundness stress test.
//!
//! Goal: be confident the verifier NEVER emits a wrong "verified/unsat" verdict
//! on small residual / convolutional / ReLU networks. A single wrong verdict
//! loses VNN-COMP, so this is the gold-standard check: for SMALL networks where
//! the exact answer is computable by dense sampling of the input box, confirm
//! the verifier never claims "Verified" when the property is actually violated.
//!
//! Pipeline exercised: CROWN root bounds + ReLU-split BaB + β-CROWN +
//! constrained child propagation through residual `Add` on conv/linear nets.
//! Entry point: [`BetaCrownVerifier::verify_graph_relu_split`], which proves the
//! property `objective · f(x) >= threshold` over the input box.
//!
//! ## Soundness invariant
//!
//! `verify_graph_relu_split(graph, input, objective, threshold)` proves a
//! LOWER bound on the margin `m(x) = objective · f(x)`: a `Verified` verdict
//! asserts `m(x) >= threshold` for ALL `x` in the input box.
//!
//! - If the verifier returns `Verified`, then the TRUE sampled minimum margin
//!   `min_x m(x)` (over thousands of sampled inputs) MUST be `>= threshold`
//!   (modulo a tiny float slack). If any sample has `m(x) < threshold` while the
//!   verifier said `Verified` → CRITICAL UNSOUNDNESS.
//! - If the verifier returns `Violated`, the returned counterexample is
//!   re-evaluated through the network and must genuinely have `m(cex) < threshold`.
//! - `Unknown` / `Timeout` / `PotentialViolation` are always sound.
//!
//! ## Exact forward evaluation
//!
//! All generated networks are piecewise-linear (Linear / Conv2d / ReLU / Add).
//! IBP on a DEGENERATE (point) input box `[x, x]` therefore returns the EXACT
//! network output `f(x)` with no relaxation — there is nothing to over-
//! approximate when lower == upper through affine + ReLU + Add. We use
//! `graph.propagate_ibp(&point_box)` as the trusted oracle.

use crate::beta_crown::nonlinear_branching::NonlinearBranchingConfig;
use crate::{
    BabVerificationStatus, BetaCrownConfig, BetaCrownVerifier, BranchingHeuristic, Conv2dLayer,
    FlattenLayer, GraphNetwork, GraphNode, Layer, LinearLayer, ReLULayer,
};
use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;
use std::time::Duration;

// ---------------------------------------------------------------------------
// Deterministic RNG (SplitMix64 → f32) so the suite is fully reproducible.
// ---------------------------------------------------------------------------

struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        // Avoid the all-zero state (SplitMix64 still works but mix the seed).
        Self {
            state: seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1),
        }
    }

    fn next_u64(&mut self) -> u64 {
        // SplitMix64.
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform f32 in [0, 1).
    fn unit(&mut self) -> f32 {
        // Use the top 24 bits for a clean f32 mantissa.
        ((self.next_u64() >> 40) as f32) / ((1u32 << 24) as f32)
    }

    /// Uniform f32 in [lo, hi].
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.unit()
    }

    /// Small signed weight, biased toward modest magnitudes to keep bounds finite.
    fn weight(&mut self) -> f32 {
        self.range(-1.2, 1.2)
    }

    fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next_u64() % (n as u64)) as usize
        }
    }
}

// ---------------------------------------------------------------------------
// Random layer builders.
// ---------------------------------------------------------------------------

fn random_linear(rng: &mut Rng, in_dim: usize, out_dim: usize) -> LinearLayer {
    let mut w = Array2::<f32>::zeros((out_dim, in_dim));
    for v in w.iter_mut() {
        *v = rng.weight();
    }
    let mut b = Array1::<f32>::zeros(out_dim);
    for v in b.iter_mut() {
        *v = rng.range(-0.5, 0.5);
    }
    LinearLayer::new(w, Some(b)).expect("valid random linear layer")
}

fn random_conv(
    rng: &mut Rng,
    in_c: usize,
    out_c: usize,
    kh: usize,
    kw: usize,
    in_h: usize,
    in_w: usize,
) -> Conv2dLayer {
    let n = out_c * in_c * kh * kw;
    let mut data = Vec::with_capacity(n);
    for _ in 0..n {
        data.push(rng.range(-0.8, 0.8));
    }
    let kernel = ArrayD::from_shape_vec(IxDyn(&[out_c, in_c, kh, kw]), data).unwrap();
    let mut bias = Array1::<f32>::zeros(out_c);
    for v in bias.iter_mut() {
        *v = rng.range(-0.4, 0.4);
    }
    Conv2dLayer::with_input_shape(kernel, Some(bias), (1, 1), (0, 0), in_h, in_w)
        .expect("valid random conv layer")
}

// ---------------------------------------------------------------------------
// Random network families. Each returns (graph, input_box, output_dim).
// All outputs are flat 1-D vectors so objectives are simple dot products.
// ---------------------------------------------------------------------------

/// Plain feed-forward MLP: Linear -> ReLU -> ... -> Linear.
fn build_mlp(rng: &mut Rng) -> (GraphNetwork, BoundedTensor, usize) {
    let in_dim = 2 + rng.below(4); // 2..=5
    let n_hidden_layers = 1 + rng.below(3); // 1..=3 ReLU blocks
    let mut graph = GraphNetwork::new();

    let mut prev = "in_linear".to_string();
    let mut dim = in_dim;
    let h = 3 + rng.below(6); // 3..=8 hidden width
    graph.add_node(GraphNode::from_input(
        &prev,
        Layer::Linear(random_linear(rng, dim, h)),
    ));
    dim = h;

    for i in 0..n_hidden_layers {
        let relu = format!("relu{i}");
        graph.add_node(GraphNode::new(
            &relu,
            Layer::ReLU(ReLULayer),
            vec![prev.clone()],
        ));
        let out_w = 3 + rng.below(6);
        let lin = format!("lin{i}");
        graph.add_node(GraphNode::new(
            &lin,
            Layer::Linear(random_linear(rng, dim, out_w)),
            vec![relu.clone()],
        ));
        dim = out_w;
        prev = lin;
    }

    let out_dim = 1 + rng.below(2); // 1..=2 outputs
    graph.add_node(GraphNode::new(
        "out",
        Layer::Linear(random_linear(rng, dim, out_dim)),
        vec![prev],
    ));
    graph.set_output("out");

    let input = random_input_box(rng, &[in_dim]);
    (graph, input, out_dim)
}

/// Residual MLP with a skip connection across a ReLU block, merged via `Add`.
///
/// ```text
/// in -> L0 (dim h)
///        |\
///        | relu -> L1 (dim h)
///        |        /
///        +--> Add (skip + branch)   [residual]
///              |
///             relu2 -> Lout
/// ```
fn build_residual_mlp(rng: &mut Rng) -> (GraphNetwork, BoundedTensor, usize) {
    let in_dim = 2 + rng.below(4); // 2..=5
    let h = 3 + rng.below(5); // 3..=7 (skip & branch must share this width)
    let mut graph = GraphNetwork::new();

    graph.add_node(GraphNode::from_input(
        "l0",
        Layer::Linear(random_linear(rng, in_dim, h)),
    ));
    graph.add_node(GraphNode::new(
        "relu0",
        Layer::ReLU(ReLULayer),
        vec!["l0".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "l1",
        Layer::Linear(random_linear(rng, h, h)),
        vec!["relu0".to_string()],
    ));
    // Residual merge: skip path (l0) + branch (l1). Both have width h.
    graph.add_node(GraphNode::new(
        "add",
        Layer::Add(crate::layers::AddLayer),
        vec!["l0".to_string(), "l1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["add".to_string()],
    ));
    let out_dim = 1 + rng.below(2);
    graph.add_node(GraphNode::new(
        "out",
        Layer::Linear(random_linear(rng, h, out_dim)),
        vec!["relu1".to_string()],
    ));
    graph.set_output("out");

    let input = random_input_box(rng, &[in_dim]);
    (graph, input, out_dim)
}

/// Two-branch residual: input fans out to two independent ReLU branches that
/// are recombined with `Add`, then a final ReLU + Linear. Exercises the
/// constrained child-backward through a DAG `Add` more aggressively.
fn build_two_branch_residual(rng: &mut Rng) -> (GraphNetwork, BoundedTensor, usize) {
    let in_dim = 2 + rng.below(3); // 2..=4
    let h = 3 + rng.below(4); // 3..=6
    let mut graph = GraphNetwork::new();

    graph.add_node(GraphNode::from_input(
        "stem",
        Layer::Linear(random_linear(rng, in_dim, h)),
    ));
    graph.add_node(GraphNode::new(
        "relu_stem",
        Layer::ReLU(ReLULayer),
        vec!["stem".to_string()],
    ));
    // Branch A
    graph.add_node(GraphNode::new(
        "a_lin",
        Layer::Linear(random_linear(rng, h, h)),
        vec!["relu_stem".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "a_relu",
        Layer::ReLU(ReLULayer),
        vec!["a_lin".to_string()],
    ));
    // Branch B
    graph.add_node(GraphNode::new(
        "b_lin",
        Layer::Linear(random_linear(rng, h, h)),
        vec!["relu_stem".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "b_relu",
        Layer::ReLU(ReLULayer),
        vec!["b_lin".to_string()],
    ));
    // Merge A + B
    graph.add_node(GraphNode::new(
        "merge",
        Layer::Add(crate::layers::AddLayer),
        vec!["a_relu".to_string(), "b_relu".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "merge_relu",
        Layer::ReLU(ReLULayer),
        vec!["merge".to_string()],
    ));
    let out_dim = 1 + rng.below(2);
    graph.add_node(GraphNode::new(
        "out",
        Layer::Linear(random_linear(rng, h, out_dim)),
        vec!["merge_relu".to_string()],
    ));
    graph.set_output("out");

    let input = random_input_box(rng, &[in_dim]);
    (graph, input, out_dim)
}

/// Small conv net: Conv2d -> ReLU -> Flatten -> Linear.
fn build_conv(rng: &mut Rng) -> (GraphNetwork, BoundedTensor, usize) {
    let in_c = 1 + rng.below(2); // 1..=2
    let in_h = 3 + rng.below(2); // 3..=4
    let in_w = in_h;
    let out_c = 1 + rng.below(2); // 1..=2
    let kh = 2;
    let kw = 2;
    let out_h = in_h - kh + 1;
    let out_w = in_w - kw + 1;
    let flat = out_c * out_h * out_w;

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "conv",
        Layer::Conv2d(random_conv(rng, in_c, out_c, kh, kw, in_h, in_w)),
    ));
    graph.add_node(GraphNode::new(
        "crelu",
        Layer::ReLU(ReLULayer),
        vec!["conv".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "flatten",
        Layer::Flatten(FlattenLayer::new(0)),
        vec!["crelu".to_string()],
    ));
    let out_dim = 1 + rng.below(2);
    graph.add_node(GraphNode::new(
        "out",
        Layer::Linear(random_linear(rng, flat, out_dim)),
        vec!["flatten".to_string()],
    ));
    graph.set_output("out");

    let input = random_input_box(rng, &[in_c, in_h, in_w]);
    (graph, input, out_dim)
}

/// Residual conv net: Conv -> ReLU -> Conv (1x1, spatial-preserving) merged with
/// a skip via `Add` on the conv feature map, then Flatten -> Linear. This is the
/// recently-changed path: BaB descending through a residual `Add` on a conv net.
fn build_residual_conv(rng: &mut Rng) -> (GraphNetwork, BoundedTensor, usize) {
    let in_c = 1 + rng.below(2); // 1..=2
    let in_h = 3 + rng.below(2); // 3..=4
    let in_w = in_h;
    let mid_c = 1 + rng.below(2); // feature channels for the residual block

    let mut graph = GraphNetwork::new();
    // Stem conv: in_c -> mid_c, 1x1 so spatial size is preserved (so the skip
    // and branch feature maps share a shape and can be Added).
    graph.add_node(GraphNode::from_input(
        "stem",
        Layer::Conv2d(random_conv(rng, in_c, mid_c, 1, 1, in_h, in_w)),
    ));
    graph.add_node(GraphNode::new(
        "stem_relu",
        Layer::ReLU(ReLULayer),
        vec!["stem".to_string()],
    ));
    // Branch conv: mid_c -> mid_c, 1x1 (preserves spatial dims).
    graph.add_node(GraphNode::new(
        "branch",
        Layer::Conv2d(random_conv(rng, mid_c, mid_c, 1, 1, in_h, in_w)),
        vec!["stem_relu".to_string()],
    ));
    // Residual: stem_relu (skip) + branch. Both are (mid_c, in_h, in_w).
    graph.add_node(GraphNode::new(
        "add",
        Layer::Add(crate::layers::AddLayer),
        vec!["stem_relu".to_string(), "branch".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "add_relu",
        Layer::ReLU(ReLULayer),
        vec!["add".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "flatten",
        Layer::Flatten(FlattenLayer::new(0)),
        vec!["add_relu".to_string()],
    ));
    let flat = mid_c * in_h * in_w;
    let out_dim = 1 + rng.below(2);
    graph.add_node(GraphNode::new(
        "out",
        Layer::Linear(random_linear(rng, flat, out_dim)),
        vec!["flatten".to_string()],
    ));
    graph.set_output("out");

    let input = random_input_box(rng, &[in_c, in_h, in_w]);
    (graph, input, out_dim)
}

/// Build a small random input box of the given shape with a modest radius.
fn random_input_box(rng: &mut Rng, shape: &[usize]) -> BoundedTensor {
    let n: usize = shape.iter().product();
    let mut lower = Vec::with_capacity(n);
    let mut upper = Vec::with_capacity(n);
    for _ in 0..n {
        let center = rng.range(-1.0, 1.0);
        let radius = rng.range(0.1, 0.8);
        lower.push(center - radius);
        upper.push(center + radius);
    }
    let lo = ArrayD::from_shape_vec(IxDyn(shape), lower).unwrap();
    let up = ArrayD::from_shape_vec(IxDyn(shape), upper).unwrap();
    BoundedTensor::new(lo, up).expect("valid random input box")
}

// ---------------------------------------------------------------------------
// Exact forward evaluation via point-IBP (exact for piecewise-linear nets).
// ---------------------------------------------------------------------------

/// Evaluate the network exactly at a single concrete point `x`.
///
/// For affine + ReLU + Conv + Add networks, IBP on a degenerate box `[x, x]`
/// returns the exact output (lower == upper). Returns the flat output vector.
fn forward_exact(graph: &GraphNetwork, shape: &[usize], x: &[f32]) -> Vec<f32> {
    let lo = ArrayD::from_shape_vec(IxDyn(shape), x.to_vec()).unwrap();
    let up = lo.clone();
    let point = BoundedTensor::new(lo, up).expect("valid point box");
    let out = graph
        .propagate_ibp(&point)
        .expect("point-IBP forward must succeed");
    let flat = out.flatten();
    let (l, u) = flat.lower_upper();
    // Sanity: degenerate input must give a (numerically) degenerate output.
    for (li, ui) in l.iter().zip(u.iter()) {
        debug_assert!(
            (li - ui).abs() <= 1e-3 * (1.0 + li.abs()),
            "point-IBP produced a non-degenerate output bound: [{li}, {ui}]"
        );
    }
    // Use the midpoint to absorb any tiny float asymmetry.
    l.iter().zip(u.iter()).map(|(a, b)| 0.5 * (a + b)).collect()
}

/// Flatten a `BoundedTensor`'s lower/upper bounds to row-major `Vec<f32>`.
fn flat_bounds(input: &BoundedTensor) -> (Vec<f32>, Vec<f32>) {
    let lo = input.lower().iter().copied().collect();
    let hi = input.upper().iter().copied().collect();
    (lo, hi)
}

/// Dot product `objective · output`.
fn margin(objective: &[f32], output: &[f32]) -> f32 {
    objective
        .iter()
        .zip(output.iter())
        .map(|(a, b)| a * b)
        .sum()
}

/// Densely sample the input box and return the true minimum margin and the
/// input achieving it.
///
/// Strategy: enumerate all 2^d box corners when `d` is small (the extrema of an
/// affine-dominated margin often live at corners), plus the center, plus a
/// large pseudo-random interior sample. This gives a tight estimate of
/// `min_x objective · f(x)`.
fn true_min_margin(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    shape: &[usize],
    objective: &[f32],
    rng: &mut Rng,
    random_samples: usize,
) -> (f32, Vec<f32>) {
    let (flat_lo, flat_hi) = flat_bounds(input);
    let d = flat_lo.len();

    let mut best = f32::INFINITY;
    let mut best_x = flat_lo.clone();

    let consider = |x: &[f32], best: &mut f32, best_x: &mut Vec<f32>| {
        let out = forward_exact(graph, shape, x);
        let m = margin(objective, &out);
        if m < *best {
            *best = m;
            *best_x = x.to_vec();
        }
    };

    // 1. All corners when dimension is small (<= 12 → <= 4096 corners).
    if d <= 12 {
        for mask in 0u32..(1u32 << d) {
            let mut x = vec![0.0_f32; d];
            for (i, xi) in x.iter_mut().enumerate() {
                *xi = if (mask >> i) & 1 == 1 {
                    flat_hi[i]
                } else {
                    flat_lo[i]
                };
            }
            consider(&x, &mut best, &mut best_x);
        }
    }

    // 2. Center (helps when extrema are mid-domain).
    let center: Vec<f32> = flat_lo
        .iter()
        .zip(flat_hi.iter())
        .map(|(a, b)| 0.5 * (a + b))
        .collect();
    consider(&center, &mut best, &mut best_x);

    // 3. Dense pseudo-random interior sampling.
    for _ in 0..random_samples {
        let mut x = vec![0.0_f32; d];
        for (i, xi) in x.iter_mut().enumerate() {
            *xi = rng.range(flat_lo[i], flat_hi[i]);
        }
        consider(&x, &mut best, &mut best_x);
    }

    (best, best_x)
}

// ---------------------------------------------------------------------------
// Verifier configuration matrix.
// ---------------------------------------------------------------------------

fn base_config() -> BetaCrownConfig {
    BetaCrownConfig {
        max_domains: 400,
        max_depth: 30,
        timeout: Duration::from_millis(1_500),
        use_alpha_crown: true,
        batch_size: 8,
        ..Default::default()
    }
}

/// Pick a branching heuristic deterministically from the case index. Biases
/// toward ReLU-splitting heuristics (the recently-changed BaB path).
fn config_for(case: usize) -> BetaCrownConfig {
    let mut cfg = base_config();
    cfg.branching_heuristic = match case % 4 {
        0 => BranchingHeuristic::Kfsb,
        1 => BranchingHeuristic::FilteredSmartBranching,
        2 => BranchingHeuristic::BoundImpact,
        _ => BranchingHeuristic::GenBaB(NonlinearBranchingConfig::default()),
    };
    cfg
}

// ---------------------------------------------------------------------------
// Per-case outcome tracking.
// ---------------------------------------------------------------------------

#[derive(Default, Debug)]
struct Tally {
    verified: usize,
    violated: usize,
    unknown: usize,
    timeout: usize,
    potential: usize,
    zero_budget_refused: usize,
}

/// Run one verification case and assert the soundness invariant. Returns the
/// outcome bucket. Panics with full detail on any unsoundness.
#[allow(clippy::too_many_arguments)]
fn run_case(
    family: &str,
    case_idx: usize,
    graph: &GraphNetwork,
    input: &BoundedTensor,
    shape: &[usize],
    objective: &[f32],
    threshold: f32,
    true_min: f32,
    true_min_x: &[f32],
    tally: &mut Tally,
) {
    let verifier = BetaCrownVerifier::new(config_for(case_idx));
    // A separate fallback regression deliberately sets the process-global
    // dense budget to zero. Serialize each verifier call against that mutation
    // and pin the ordinary budget so this soundness stress cannot inherit its
    // temporary policy. Keep the guard per case so other environment tests are
    // never starved behind this multi-minute stress lane.
    let verification = crate::tests::with_crown_dense_budget_mb("2048", || {
        verifier.verify_graph_relu_split(graph, input, objective, threshold)
    });
    let result = match verification {
        Ok(result) => result,
        // The full parallel suite contains explicit zero-budget gate tests.
        // This stress lane does not mutate process-global policy, so it can
        // briefly observe that override.  A direct zero-budget refusal carries
        // no proof authority and is therefore recorded as a non-verdict, not
        // mistaken for a soundness failure.  Every other verifier error remains
        // fatal, and the suite-level coverage assertions below prevent these
        // refusals from making the oracle vacuous.
        Err(ny_core::NyError::CpuMemoryExceeded {
            budget_bytes: 0, ..
        }) => {
            tally.zero_budget_refused += 1;
            return;
        }
        Err(error) => {
            panic!("[{family} #{case_idx}] verify_graph_relu_split errored: {error:?}")
        }
    };

    // Float slack for the soundness comparison. The verifier proves
    // m(x) >= threshold; sampling is exact (point-IBP), so we only need a small
    // epsilon to absorb f32 accumulation in the dot product / conv.
    let slack = 1e-3_f32 * (1.0 + threshold.abs() + true_min.abs());

    match result.result {
        BabVerificationStatus::Verified => {
            tally.verified += 1;
            // SOUNDNESS INVARIANT: every sampled point must satisfy m(x) >= threshold.
            assert!(
                true_min >= threshold - slack,
                "\n=== CRITICAL UNSOUNDNESS ===\n\
                 family={family} case={case_idx} heuristic={:?}\n\
                 verifier verdict: VERIFIED (claims objective·f(x) >= {threshold} for all x)\n\
                 but a sampled input VIOLATES the property:\n\
                   true min margin   = {true_min}\n\
                   threshold         = {threshold}\n\
                   violating input x = {true_min_x:?}\n\
                   objective         = {objective:?}\n\
                   input box lower   = {:?}\n\
                   input box upper   = {:?}\n\
                   output@x          = {:?}\n\
                 This is a WRONG 'verified' verdict on a violated property.\n",
                config_for(case_idx).branching_heuristic,
                flat_bounds(input).0,
                flat_bounds(input).1,
                forward_exact(graph, shape, true_min_x),
            );
        }
        BabVerificationStatus::Violated {
            ref counterexample, ..
        } => {
            tally.violated += 1;
            // Double-check the counterexample genuinely violates the property.
            assert_eq!(
                counterexample.len(),
                shape.iter().product::<usize>(),
                "[{family} #{case_idx}] counterexample length mismatch"
            );
            // Counterexample must lie within the input box (allow tiny slack).
            let (flat_lo, flat_hi) = flat_bounds(input);
            for (i, &c) in counterexample.iter().enumerate() {
                assert!(
                    c >= flat_lo[i] - 1e-3 && c <= flat_hi[i] + 1e-3,
                    "[{family} #{case_idx}] counterexample[{i}]={c} outside box [{}, {}]",
                    flat_lo[i],
                    flat_hi[i]
                );
            }
            let out = forward_exact(graph, shape, counterexample);
            let m = margin(objective, &out);
            assert!(
                m < threshold + slack,
                "[{family} #{case_idx}] verifier reported VIOLATED but the \
                 counterexample does NOT violate: margin={m} >= threshold={threshold}\n\
                 counterexample={counterexample:?} output={out:?} objective={objective:?}"
            );
        }
        BabVerificationStatus::PotentialViolation { .. } => tally.potential += 1,
        BabVerificationStatus::Unknown { .. } => tally.unknown += 1,
        BabVerificationStatus::Timeout => tally.timeout += 1,
    }
}

// ---------------------------------------------------------------------------
// The stress test driver.
// ---------------------------------------------------------------------------

type Builder = fn(&mut Rng) -> (GraphNetwork, BoundedTensor, usize);

/// Build a network from `builder`, derive an exact true-min margin, then probe
/// the verifier with several thresholds near the true margin (the hardest
/// region for soundness: some verify, some violate, some borderline).
fn stress_family(
    family: &str,
    builder: Builder,
    seed_base: u64,
    num_networks: usize,
    tally: &mut Tally,
    case_counter: &mut usize,
) {
    for net_i in 0..num_networks {
        let mut rng = Rng::new(seed_base.wrapping_add(net_i as u64 * 0x1000));
        let (graph, input, out_dim) = builder(&mut rng);
        let shape: Vec<usize> = input.lower().shape().to_vec();

        // Random objective over the (flat) outputs, components in [-1, 1],
        // guaranteed not all-zero.
        let mut objective = vec![0.0_f32; out_dim];
        loop {
            let mut any = false;
            for o in objective.iter_mut() {
                *o = rng.range(-1.0, 1.0);
                if o.abs() > 1e-2 {
                    any = true;
                }
            }
            if any {
                break;
            }
        }

        // Exact true minimum margin over the input box.
        let (true_min, true_min_x) =
            true_min_margin(&graph, &input, &shape, &objective, &mut rng, 4000);

        // Span estimate to scale threshold offsets: probe a few more random
        // points for a coarse upper margin; fall back to a small floor.
        let span = {
            let mut hi = true_min;
            let (flat_lo, flat_hi) = flat_bounds(&input);
            for _ in 0..200 {
                let x: Vec<f32> = flat_lo
                    .iter()
                    .zip(flat_hi.iter())
                    .map(|(a, b)| rng.range(*a, *b))
                    .collect();
                let m = margin(&objective, &forward_exact(&graph, &shape, &x));
                if m > hi {
                    hi = m;
                }
            }
            (hi - true_min).abs().max(0.05)
        };

        // Probe thresholds straddling the true minimum:
        //   - well below  → should be Verified (sound to verify)
        //   - just below  → borderline-verifiable
        //   - just above  → property genuinely violated near the optimum
        //   - well above  → clearly violated
        let thresholds = [
            true_min - 0.30 * span,
            true_min - 0.02 * span,
            true_min + 0.02 * span,
            true_min + 0.30 * span,
        ];

        for &threshold in &thresholds {
            run_case(
                family,
                *case_counter,
                &graph,
                &input,
                &shape,
                &objective,
                threshold,
                true_min,
                &true_min_x,
                tally,
            );
            *case_counter += 1;
        }
    }
}

/// Main end-to-end BaB soundness stress test.
///
/// Generates hundreds of small random residual/conv/ReLU networks, computes the
/// EXACT true-minimum margin for each by dense sampling, and confirms the
/// verifier never returns `Verified` for a threshold the true margin violates.
/// The wall-clock guard is a hang sentinel: the semantic coverage assertions
/// below must finish even when unrelated suites saturate a shared builder.
#[ntest::timeout(600_000)]
#[test]
fn bab_never_verifies_a_violated_property_stress() {
    let mut tally = Tally::default();
    let mut cases = 0usize;

    // Network count per family (each yields 4 threshold probes = 4 cases).
    // Bias toward residual/conv structures (recently-changed constrained
    // child-backward through Add).
    stress_family("mlp", build_mlp, 0xA11CE, 8, &mut tally, &mut cases);
    stress_family(
        "residual_mlp",
        build_residual_mlp,
        0xB0B,
        14,
        &mut tally,
        &mut cases,
    );
    stress_family(
        "two_branch_residual",
        build_two_branch_residual,
        0xC0FFEE,
        14,
        &mut tally,
        &mut cases,
    );
    stress_family("conv", build_conv, 0xD15EA5E, 10, &mut tally, &mut cases);
    stress_family(
        "residual_conv",
        build_residual_conv,
        0xE1F,
        16,
        &mut tally,
        &mut cases,
    );

    println!(
        "BaB soundness stress: {cases} cases | verified={} violated={} \
         unknown={} potential={} timeout={} zero_budget_refused={}",
        tally.verified,
        tally.violated,
        tally.unknown,
        tally.potential,
        tally.timeout,
        tally.zero_budget_refused
    );

    // Coverage sanity: the suite must actually exercise meaningful BaB outcomes,
    // not silently degrade to all-Unknown (which would make the soundness check
    // vacuous). We require a healthy number of Verified verdicts (the verdicts
    // whose soundness we are stress-testing) and that many cases land in the
    // property-FAILS region (Violated with a checked counterexample, or
    // PotentialViolation — the BaB engine's "found a sub-domain whose upper
    // bound is below threshold" signal). Either way, the verifier correctly
    // refused to (unsoundly) verify a violated property in those cases.
    assert!(
        tally.verified >= 30,
        "expected many Verified verdicts to stress-test (got {}); \
         the soundness invariant would be vacuous otherwise",
        tally.verified
    );
    assert!(
        tally.violated + tally.potential >= 20,
        "expected many property-fails verdicts (Violated/PotentialViolation) \
         exercising the near-optimum thresholds (got violated={} potential={})",
        tally.violated,
        tally.potential
    );
    assert!(
        tally.zero_budget_refused <= cases / 4,
        "too many cases ({}) observed a parallel zero-budget refusal; \
         the soundness coverage is no longer representative",
        tally.zero_budget_refused
    );
    assert!(cases >= 200, "expected hundreds of cases (got {cases})");
}
