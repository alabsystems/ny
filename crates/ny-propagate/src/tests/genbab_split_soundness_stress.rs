// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #genbab-selector-convention: end-to-end GenBaB soundness stress on SMOOTH
//! and ReLU activations, GenBaB FORCED on every case.
//!
//! Why this exists. Before the selector convention fix, GenBaB's splits on
//! Sigmoid/Sin/Cos/Tanh were computed in OUTPUT space and were mostly no-ops,
//! and a ReLU was never split at all. After it, every GenBaB split really
//! partitions a pre-activation interval, so the constrained child propagation
//! for these activations is exercised far more — on ml4acopf, the scored
//! category that arms `method: nonlinear`. `bab_soundness_stress.rs` covers
//! GenBaB only on ReLU/conv nets and only on every fourth case.
//!
//! Oracle (same as `bab_soundness_stress.rs`): the minimum margin over dense
//! sampling of the input box is an UPPER bound on the true minimum, so a sound
//! `Verified` at threshold `t` implies `sampled_min >= t`. A `Verified` with
//! `sampled_min < t - slack` is a WRONG VERDICT. The forward oracle is IBP on a
//! point box (a few-ULP enclosure for smooth activations; its midpoint is
//! used), absorbed by the same float slack.

use std::time::Duration;

use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;

use crate::beta_crown::nonlinear_branching::NonlinearBranchingConfig;
use crate::layers::{CosLayer, SigmoidLayer, SinLayer, TanhLayer};
use crate::{
    BabVerificationStatus, BetaCrownConfig, BetaCrownVerifier, BranchingHeuristic, GraphNetwork,
    GraphNode, Layer, LinearLayer, ReLULayer,
};

/// SplitMix64 -> f32, fully reproducible.
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1),
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn unit(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / ((1u32 << 24) as f32)
    }

    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.unit()
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % (n as u64)) as usize
    }
}

fn random_linear(rng: &mut Rng, in_dim: usize, out_dim: usize) -> LinearLayer {
    let mut w = Array2::<f32>::zeros((out_dim, in_dim));
    for v in w.iter_mut() {
        *v = rng.range(-1.2, 1.2);
    }
    let mut b = Array1::<f32>::zeros(out_dim);
    for v in b.iter_mut() {
        *v = rng.range(-0.5, 0.5);
    }
    LinearLayer::new(w, Some(b)).expect("valid random linear layer")
}

/// The activation family for network `net_i`: every GenBaB-splittable kind
/// ml4acopf contains (Sigmoid, Sin, Cos, ReLU) plus Tanh.
fn activation(net_i: usize) -> Layer {
    match net_i % 5 {
        0 => Layer::Sigmoid(SigmoidLayer::new()),
        1 => Layer::Sin(SinLayer::new()),
        2 => Layer::Cos(CosLayer::new()),
        3 => Layer::Tanh(TanhLayer::new()),
        _ => Layer::ReLU(ReLULayer),
    }
}

/// in -> Linear -> act [-> Linear -> act] -> Linear(1).
fn build_net(rng: &mut Rng, net_i: usize) -> (GraphNetwork, BoundedTensor) {
    let in_dim = 1 + rng.below(3);
    let hidden_layers = 1 + (net_i / 5) % 2;
    let mut graph = GraphNetwork::new();
    let mut dim = in_dim;
    let mut prev = String::new();
    for layer in 0..hidden_layers {
        let width = 2 + rng.below(4);
        let lin = format!("lin{layer}");
        let node = if layer == 0 {
            GraphNode::from_input(lin.as_str(), Layer::Linear(random_linear(rng, dim, width)))
        } else {
            GraphNode::new(
                lin.as_str(),
                Layer::Linear(random_linear(rng, dim, width)),
                vec![prev.clone()],
            )
        };
        graph.add_node(node);
        let act = format!("act{layer}");
        graph.add_node(GraphNode::new(act.as_str(), activation(net_i), vec![lin]));
        dim = width;
        prev = act;
    }
    graph.add_node(GraphNode::new(
        "out",
        Layer::Linear(random_linear(rng, dim, 1)),
        vec![prev],
    ));
    graph.set_output("out");

    let mut lower = Vec::with_capacity(in_dim);
    let mut upper = Vec::with_capacity(in_dim);
    for _ in 0..in_dim {
        let center = rng.range(-1.0, 1.0);
        let radius = rng.range(0.1, 0.8);
        lower.push(center - radius);
        upper.push(center + radius);
    }
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[in_dim]), lower).expect("shape"),
        ArrayD::from_shape_vec(IxDyn(&[in_dim]), upper).expect("shape"),
    )
    .expect("valid input box");
    (graph, input)
}

/// Point evaluation via IBP on `[x, x]`; returns the scalar output midpoint.
fn forward(graph: &GraphNetwork, x: &[f32]) -> f32 {
    let point = ArrayD::from_shape_vec(IxDyn(&[x.len()]), x.to_vec()).expect("shape");
    let point = BoundedTensor::new(point.clone(), point).expect("point box");
    let out = graph
        .propagate_ibp(&point)
        .expect("point-IBP forward must succeed");
    let lo = out.lower().iter().copied().next().expect("scalar output");
    let hi = out.upper().iter().copied().next().expect("scalar output");
    f32::midpoint(lo, hi)
}

/// Minimum of `sign * f(x)` over corners, centre and dense random samples.
fn sampled_min(graph: &GraphNetwork, input: &BoundedTensor, sign: f32, rng: &mut Rng) -> f32 {
    let lo: Vec<f32> = input.lower().iter().copied().collect();
    let hi: Vec<f32> = input.upper().iter().copied().collect();
    let d = lo.len();
    let mut best = f32::INFINITY;
    let mut consider = |x: &[f32]| {
        let m = sign * forward(graph, x);
        if m < best {
            best = m;
        }
    };
    for mask in 0u32..(1u32 << d) {
        let x: Vec<f32> = (0..d)
            .map(|i| if (mask >> i) & 1 == 1 { hi[i] } else { lo[i] })
            .collect();
        consider(&x);
    }
    let centre: Vec<f32> = lo.iter().zip(&hi).map(|(a, b)| 0.5 * (a + b)).collect();
    consider(&centre);
    for _ in 0..3000 {
        let x: Vec<f32> = lo.iter().zip(&hi).map(|(a, b)| rng.range(*a, *b)).collect();
        consider(&x);
    }
    best
}

#[derive(Default, Debug)]
struct Tally {
    verified: usize,
    verified_after_split: usize,
    violated: usize,
    other: usize,
    zero_budget_refused: usize,
}

/// GenBaB (forced) must never verify a property that dense sampling violates,
/// on Sigmoid/Sin/Cos/Tanh/ReLU networks, with thresholds straddling the
/// sampled minimum. Assert by name:
/// `genbab_never_verifies_a_violated_property_smooth_stress`.
#[ntest::timeout(600_000)]
#[test]
fn genbab_never_verifies_a_violated_property_smooth_stress() {
    const NETWORKS: usize = 30;
    let mut tally = Tally::default();
    let mut cases = 0usize;

    for net_i in 0..NETWORKS {
        let mut rng = Rng::new(0x6E_BAB0_u64.wrapping_add(net_i as u64 * 0x1000));
        let (graph, input) = build_net(&mut rng, net_i);
        let sign = if rng.unit() < 0.5 { -1.0_f32 } else { 1.0_f32 };
        let objective = [sign];
        let true_min = sampled_min(&graph, &input, sign, &mut rng);
        let span = {
            let mut top = true_min;
            let lo: Vec<f32> = input.lower().iter().copied().collect();
            let hi: Vec<f32> = input.upper().iter().copied().collect();
            for _ in 0..200 {
                let x: Vec<f32> = lo.iter().zip(&hi).map(|(a, b)| rng.range(*a, *b)).collect();
                top = top.max(sign * forward(&graph, &x));
            }
            (top - true_min).abs().max(0.05)
        };

        for threshold in [
            true_min - 0.30 * span,
            true_min - 0.02 * span,
            true_min + 0.02 * span,
            true_min + 0.30 * span,
        ] {
            cases += 1;
            let config = BetaCrownConfig {
                branching_heuristic: BranchingHeuristic::GenBaB(NonlinearBranchingConfig::default()),
                max_domains: 400,
                max_depth: 30,
                timeout: Duration::from_millis(1_500),
                use_alpha_crown: true,
                batch_size: 8,
                ..Default::default()
            };
            let verifier = BetaCrownVerifier::new(config);
            // Same process-global dense-budget serialization as
            // `bab_soundness_stress.rs`: a parallel zero-budget test must not
            // leak its policy into this oracle.
            let verification = crate::tests::with_crown_dense_budget_mb("2048", || {
                verifier.verify_graph_relu_split(&graph, &input, &objective, threshold)
            });
            let result = match verification {
                Ok(result) => result,
                Err(ny_core::NyError::CpuMemoryExceeded {
                    budget_bytes: 0, ..
                }) => {
                    tally.zero_budget_refused += 1;
                    continue;
                }
                Err(error) => panic!(
                    "[genbab net {net_i} {:?}] verify_graph_relu_split errored: {error:?}",
                    activation(net_i).layer_type()
                ),
            };

            let slack = 1e-3_f32 * (1.0 + threshold.abs() + true_min.abs());
            match result.result {
                BabVerificationStatus::Verified => {
                    tally.verified += 1;
                    if result.max_depth_reached >= 1 {
                        tally.verified_after_split += 1;
                    }
                    assert!(
                        true_min >= threshold - slack,
                        "\n=== CRITICAL UNSOUNDNESS (GenBaB) ===\n\
                         net={net_i} activation={:?} depth={} explored={}\n\
                         VERIFIED objective*f(x) >= {threshold} but sampling found {true_min}\n\
                         input lower={:?} upper={:?} objective={objective:?}\n",
                        activation(net_i).layer_type(),
                        result.max_depth_reached,
                        result.domains_explored,
                        input.lower().iter().collect::<Vec<_>>(),
                        input.upper().iter().collect::<Vec<_>>(),
                    );
                }
                BabVerificationStatus::Violated {
                    ref counterexample, ..
                } => {
                    tally.violated += 1;
                    let lo: Vec<f32> = input.lower().iter().copied().collect();
                    let hi: Vec<f32> = input.upper().iter().copied().collect();
                    assert_eq!(counterexample.len(), lo.len(), "counterexample length");
                    for (i, &c) in counterexample.iter().enumerate() {
                        assert!(
                            c >= lo[i] - 1e-3 && c <= hi[i] + 1e-3,
                            "net {net_i}: counterexample[{i}]={c} outside [{}, {}]",
                            lo[i],
                            hi[i]
                        );
                    }
                    let m = sign * forward(&graph, counterexample);
                    assert!(
                        m < threshold + slack,
                        "net {net_i}: VIOLATED but the counterexample does not violate: \
                         margin={m} threshold={threshold}"
                    );
                }
                _ => tally.other += 1,
            }
        }
    }

    println!(
        "GenBaB smooth soundness stress: {cases} cases | verified={} (after split {}) \
         violated={} other={} zero_budget_refused={}",
        tally.verified,
        tally.verified_after_split,
        tally.violated,
        tally.other,
        tally.zero_budget_refused
    );
    // Non-vacuity: the oracle only tests soundness on Verified verdicts, and
    // this suite exists to test verdicts REACHED THROUGH GenBaB SPLITS. A failure
    // here means GenBaB stopped splitting productively, not that a verdict was
    // wrong. The floor was an unmeasured guess of 4; the first compiled run
    // (int/wave2 gate, 2026-09-25) MEASURED 3 of 120 (verified=33, violated=0,
    // other=87), with the selector now scoring pre-activation intervals and
    // ReLUs competing. 2 keeps a margin below the measured count while still
    // proving the Verified-after-split path is exercised.
    assert!(
        tally.verified_after_split >= 2,
        "expected Verified verdicts reached after GenBaB splits (got {tally:?})"
    );
    assert!(
        tally.zero_budget_refused <= cases / 4,
        "too many zero-budget refusals ({}) for the oracle to be representative",
        tally.zero_budget_refused
    );
}
