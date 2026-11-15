// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #relusplitter-lp-refine adapter tests.
//!
//! Hermetic tests (always on under `mip`) pin the chain gate, the decline
//! reasons, the layer scope and — the load-bearing one — CONTAINMENT of every
//! candidate the refiner returns: on random MLPs it is checked against dense
//! float sampling (tolerance `1e-9 * (1 + |v|)` for the sampler's own f64
//! rounding) and against an EXACT-RATIONAL forward at every box vertex and at
//! each neuron's extreme samples, with no tolerance.
//!
//! The real-ONNX audit is `external-vnncomp` (it needs the gitignored
//! `benchmarks/vnncomp2026` tree — symlink it into a worktree — and runs a few
//! hundred LPs, so run it in release):
//!
//! ```text
//! targo --unverified test --release -p ny-cli --features mip,external-vnncomp --bin ny lp_refined_
//! ```

use super::*;
use ndarray::{Array1, Array2};
use num_rational::BigRational;
use num_traits::Zero;
use ny_propagate::layers::{LinearLayer, ReLULayer};
use ny_propagate::Network;
use ny_tensor::{next_down_f32, next_up_f32};

/// Deterministic xorshift64 (tests only).
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn unit(&mut self) -> f64 {
        #[allow(clippy::cast_precision_loss)]
        let v = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        v
    }

    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }
}

/// A random `Linear -> ReLU -> ... -> Linear` network with f32 parameters.
fn random_network(rng: &mut Rng, dims: &[usize]) -> Network {
    let mut network = Network::new();
    let affine = dims.len() - 1;
    for (k, pair) in dims.windows(2).enumerate() {
        let (fan_in, fan_out) = (pair[0], pair[1]);
        #[allow(clippy::cast_possible_truncation)]
        let w: Vec<f32> = (0..fan_in * fan_out)
            .map(|_| rng.uniform(-1.0, 1.0) as f32)
            .collect();
        #[allow(clippy::cast_possible_truncation)]
        let b: Vec<f32> = (0..fan_out)
            .map(|_| rng.uniform(-0.5, 0.5) as f32)
            .collect();
        network.add_layer(Layer::Linear(
            LinearLayer::new(
                Array2::from_shape_vec((fan_out, fan_in), w).expect("weight shape"),
                Some(Array1::from_vec(b)),
            )
            .expect("affine layer"),
        ));
        if k + 1 < affine {
            network.add_layer(Layer::ReLU(ReLULayer));
        }
    }
    network
}

fn box_tensor(lower: &[f32], upper: &[f32]) -> BoundedTensor {
    BoundedTensor::new(
        ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[lower.len()]), lower.to_vec())
            .expect("lower"),
        ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[upper.len()]), upper.to_vec())
            .expect("upper"),
    )
    .expect("finite box")
}

fn random_box(rng: &mut Rng, n: usize) -> (Vec<f32>, Vec<f32>) {
    let mut lower = Vec::with_capacity(n);
    let mut upper = Vec::with_capacity(n);
    for _ in 0..n {
        #[allow(clippy::cast_possible_truncation)]
        let c = rng.uniform(-1.0, 1.0) as f32;
        #[allow(clippy::cast_possible_truncation)]
        let r = rng.uniform(0.2, 0.8) as f32;
        lower.push(c - r);
        upper.push(c + r);
    }
    (lower, upper)
}

/// Sound IBP boxes for every chain node (pre-activation Linear nodes AND the
/// ReLU nodes), keyed by graph node name: f64 interval arithmetic widened by a
/// generous slack, cast OUTWARD to f32.
fn ibp_node_bounds(
    chain: &ReluChain,
    lower: &[f32],
    upper: &[f32],
) -> HashMap<String, BoundedTensor> {
    let mut map = HashMap::new();
    let mut lo: Vec<f64> = lower.iter().map(|&v| f64::from(v)).collect();
    let mut hi: Vec<f64> = upper.iter().map(|&v| f64::from(v)).collect();
    for k in 0..chain.weights.len() {
        let (fan_in, fan_out) = (chain.layer_dims[k], chain.layer_dims[k + 1]);
        let mut zl = Vec::with_capacity(fan_out);
        let mut zu = Vec::with_capacity(fan_out);
        for i in 0..fan_out {
            let mut l = chain.biases[k][i];
            let mut u = chain.biases[k][i];
            for j in 0..fan_in {
                let w = chain.weights[k][i * fan_in + j];
                if w >= 0.0 {
                    l += w * lo[j];
                    u += w * hi[j];
                } else {
                    l += w * hi[j];
                    u += w * lo[j];
                }
            }
            let slack = 1e-9 * (1.0 + l.abs().max(u.abs()));
            #[allow(clippy::cast_possible_truncation)]
            {
                zl.push(next_down_f32((l - slack) as f32));
                zu.push(next_up_f32((u + slack) as f32));
            }
        }
        if k < chain.pre_nodes.len() {
            map.insert(chain.pre_nodes[k].clone(), box_tensor(&zl, &zu));
            let rl: Vec<f32> = zl.iter().map(|v| v.max(0.0)).collect();
            let ru: Vec<f32> = zu.iter().map(|v| v.max(0.0)).collect();
            map.insert(chain.relu_nodes[k].clone(), box_tensor(&rl, &ru));
            lo = rl.iter().map(|&v| f64::from(v)).collect();
            hi = ru.iter().map(|&v| f64::from(v)).collect();
        }
    }
    map
}

/// Pre-activations of every hidden layer, f64.
fn hidden_preacts(chain: &ReluChain, x: &[f64]) -> Vec<Vec<f64>> {
    let mut out = Vec::new();
    let mut a = x.to_vec();
    for k in 0..chain.hidden() {
        let (fan_in, fan_out) = (chain.layer_dims[k], chain.layer_dims[k + 1]);
        let z: Vec<f64> = (0..fan_out)
            .map(|i| {
                (0..fan_in)
                    .map(|j| chain.weights[k][i * fan_in + j] * a[j])
                    .sum::<f64>()
                    + chain.biases[k][i]
            })
            .collect();
        a = z.iter().map(|v| v.max(0.0)).collect();
        out.push(z);
    }
    out
}

/// Pre-activations of every hidden layer, exact rationals.
fn hidden_preacts_exact(chain: &ReluChain, x: &[f64]) -> Vec<Vec<BigRational>> {
    let q = |v: f64| BigRational::from_float(v).expect("finite");
    let mut out = Vec::new();
    let mut a: Vec<BigRational> = x.iter().map(|&v| q(v)).collect();
    for k in 0..chain.hidden() {
        let (fan_in, fan_out) = (chain.layer_dims[k], chain.layer_dims[k + 1]);
        let z: Vec<BigRational> = (0..fan_out)
            .map(|i| {
                let mut acc = q(chain.biases[k][i]);
                for (j, aj) in a.iter().enumerate().take(fan_in) {
                    acc += q(chain.weights[k][i * fan_in + j]) * aj;
                }
                acc
            })
            .collect();
        a = z
            .iter()
            .map(|v| {
                if *v > BigRational::zero() {
                    v.clone()
                } else {
                    BigRational::zero()
                }
            })
            .collect();
        out.push(z);
    }
    out
}

fn refine_with(
    config: MipRefineIntermediates,
    graph: &GraphNetwork,
    input: &BoundedTensor,
    node_bounds: &HashMap<String, BoundedTensor>,
    deadline: Option<Instant>,
) -> RootIntermRefineOutcome {
    LpRootIntermRefiner::new(config).refine(&RootIntermRefineRequest {
        graph,
        input_bounds: input,
        node_bounds,
        deadline,
    })
}

fn far() -> Option<Instant> {
    Some(Instant::now() + Duration::from_secs(120))
}

fn generous() -> MipRefineIntermediates {
    MipRefineIntermediates {
        layers: 3,
        budget_frac: 0.9,
    }
}

#[test]
fn chain_gate_maps_pre_activation_and_relu_nodes() {
    let mut rng = Rng::new(1);
    let network = random_network(&mut rng, &[3, 5, 4, 2]);
    let graph = GraphNetwork::from_sequential(&network).expect("graph");
    let chain = extract_relu_chain(&graph).expect("pure chain");
    assert_eq!(chain.layer_dims, vec![3, 5, 4, 2]);
    assert_eq!(chain.pre_nodes, vec!["layer_0", "layer_2"]);
    assert_eq!(chain.relu_nodes, vec!["layer_1", "layer_3"]);
    assert_eq!(chain.hidden(), 2);
    // Parameters are taken EXACTLY (f32 -> f64 is exact).
    let Layer::Linear(first) = &network.layers()[0] else {
        panic!("first layer is Linear");
    };
    // Row 1, column 2 of a 3-input layer is flat index 3 + 2.
    assert_eq!(chain.weights[0][3 + 2], f64::from(first.weight()[[1, 2]]));
    assert_eq!(
        chain.biases[0][4],
        f64::from(first.bias().expect("bias")[4])
    );
}

#[test]
fn chain_gate_declines_everything_else() {
    // Two ReLUs in a row.
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(ndarray::arr2(&[[1.0_f32]]), None).expect("affine"),
    ));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(
        LinearLayer::new(ndarray::arr2(&[[1.0_f32]]), None).expect("affine"),
    ));
    let graph = GraphNetwork::from_sequential(&network).expect("graph");
    assert_eq!(extract_relu_chain(&graph).unwrap_err(), "not_relu_chain");

    // A single affine layer has no hidden layer at all.
    let mut single = Network::new();
    single.add_layer(Layer::Linear(
        LinearLayer::new(ndarray::arr2(&[[1.0_f32, 2.0]]), None).expect("affine"),
    ));
    let graph = GraphNetwork::from_sequential(&single).expect("graph");
    assert!(extract_relu_chain(&graph).is_err());
}

/// The load-bearing test: every candidate the refiner returns encloses the
/// true forward (sampled and exact-rational), is never wider than the stored
/// box it started from, covers only layers `1..=layers`, and something is
/// actually tightened (otherwise containment is vacuous).
#[test]
#[ntest::timeout(180_000)]
fn refiner_candidates_contain_the_sampled_and_exact_forward() {
    let mut tightened = 0usize;
    for seed in 1..=4u64 {
        let mut rng = Rng::new(0xc7_0000 + seed);
        let network = random_network(&mut rng, &[4, 8, 8, 8, 3]);
        let graph = GraphNetwork::from_sequential(&network).expect("graph");
        let chain = extract_relu_chain(&graph).expect("pure chain");
        let (lower, upper) = random_box(&mut rng, 4);
        let input = box_tensor(&lower, &upper);
        let stored = ibp_node_bounds(&chain, &lower, &upper);
        let outcome = refine_with(generous(), &graph, &input, &stored, far());
        let t = &outcome.telemetry;
        assert!(t.engaged, "seed {seed}: {t:?}");
        assert_eq!(t.reason, "ok");
        // hidden = 3 -> layers 1..=2 (layer 0 is never re-solved).
        assert_eq!(t.layers, 2);
        assert!(!t.deadline_hit);
        assert!(t.rigorous <= t.lp_solves);
        tightened += t.tightened_neurons;
        assert!(!outcome.candidates.contains_key(&chain.pre_nodes[0]));
        assert!(!outcome.candidates.contains_key(&chain.relu_nodes[0]));

        // Never wider than the stored box.
        for (name, (cl, cu)) in &outcome.candidates {
            let s = &stored[name];
            for (i, (&l, &u)) in cl.iter().zip(cu).enumerate() {
                assert!(
                    l >= f64::from(s.lower()[i]) && u <= f64::from(s.upper()[i]) && l <= u,
                    "seed {seed} {name}[{i}] widened or crossed"
                );
            }
        }

        let mut points: Vec<Vec<f64>> = (0..(1usize << 4))
            .map(|mask| {
                (0..4)
                    .map(|j| {
                        if mask & (1 << j) == 0 {
                            f64::from(lower[j])
                        } else {
                            f64::from(upper[j])
                        }
                    })
                    .collect()
            })
            .collect();
        for _ in 0..20_000 {
            points.push(
                (0..4)
                    .map(|j| rng.uniform(f64::from(lower[j]), f64::from(upper[j])))
                    .collect(),
            );
        }
        let mut extremes: Vec<Vec<f64>> = Vec::new();
        for layer in 1..chain.hidden() {
            let (cl, cu) = &outcome.candidates[&chain.pre_nodes[layer]];
            let (rl, ru) = &outcome.candidates[&chain.relu_nodes[layer]];
            let width = cl.len();
            let mut arg_lo = vec![(f64::INFINITY, 0usize); width];
            let mut arg_hi = vec![(f64::NEG_INFINITY, 0usize); width];
            for (p, x) in points.iter().enumerate() {
                let z = &hidden_preacts(&chain, x)[layer];
                for i in 0..width {
                    let v = z[i];
                    let tol = 1e-9 * (1.0 + v.abs());
                    assert!(
                        v >= cl[i] - tol && v <= cu[i] + tol,
                        "seed {seed} layer {layer} neuron {i}: {v} escapes [{}, {}]",
                        cl[i],
                        cu[i]
                    );
                    let r = v.max(0.0);
                    assert!(r >= rl[i] - tol && r <= ru[i] + tol);
                    if v < arg_lo[i].0 {
                        arg_lo[i] = (v, p);
                    }
                    if v > arg_hi[i].0 {
                        arg_hi[i] = (v, p);
                    }
                }
            }
            for (_, p) in arg_lo.iter().chain(arg_hi.iter()) {
                extremes.push(points[*p].clone());
            }
        }
        // Exact rationals, no tolerance: vertices + every extreme sample.
        for x in points.iter().take(1 << 4).chain(extremes.iter()) {
            let exact = hidden_preacts_exact(&chain, x);
            for layer in 1..chain.hidden() {
                let (cl, cu) = &outcome.candidates[&chain.pre_nodes[layer]];
                for (i, v) in exact[layer].iter().enumerate() {
                    let lo = BigRational::from_float(cl[i]).expect("finite");
                    let hi = BigRational::from_float(cu[i]).expect("finite");
                    assert!(
                        &lo <= v && v <= &hi,
                        "seed {seed} layer {layer} neuron {i}: exact value escapes"
                    );
                }
            }
        }
    }
    assert!(
        tightened > 0,
        "no neuron was tightened on any seed: the containment check is vacuous"
    );
}

#[test]
fn refiner_respects_the_layer_scope() {
    let mut rng = Rng::new(99);
    let network = random_network(&mut rng, &[4, 6, 6, 6, 6, 2]);
    let graph = GraphNetwork::from_sequential(&network).expect("graph");
    let chain = extract_relu_chain(&graph).expect("pure chain");
    let (lower, upper) = random_box(&mut rng, 4);
    let input = box_tensor(&lower, &upper);
    let stored = ibp_node_bounds(&chain, &lower, &upper);
    let outcome = refine_with(
        MipRefineIntermediates {
            layers: 1,
            budget_frac: 0.9,
        },
        &graph,
        &input,
        &stored,
        far(),
    );
    assert_eq!(outcome.telemetry.layers, 1);
    let mut keys: Vec<&String> = outcome.candidates.keys().collect();
    keys.sort();
    assert_eq!(keys, vec![&chain.pre_nodes[1], &chain.relu_nodes[1]]);
}

#[test]
fn refiner_declines_without_a_bounded_window_or_its_inputs() {
    let mut rng = Rng::new(5);
    let network = random_network(&mut rng, &[3, 4, 4, 2]);
    let graph = GraphNetwork::from_sequential(&network).expect("graph");
    let chain = extract_relu_chain(&graph).expect("pure chain");
    let (lower, upper) = random_box(&mut rng, 3);
    let input = box_tensor(&lower, &upper);
    let stored = ibp_node_bounds(&chain, &lower, &upper);

    let none = refine_with(generous(), &graph, &input, &stored, None);
    assert_eq!(none.telemetry.reason, "no_deadline");
    assert!(!none.telemetry.engaged && none.candidates.is_empty());

    let past = Instant::now()
        .checked_sub(Duration::from_millis(5))
        .expect("representable");
    let spent = refine_with(generous(), &graph, &input, &stored, Some(past));
    assert_eq!(spent.telemetry.reason, "no_budget");
    assert!(spent.candidates.is_empty());

    let mut missing = stored.clone();
    missing.remove(&chain.pre_nodes[1]);
    let gone = refine_with(generous(), &graph, &input, &missing, far());
    assert_eq!(gone.telemetry.reason, "missing_node_bounds");
    assert!(gone.candidates.is_empty());

    let short = box_tensor(&lower[..2], &upper[..2]);
    let bad = refine_with(generous(), &graph, &short, &stored, far());
    assert_eq!(bad.telemetry.reason, "bad_input_box");

    let invalid = refine_with(
        MipRefineIntermediates {
            layers: 0,
            budget_frac: 0.5,
        },
        &graph,
        &input,
        &stored,
        far(),
    );
    assert_eq!(invalid.telemetry.reason, "invalid_config");
}

/// One hidden layer: nothing to refine past layer 0, so the refiner declines.
#[test]
fn refiner_declines_a_single_hidden_layer_net() {
    let mut rng = Rng::new(8);
    let network = random_network(&mut rng, &[3, 4, 2]);
    let graph = GraphNetwork::from_sequential(&network).expect("graph");
    let chain = extract_relu_chain(&graph).expect("pure chain");
    let (lower, upper) = random_box(&mut rng, 3);
    let input = box_tensor(&lower, &upper);
    let stored = ibp_node_bounds(&chain, &lower, &upper);
    let outcome = refine_with(generous(), &graph, &input, &stored, far());
    assert_eq!(outcome.telemetry.reason, "no_refinable_layer");
    assert!(outcome.candidates.is_empty());
}

// ---------------------------------------------------------------------------
// Real-ONNX soundness audit (external-vnncomp).
// ---------------------------------------------------------------------------

#[cfg(feature = "external-vnncomp")]
mod corpus {
    use super::*;
    use std::path::{Path, PathBuf};

    fn relusplitter_root() -> PathBuf {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../benchmarks/vnncomp2026/benchmarks/relusplitter_2026/2.0");
        assert!(
            root.join("onnx/model_2_2.onnx").is_file(),
            "relusplitter corpus missing at {} — symlink benchmarks/vnncomp2026 from the main tree",
            root.display()
        );
        root
    }

    /// `lp_refined_intermediates_contain_sampled_activations` on the real
    /// relusplitter gap row `model_2_2` / `d2_eps_0.05_sample_6_label_6`:
    /// load the ONNX the way production does, refine layers 1..=3 from sound
    /// IBP boxes, and check every refined pre-activation against 1e5 sampled
    /// inputs (f64 forward) plus the exact-rational forward at the box centre
    /// and at each neuron's extreme samples.
    #[test]
    #[expect(
        clippy::manual_midpoint,
        reason = "retain independent oracle sample arithmetic"
    )]
    fn lp_refined_intermediates_contain_sampled_activations_model_2_2() {
        let root = relusplitter_root();
        let network = ny_onnx::load_onnx(root.join("onnx/model_2_2.onnx"))
            .expect("load model_2_2")
            .to_propagate_network()
            .expect("sequential network");
        let spec =
            ny_onnx::vnnlib::load_vnnlib(root.join("vnnlib/d2_eps_0.05_sample_6_label_6.vnnlib"))
                .expect("load vnnlib");
        let (lower, upper) = spec.split_input_bounds_f32();
        let graph = GraphNetwork::from_sequential(&network).expect("graph");
        let chain = extract_relu_chain(&graph).expect("model_2_2 is a pure ReLU chain");
        assert_eq!(chain.layer_dims, vec![784, 128, 256, 256, 256, 10]);
        let input = box_tensor(&lower, &upper);
        let stored = ibp_node_bounds(&chain, &lower, &upper);
        let outcome = refine_with(
            MipRefineIntermediates {
                layers: 3,
                budget_frac: 0.9,
            },
            &graph,
            &input,
            &stored,
            Some(Instant::now() + Duration::from_mins(15)),
        );
        eprintln!("{:?}", outcome.telemetry);
        assert!(outcome.telemetry.engaged, "{:?}", outcome.telemetry);
        assert!(outcome.telemetry.tightened_neurons > 0);

        let mut rng = Rng::new(0x2_2);
        let n = lower.len();
        let mut extremes: HashMap<(usize, usize), (f64, Vec<f64>, f64, Vec<f64>)> = HashMap::new();
        for sample in 0..100_000 {
            let x: Vec<f64> = (0..n)
                .map(|j| {
                    if sample % 2 == 0 {
                        rng.uniform(f64::from(lower[j]), f64::from(upper[j]))
                    } else if rng.unit() < 0.5 {
                        f64::from(lower[j])
                    } else {
                        f64::from(upper[j])
                    }
                })
                .collect();
            let z = hidden_preacts(&chain, &x);
            for layer in 1..=outcome.telemetry.layers {
                let Some((cl, cu)) = outcome.candidates.get(&chain.pre_nodes[layer]) else {
                    continue;
                };
                for (i, &v) in z[layer].iter().enumerate() {
                    let tol = 1e-9 * (1.0 + v.abs());
                    assert!(
                        v >= cl[i] - tol && v <= cu[i] + tol,
                        "layer {layer} neuron {i}: sample {v} escapes [{}, {}]",
                        cl[i],
                        cu[i]
                    );
                    let e = extremes.entry((layer, i)).or_insert((
                        f64::INFINITY,
                        Vec::new(),
                        f64::NEG_INFINITY,
                        Vec::new(),
                    ));
                    if v < e.0 {
                        e.0 = v;
                        e.1.clone_from(&x);
                    }
                    if v > e.2 {
                        e.2 = v;
                        e.3.clone_from(&x);
                    }
                }
            }
        }
        // Exact rationals at a bounded subset of extreme points (each check is
        // a full exact forward; 64 points keep the test minutes, not hours).
        let mut exact_points: Vec<Vec<f64>> = extremes
            .values()
            .flat_map(|e| [e.1.clone(), e.3.clone()])
            .take(64)
            .collect();
        exact_points.push(
            (0..n)
                .map(|j| (f64::from(lower[j]) + f64::from(upper[j])) / 2.0)
                .collect(),
        );
        for x in &exact_points {
            let exact = hidden_preacts_exact(&chain, x);
            for layer in 1..=outcome.telemetry.layers {
                let Some((cl, cu)) = outcome.candidates.get(&chain.pre_nodes[layer]) else {
                    continue;
                };
                for (i, v) in exact[layer].iter().enumerate() {
                    let lo = BigRational::from_float(cl[i]).expect("finite");
                    let hi = BigRational::from_float(cu[i]).expect("finite");
                    assert!(
                        &lo <= v && v <= &hi,
                        "layer {layer} neuron {i}: exact escape"
                    );
                }
            }
        }
    }
}
