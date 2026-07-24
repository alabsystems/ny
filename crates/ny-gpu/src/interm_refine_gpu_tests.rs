// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Oracle tests for per-subdomain INTERMEDIATE-BOUND refinement
//! (#interm-refine, `NY_INTERM_REFINE=1` production gate; the debug surface
//! calls the ungated production helper directly).
//!
//! Fixture: a conv resnet whose LAST ReLU (`relu_out`) sits between the
//! residual `add` and the output conv — the cifar100 `Gemm_56 → Relu_57 →
//! Gemm_58` shape. The production refinement runs ONE identity-seeded sound
//! backward over the truncated stack (`add` down to the input) and intersects
//! with the inherited (split-clamped) cache entry.
//!
//! 1. **Containment**: refined ⊆ inherited (the intersection contract), and at
//!    least one neuron strictly tightens (the lane actually fires).
//! 2. **Enclosure oracle (the soundness gate)**: sample the ROOT input box,
//!    keep the points satisfying the subdomain's `relu_out` split predicates
//!    (the subdomain differs from the root ONLY by those half-spaces), and
//!    check the refined `[l', u']` enclose the kept points' concrete
//!    pre-activations at the seed node.
//! 3. Root-domain variant (no splits): refined bounds enclose ALL sampled
//!    points' pre-activations.

use ndarray::{Array1, ArrayD, IxDyn};
use ny_propagate::beta_crown::gpu_beta_debug::{debug_interm_refine_last_relu, DebugSplit};
use ny_propagate::{
    layers::{AddLayer, Conv2dLayer, ReLULayer},
    GraphNetwork, GraphNode, Layer,
};
use ny_tensor::BoundedTensor;

use crate::wgpu_device::test_support::{gpu_test_serial_guard, require_device};

/// Deterministic LCG in [-1, 1).
struct Lcg(u64);
impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

fn conv(
    name: &str,
    input: &str,
    in_c: usize,
    out_c: usize,
    k: usize,
    pad: usize,
    hw: usize,
    rng: &mut Lcg,
) -> GraphNode {
    let kernel = ArrayD::from_shape_fn(IxDyn(&[out_c, in_c, k, k]), |_| rng.next_f32() * 0.4);
    let bias = Array1::from_shape_fn(out_c, |_| rng.next_f32() * 0.1);
    let mut layer = Conv2dLayer::new(kernel, Some(bias), (1, 1), (pad, pad)).expect("conv layer");
    layer.input_shape = Some((hw, hw));
    if input.is_empty() {
        GraphNode::from_input(name, Layer::Conv2d(layer))
    } else {
        GraphNode::new(name, Layer::Conv2d(layer), vec![input.to_string()])
    }
}

fn relu(name: &str, input: &str) -> GraphNode {
    GraphNode::new(name, Layer::ReLU(ReLULayer), vec![input.to_string()])
}

fn add(name: &str, a: &str, b: &str) -> GraphNode {
    GraphNode::new(
        name,
        Layer::Add(AddLayer),
        vec![a.to_string(), b.to_string()],
    )
}

/// input → conv1 → relu1 → conv2 → add(conv2, conv1) → relu_out → convo.
/// The last ReLU (`relu_out`) feeds the output through a unary chain; the
/// refinement seed node is `add` (its pre-activation), and the truncated stack
/// below the seed contains the residual block — the cifar100 shape in miniature.
fn last_relu_resnet(hw: usize, rng: &mut Lcg) -> GraphNetwork {
    let mut g = GraphNetwork::new();
    g.add_node(conv("conv1", "", 2, 4, 3, 1, hw, rng));
    g.add_node(relu("relu1", "conv1"));
    g.add_node(conv("conv2", "relu1", 4, 4, 3, 1, hw, rng));
    g.add_node(add("add", "conv2", "conv1"));
    g.add_node(relu("relu_out", "add"));
    g.add_node(conv("convo", "relu_out", 4, 3, 1, 0, hw, rng));
    g.set_output("convo");
    g
}

fn input_box(hw: usize, rng: &mut Lcg) -> BoundedTensor {
    let center = ArrayD::from_shape_fn(IxDyn(&[2, hw, hw]), |_| rng.next_f32() * 0.5);
    let radius = 0.15f32;
    BoundedTensor::new(center.mapv(|c| c - radius), center.mapv(|c| c + radius)).expect("input box")
}

/// Splits on `relu_out` unstable neurons, branch = the side with more interval
/// mass (`is_active ⇔ u ≥ −l`) so random box samples survive the predicate
/// filter with probability ≥ 1/2 each.
fn pick_splits_mass(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    relu_node: &str,
    pre_node: &str,
    n: usize,
) -> Vec<DebugSplit> {
    let ibp_map = graph.collect_node_bounds(input).expect("IBP node bounds");
    let pre = ibp_map[pre_node].flatten();
    let mut splits = Vec::new();
    for i in 0..pre.len() {
        let (l, u) = (pre.lower()[[i]], pre.upper()[[i]]);
        if l < 0.0 && u > 0.0 {
            let k = splits.len();
            let beta = match k % 3 {
                0 => 0.05,
                1 => 0.0,
                _ => 0.12,
            };
            splits.push((relu_node.to_string(), i, u >= -l, beta));
            if splits.len() == n {
                break;
            }
        }
    }
    assert!(
        !splits.is_empty(),
        "no unstable neurons found at {pre_node} — fixture too tight"
    );
    splits
}

/// Concrete pre-activation values at `pre_node` for a point (degenerate box
/// forward — exact for conv/add up to f32 rounding).
fn concrete_pre(graph: &GraphNetwork, point: &BoundedTensor, pre_node: &str) -> Vec<f32> {
    let nb = graph.collect_node_bounds(point).expect("point forward");
    let pre = nb[pre_node].flatten();
    (0..pre.len())
        .map(|i| f32::midpoint(pre.lower()[[i]], pre.upper()[[i]]))
        .collect()
}

/// Shared driver: run the production refinement for the subdomain defined by
/// `splits`, assert containment-in-inherited + strict tightening, then the
/// sampling enclosure oracle over predicate-satisfying points.
fn run_refine_oracle(splits: &[DebugSplit], seed: u64, label: &str) {
    let _guard = gpu_test_serial_guard();
    let device = require_device();
    let mut rng = Lcg(seed);
    let graph = last_relu_resnet(4, &mut rng);
    let input = input_box(4, &mut rng);

    let (seed_node, inherited, refined) =
        debug_interm_refine_last_relu(&graph, &input, splits, device.as_ref()).unwrap_or_else(
            || panic!("{label}: refinement must fire on this fixture (sound GPU present)"),
        );
    assert_eq!(
        seed_node, "add",
        "{label}: seed must be the last ReLU's input"
    );
    assert_eq!(inherited.len(), refined.len(), "{label}: dim mismatch");

    // CLAIM 1a — containment: refined ⊆ inherited (the intersection contract).
    let mut strictly_tightened = 0usize;
    for (j, (&(il, iu), &(rl, ru))) in inherited.iter().zip(refined.iter()).enumerate() {
        assert!(
            rl >= il && ru <= iu,
            "{label}: neuron {j} refined [{rl}, {ru}] escapes inherited [{il}, {iu}]"
        );
        assert!(
            rl.is_finite() && ru.is_finite() && rl <= ru,
            "{label}: neuron {j} refined bounds invalid [{rl}, {ru}]"
        );
        if rl > il || ru < iu {
            strictly_tightened += 1;
        }
    }
    // CLAIM 1b — the lane fires: the CROWN identity backward must beat the
    // inherited IBP-based cache somewhere on this deep-enough fixture.
    assert!(
        strictly_tightened > 0,
        "{label}: refinement tightened nothing — the lane did not engage"
    );

    // CLAIM 2 — enclosure oracle. The subdomain = root box ∩ the `relu_out`
    // split half-spaces on the SEED node's pre-activations; sample root points,
    // keep those satisfying every predicate, check [l', u'] enclosure.
    let in_lo: Vec<f32> = input.lower().iter().copied().collect();
    let in_hi: Vec<f32> = input.upper().iter().copied().collect();
    let in_shape: Vec<usize> = input.shape().to_vec();
    let mut kept = 0usize;
    for _ in 0..1200 {
        let point: Vec<f32> = (0..in_lo.len())
            .map(|i| {
                let f = f32::midpoint(rng.next_f32(), 1.0);
                in_lo[i] + f * (in_hi[i] - in_lo[i])
            })
            .collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&in_shape), point).expect("point shape");
        let point_box = BoundedTensor::new(arr.clone(), arr).expect("point box");
        let z = concrete_pre(&graph, &point_box, &seed_node);
        assert_eq!(z.len(), refined.len(), "{label}: concrete dim mismatch");

        // Predicate filter (skip a tiny boundary band — f32 eval noise).
        let satisfies = splits.iter().all(|(_, idx, is_active, _)| {
            if *is_active {
                z[*idx] >= 1e-4
            } else {
                z[*idx] <= -1e-4
            }
        });
        if !satisfies {
            continue;
        }
        kept += 1;
        for (j, &zj) in z.iter().enumerate() {
            let (rl, ru) = refined[j];
            let tol = 1e-4f32 + 1e-4 * zj.abs();
            assert!(
                rl - tol <= zj && zj <= ru + tol,
                "{label}: kept point violates refined bounds at neuron {j}: \
                 z={zj} outside [{rl}, {ru}]"
            );
        }
    }
    assert!(
        kept >= 20,
        "{label}: only {kept} samples satisfied the split predicates — \
         oracle underpowered, fix the fixture"
    );
}

/// Subdomain with real last-ReLU splits (the measured cifar100 shape: 100% of
/// premises on the last ReLU): refined bounds must contain every
/// predicate-satisfying sample and strictly tighten the inherited entry.
#[test]
fn interm_refine_encloses_subdomain_reachable_set_with_last_relu_splits() {
    let mut rng = Lcg(0xC1FA_0885);
    let graph = last_relu_resnet(4, &mut rng);
    let input = input_box(4, &mut rng);
    let splits = pick_splits_mass(&graph, &input, "relu_out", "add", 2);
    run_refine_oracle(&splits, 0xC1FA_0885, "last-relu-splits");
}

/// Root domain (no splits): the refinement is a plain sound α-CROWN identity
/// backward — refined bounds must enclose ALL sampled points.
#[test]
fn interm_refine_encloses_root_reachable_set_without_splits() {
    run_refine_oracle(&[], 0xC1FA_0885, "root-no-splits");
}
