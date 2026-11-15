// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #vit-root-alpha (ENGINE_PROGRAM §2 V2) at the CLI boundary — the only crate
//! that sees BOTH sides of the search/certify split.
//!
//! `real_cpu_f32_graph_search_is_certified_contained_and_never_looser`: the
//! REAL `ny_bbe::CpuF32Backend`, registered exactly as `main.rs` does, driven
//! through the graph lane's public α-CROWN collection on random attention
//! blocks (monolithic softmax and the fixed-index decomposition). Whatever the
//! search proposes, the published margin bounds are never looser than the
//! certified incumbent on any row (the incumbent is the certified ascent's
//! own first fold: `margin_alpha_search: ascent` with one iteration publishes
//! exactly it) and never exceed the network's margin at sampled points. Every
//! joint run must also ENGAGE — the phase counters
//! (`graph_root_phase_counts`) must show one entered, unrefused phase with one
//! certified candidate — because a refused phase publishes the incumbent and
//! would satisfy every bound assertion here while testing nothing.

use std::sync::Arc;

use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_propagate::bounds::{AlphaSpecAscent, AlphaSpecEarlyExit, MarginAlphaSearch};
use ny_propagate::layers::{BilinearCrownLayer, LinearLayer, ReLULayer, SoftmaxLayer};
use ny_propagate::network::graph_root_search::graph_root_phase_counts;
use ny_propagate::{AlphaCrownConfig, GraphNetwork, GraphNode, Layer};
use ny_tensor::BoundedTensor;

const T: usize = 3;
const D: usize = 2;
const OUT: usize = T * 4;

struct Lcg(u64);

impl Lcg {
    fn unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }

    fn values(&mut self, n: usize, scale: f64) -> Vec<f32> {
        (0..n)
            .map(|_| (scale * (2.0 * self.unit() - 1.0)) as f32)
            .collect()
    }
}

fn linear(rng: &mut Lcg, n_out: usize, n_in: usize, scale: f64) -> Layer {
    let weight = Array2::from_shape_vec((n_out, n_in), rng.values(n_out * n_in, scale))
        .expect("weight shape");
    let bias = Array1::from(rng.values(n_out, 0.2));
    Layer::Linear(LinearLayer::new(weight, Some(bias)).expect("linear layer"))
}

fn input_box(seed: u64) -> BoundedTensor {
    let mut rng = Lcg(seed ^ 0xc11);
    let centre = rng.values(T * D, 1.0);
    let lower: Vec<f32> = centre.iter().map(|c| c - 0.3).collect();
    let upper: Vec<f32> = centre.iter().map(|c| c + 0.3).collect();
    BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[T, D]), lower).expect("lower"),
        ArrayD::from_shape_vec(IxDyn(&[T, D]), upper).expect("upper"),
    )
    .expect("input box")
}

fn attention_block(seed: u64, input: &BoundedTensor, fixed_index: bool) -> GraphNetwork {
    let mut rng = Lcg(seed);
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("q", linear(&mut rng, D, D, 0.9)));
    graph.add_node(GraphNode::from_input("k", linear(&mut rng, D, D, 0.9)));
    graph.add_node(GraphNode::from_input("v", linear(&mut rng, D, D, 0.9)));
    graph.add_node(GraphNode::binary(
        "qk",
        Layer::BilinearCrown(BilinearCrownLayer::new(true, Some(0.7))),
        "q",
        "k",
    ));
    graph.add_node(GraphNode::new(
        "sm",
        Layer::Softmax(SoftmaxLayer::new(-1)),
        vec!["qk".to_string()],
    ));
    graph.add_node(GraphNode::binary(
        "av",
        Layer::BilinearCrown(BilinearCrownLayer::new(false, None)),
        "sm",
        "v",
    ));
    graph.add_node(GraphNode::new(
        "proj",
        linear(&mut rng, 3, D, 0.9),
        vec!["av".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "act",
        Layer::ReLU(ReLULayer),
        vec!["proj".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "head",
        linear(&mut rng, 4, 3, 1.0),
        vec!["act".to_string()],
    ));
    graph.set_output("head");
    if fixed_index {
        graph
            .decompose_softmax_complex_fixed_index(input)
            .expect("the fixed-index rewrite runs");
    }
    graph
}

fn margin_rows() -> Vec<AlphaSpecEarlyExit> {
    (1..4)
        .map(|j| {
            let mut objective = vec![0.0f32; OUT];
            objective[0] = 1.0;
            objective[j] = -1.0;
            AlphaSpecEarlyExit {
                objective,
                threshold: 50.0,
                verify_upper_bound: false,
            }
        })
        .collect()
}

fn config(fixed_index: bool, search: MarginAlphaSearch, iterations: usize) -> AlphaCrownConfig {
    AlphaCrownConfig {
        iterations,
        learning_rate: 0.3,
        early_stop_patience: 1000,
        adaptive_skip: false,
        root_alpha_margin: true,
        root_alpha_margin_gradient: true,
        matmul_alpha: true,
        alpha_spec_slots: 0,
        spec_ascent: AlphaSpecAscent::new(margin_rows()),
        margin_alpha_freeze_mul_reciprocal: fixed_index,
        margin_alpha_search: search,
        ..AlphaCrownConfig::default()
    }
}

/// The published margin bounds `(lower, upper)` of one collection.
fn published(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    config: &AlphaCrownConfig,
) -> (Vec<f32>, Vec<f32>) {
    let (_, state) = graph
        .collect_alpha_crown_bounds_dag_with_engine(input, config, None)
        .expect("the α-CROWN collection runs");
    let publication = state
        .margin_alpha_publication()
        .expect("the certified margin lane publishes (no publication = the lane never ran)");
    (publication.lower().to_vec(), publication.upper().to_vec())
}

#[test]
fn real_cpu_f32_graph_search_is_certified_contained_and_never_looser() {
    let _env = ny_test_utils::env::lock_env_shared();
    // Exactly the registration `main.rs` performs (first install wins; the test
    // binary never runs `main`).
    ny_propagate::network::graph_root_search::set_graph_root_search_backend_factory(|| {
        Some(Arc::new(ny_bbe::CpuF32Backend::with_threads(2))
            as Arc<dyn ny_core::search::SearchBackend>)
    });
    let rows = margin_rows();
    let (mut runs, mut tighter, mut compared, mut kept_runs) = (0usize, 0usize, 0usize, 0usize);
    for fixed_index in [false, true] {
        for seed in 1..=3u64 {
            let input = input_box(seed);
            let graph = attention_block(0x71_7000 + seed, &input, fixed_index);
            let context = format!("fixed_index={fixed_index} seed={seed}");
            // The certified incumbent: the ascent with ONE iteration publishes
            // exactly its first certified fold.
            let (inc_lo, inc_hi) = published(
                &graph,
                &input,
                &config(fixed_index, MarginAlphaSearch::Ascent, 1),
            );
            let before = graph_root_phase_counts();
            let (lo, hi) = published(
                &graph,
                &input,
                &config(fixed_index, MarginAlphaSearch::JointF32, 40),
            );
            let after = graph_root_phase_counts();
            // ENGAGEMENT: a refused phase publishes the
            // incumbent, which satisfies every bound assertion below. Only
            // this test arms the selector in this binary, so the deltas are
            // this run's own.
            assert_eq!(
                (
                    after.entered - before.entered,
                    after.refused - before.refused,
                    after.certified - before.certified,
                    after.alarms - before.alarms,
                ),
                (1, 0, 1, 0),
                "{context}: the joint phase must run, search, and certify ONE candidate \
                 (entered, refused, certified, alarms) — a refusal here is a VOID arm, \
                 not a pass. before={before:?} after={after:?}"
            );
            kept_runs += after.kept - before.kept;
            runs += 1;
            for r in 0..rows.len() {
                assert!(
                    lo[r] >= inc_lo[r] && hi[r] <= inc_hi[r],
                    "{context} row {r}: the joint search published [{}, {}], looser than the \
                     certified incumbent [{}, {}]",
                    lo[r],
                    hi[r],
                    inc_lo[r],
                    inc_hi[r]
                );
                if lo[r] > inc_lo[r] {
                    tighter += 1;
                }
            }
            // Containment against the network itself.
            let lo_in: Vec<f32> = input.lower().iter().copied().collect();
            let hi_in: Vec<f32> = input.upper().iter().copied().collect();
            let mut rng = Lcg(seed ^ 0x5a);
            for sample in 0..(1usize << (T * D)) + 16 {
                let point: Vec<f32> = (0..T * D)
                    .map(|i| {
                        if sample < 1 << (T * D) {
                            if (sample >> i) & 1 == 1 {
                                hi_in[i]
                            } else {
                                lo_in[i]
                            }
                        } else {
                            let t = rng.unit() as f32;
                            (lo_in[i] + (hi_in[i] - lo_in[i]) * t).clamp(lo_in[i], hi_in[i])
                        }
                    })
                    .collect();
                let concrete = BoundedTensor::concrete(
                    ArrayD::from_shape_vec(IxDyn(&[T, D]), point).expect("point"),
                )
                .expect("concrete point");
                let y = graph.propagate_ibp(&concrete).expect("point forward");
                let y_lo: Vec<f64> = y.lower().iter().map(|v| f64::from(*v)).collect();
                let y_hi: Vec<f64> = y.upper().iter().map(|v| f64::from(*v)).collect();
                for (r, row) in rows.iter().enumerate() {
                    let (mut v_lo, mut v_hi, mut magnitude) = (0.0f64, 0.0f64, 0.0f64);
                    for (c, &coefficient) in row.objective.iter().enumerate() {
                        let c64 = f64::from(coefficient);
                        if c64 >= 0.0 {
                            v_lo += c64 * y_lo[c];
                            v_hi += c64 * y_hi[c];
                        } else {
                            v_lo += c64 * y_hi[c];
                            v_hi += c64 * y_lo[c];
                        }
                        magnitude += c64.abs() * y_lo[c].abs().max(y_hi[c].abs());
                    }
                    let tol = 1e-12 * (1.0 + magnitude);
                    assert!(
                        f64::from(lo[r]) <= v_hi + tol,
                        "{context} sample {sample} row {r}: published lower {} above the \
                         network's margin [{v_lo}, {v_hi}] — a false-proof mechanism",
                        lo[r]
                    );
                    assert!(
                        f64::from(hi[r]) >= v_lo - tol,
                        "{context} sample {sample} row {r}: published upper {} below the \
                         network's margin [{v_lo}, {v_hi}]",
                        hi[r]
                    );
                    compared += 1;
                }
            }
        }
    }
    assert_eq!(runs, 6);
    assert!(compared > 0);
    eprintln!(
        "[vit-root-alpha-test] runs={runs} kept={kept_runs} rows_tightened={tighter} \
         containment={compared}"
    );
}
