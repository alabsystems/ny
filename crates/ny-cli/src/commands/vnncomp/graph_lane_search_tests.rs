// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #graph-lane-search (ENGINE_PROGRAM §2 M7) at the CLI boundary — the only
//! crate that sees BOTH sides of the search/certify split.
//!
//! `real_cpu_f32_graph_lane_search_is_sound_engaged_and_counted`: the REAL
//! `ny_bbe::CpuF32Backend`, registered exactly as `main.rs` does, driven
//! through the graph β-CROWN multi-objective lane (root phase + shared-executor
//! waves on the CPU engine) on random ReLU nets, plain and residual. A
//! property violated at the box centre is never `Verified` (the soundness
//! half), and the lane must ENGAGE — the M7 counters must show armed root
//! phases with certified candidates and armed waves with certified domains,
//! and no alarm — because a refused phase leaves the lane's own bounds and
//! would satisfy the soundness half while testing nothing. With the lane's own
//! root α ascent off (`use_alpha_crown: false`, the incumbent is the adaptive
//! heuristic), the real search must also RAISE at least one certified row.

use std::sync::Arc;

use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_core::NaiveCpuGemmEngine;
use ny_propagate::layers::{AddLayer, LinearLayer, ReLULayer};
use ny_propagate::network::graph_root_search::{
    graph_lane_search_counts, set_graph_root_search_backend_factory,
};
use ny_propagate::{
    BabVerificationStatus, BetaCrownConfig, BetaCrownVerifier, GraphLaneSearch, GraphNetwork,
    GraphNode, Layer,
};
use ny_tensor::BoundedTensor;

const IN: usize = 4;
const HIDDEN: usize = 8;
const CLASSES: usize = 3;

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

fn relu_net(seed: u64, residual: bool) -> GraphNetwork {
    let mut rng = Lcg(seed);
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "l0",
        linear(&mut rng, HIDDEN, IN, 1.0),
    ));
    graph.add_node(GraphNode::new(
        "r0",
        Layer::ReLU(ReLULayer),
        vec!["l0".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "l1",
        linear(&mut rng, HIDDEN, HIDDEN, 0.7),
        vec!["r0".to_string()],
    ));
    let pre = if residual {
        graph.add_node(GraphNode::binary("skip", Layer::Add(AddLayer), "l1", "r0"));
        "skip"
    } else {
        "l1"
    };
    graph.add_node(GraphNode::new(
        "r1",
        Layer::ReLU(ReLULayer),
        vec![pre.to_string()],
    ));
    graph.add_node(GraphNode::new(
        "out",
        linear(&mut rng, CLASSES, HIDDEN, 0.7),
        vec!["r1".to_string()],
    ));
    graph.set_output("out");
    graph
}

fn input_box(seed: u64) -> BoundedTensor {
    let mut rng = Lcg(seed ^ 0x9e37);
    let centre = rng.values(IN, 1.0);
    BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[IN]), centre.iter().map(|c| c - 0.4).collect())
            .expect("lower"),
        ArrayD::from_shape_vec(IxDyn(&[IN]), centre.iter().map(|c| c + 0.4).collect())
            .expect("upper"),
    )
    .expect("input box")
}

/// Margin rows `y_t − y_c` and thresholds just above their values at the box
/// centre: the property is violated there, so no sound run may verify it.
fn sat_property(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    target: usize,
) -> (Vec<Vec<f32>>, Vec<f32>) {
    let centre: Vec<f32> = input
        .lower()
        .iter()
        .zip(input.upper().iter())
        .map(|(l, u)| 0.5 * (l + u))
        .collect();
    let concrete =
        BoundedTensor::concrete(ArrayD::from_shape_vec(IxDyn(&[IN]), centre).expect("centre"))
            .expect("concrete centre");
    let y = graph.propagate_ibp(&concrete).expect("point forward");
    let y: Vec<f64> = y
        .lower()
        .iter()
        .zip(y.upper().iter())
        .map(|(&l, &u)| f64::midpoint(f64::from(l), f64::from(u)))
        .collect();
    let rows: Vec<Vec<f32>> = (0..CLASSES)
        .filter(|&c| c != target)
        .map(|c| {
            let mut row = vec![0.0f32; CLASSES];
            row[target] = 1.0;
            row[c] = -1.0;
            row
        })
        .collect();
    let thresholds = rows
        .iter()
        .map(|row| {
            let v: f64 = row.iter().zip(&y).map(|(&c, &v)| f64::from(c) * v).sum();
            (v + 0.02) as f32
        })
        .collect();
    (rows, thresholds)
}

#[test]
fn real_cpu_f32_graph_lane_search_is_sound_engaged_and_counted() {
    let _env = ny_test_utils::env::lock_env_shared();
    // Exactly the registration `main.rs` performs (first install wins; the test
    // binary never runs `main`, and V2's test registers the same backend).
    set_graph_root_search_backend_factory(|| {
        Some(Arc::new(ny_bbe::CpuF32Backend::with_threads(2))
            as Arc<dyn ny_core::search::SearchBackend>)
    });
    let engine = NaiveCpuGemmEngine;
    // Only this test arms `bab.graph_search` in this binary, so the counter
    // deltas are this test's own.
    let before = graph_lane_search_counts();
    let mut heuristic_raised = 0usize;
    let mut runs = 0usize;
    for use_alpha_crown in [true, false] {
        for seed in 0..4u64 {
            let residual = seed % 2 == 1;
            let input = input_box(seed);
            let graph = relu_net(0x006a_0000 + seed, residual);
            let (rows, thresholds) = sat_property(&graph, &input, (seed % 3) as usize);
            let verifier = BetaCrownVerifier::new(BetaCrownConfig {
                graph_search: GraphLaneSearch::CpuF32,
                use_alpha_crown,
                timeout: std::time::Duration::from_secs(20),
                max_domains: 300,
                max_depth: 12,
                batch_size: 4,
                ..BetaCrownConfig::default()
            });
            let phase_before = graph_lane_search_counts();
            let result = verifier
                .verify_graph_relu_split_multi_objective_with_engine(
                    &graph,
                    &input,
                    &rows,
                    &thresholds,
                    Some(&engine),
                    None,
                )
                .expect("the lane completes");
            let phase_after = graph_lane_search_counts();
            assert!(
                !matches!(result.result, BabVerificationStatus::Verified),
                "use_alpha_crown={use_alpha_crown} seed={seed} residual={residual}: a property \
                 violated at the box centre was VERIFIED — a false-proof mechanism"
            );
            if !use_alpha_crown {
                heuristic_raised += phase_after.rows_raised - phase_before.rows_raised;
            }
            runs += 1;
        }
    }
    let after = graph_lane_search_counts();
    eprintln!("[graph-lane-search-test] runs={runs} before={before:?} after={after:?}");
    assert_eq!(runs, 8);
    assert!(
        after.root_phases - before.root_phases >= 8,
        "every armed run enters the root phase"
    );
    assert!(
        after.root_certified > before.root_certified,
        "no root candidate was certified — the real search never reached the lane (VOID)"
    );
    assert!(
        after.waves > before.waves && after.domains_certified > before.domains_certified,
        "no wave domain was certified — the batched domain search never reached the lane (VOID)"
    );
    assert_eq!(
        after.alarms, before.alarms,
        "a sound lane never raises the disjoint alarm"
    );
    assert!(
        heuristic_raised > 0,
        "against the adaptive-heuristic incumbent the real search must raise at least one \
         certified row"
    );
}
