// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #phase-cost: the per-phase cost census must NEVER change a verdict.
//!
//! The census (`ny_core::phase_cost`) is written at phase boundaries all over
//! the BaB lanes. Its contract is observability only: nothing reads it to make
//! a decision, so the only way it can move a result is through the wall clock
//! it consumes. This binary pins that on deterministic, deadline-free runs of
//! every BaB family the guards sit in (graph ReLU split with and without the
//! root alpha, multi-objective ReLU split, graph input split, and the
//! sequential engine's ReLU split and input split): the result
//! fingerprint (status, domain/depth/cut counts, output-bound BITS) at the
//! DEFAULT level (`coarse`) and at `detail` must equal the fingerprint with the
//! census `off`, run for run.
//!
//! review(cost-census): a BaB result carries `output_bounds` ONLY on a root
//! early exit, and every family above ends at a domain limit or an undecided
//! status, so all nine fingerprints read `bounds=None` and the explored counts
//! saturate at the limit — a coarse-level-dependent 1e-3 loosening of the
//! graph ReLU-split root bound SURVIVED the test (mutant MB). The ROOT-CLOSING
//! families below carry the root bound bits of each lane, and the test now
//! demands that at least two fingerprints carry bounds.
//!
//! It also demands the lever's own observable, so a null cannot be a dark
//! census: at `coarse` the BaB phases MUST be entered, and at `off` nothing may
//! be. It lives in its own integration-test binary so the process-global
//! recorder is exclusively this test's (a lib-test binary runs thousands of
//! other tests on the same recorder concurrently).

use std::time::Duration;

use ndarray::{arr1, arr2};
use ny_core::phase_cost::{self, DetailLevel, PhaseCostSnapshot};
use ny_propagate::layers::{LinearLayer, ReLULayer};
use ny_propagate::{
    BetaCrownConfig, BetaCrownResult, BetaCrownVerifier, BranchingHeuristic, GraphNetwork,
    GraphNode, Layer, Network,
};
use ny_tensor::BoundedTensor;

fn chain(layers: Vec<(&str, Layer)>) -> GraphNetwork {
    let mut graph = GraphNetwork::new();
    let mut prev: Option<String> = None;
    for (name, layer) in layers {
        match &prev {
            None => graph.add_node(GraphNode::from_input(name, layer)),
            Some(p) => graph.add_node(GraphNode::new(name, layer, vec![p.clone()])),
        }
        prev = Some(name.to_string());
    }
    graph.set_output(prev.expect("non-empty chain"));
    graph
}

fn linear(w: ndarray::Array2<f32>, b: Option<ndarray::Array1<f32>>) -> Layer {
    Layer::Linear(LinearLayer::new(w, b).expect("valid linear"))
}

/// 2 -> 2 -> 1, one hidden ReLU layer (the deadline-parity fixture).
fn simple_relu_graph() -> GraphNetwork {
    chain(vec![
        (
            "linear1",
            linear(arr2(&[[1.0_f32, -1.0], [-1.0, 1.0]]), None),
        ),
        ("relu1", Layer::ReLU(ReLULayer)),
        ("linear2", linear(arr2(&[[1.0_f32, 1.0]]), None)),
    ])
}

/// 2 -> 4 -> 4 -> 1, two hidden ReLU layers: enough unstable neurons that the
/// BaB branches, scores candidates and bounds children for several rounds.
fn deeper_relu_graph() -> GraphNetwork {
    chain(vec![
        (
            "fc1",
            linear(
                arr2(&[[1.0_f32, -0.5], [-0.75, 1.0], [0.5, 0.25], [-0.25, -1.0]]),
                Some(arr1(&[0.125_f32, -0.25, 0.0, 0.375])),
            ),
        ),
        ("relu1", Layer::ReLU(ReLULayer)),
        (
            "fc2",
            linear(
                arr2(&[
                    [0.5_f32, -1.0, 0.75, 0.25],
                    [-0.5, 0.5, -0.25, 1.0],
                    [1.0, 0.25, -0.5, -0.75],
                    [-0.25, -0.75, 1.0, 0.5],
                ]),
                Some(arr1(&[0.0_f32, 0.125, -0.125, 0.25])),
            ),
        ),
        ("relu2", Layer::ReLU(ReLULayer)),
        (
            "fc3",
            linear(arr2(&[[1.0_f32, -0.5, 0.75, -1.0]]), Some(arr1(&[0.5_f32]))),
        ),
    ])
}

/// The sequential (non-graph) form of `deeper_relu_graph`: the engine the
/// ACAS-style input-split rows run on.
fn deeper_relu_sequential() -> Network {
    let mut net = Network::new();
    net.add_layer(linear(
        arr2(&[[1.0_f32, -0.5], [-0.75, 1.0], [0.5, 0.25], [-0.25, -1.0]]),
        Some(arr1(&[0.125_f32, -0.25, 0.0, 0.375])),
    ));
    net.add_layer(Layer::ReLU(ReLULayer));
    net.add_layer(linear(
        arr2(&[
            [0.5_f32, -1.0, 0.75, 0.25],
            [-0.5, 0.5, -0.25, 1.0],
            [1.0, 0.25, -0.5, -0.75],
            [-0.25, -0.75, 1.0, 0.5],
        ]),
        Some(arr1(&[0.0_f32, 0.125, -0.125, 0.25])),
    ));
    net.add_layer(Layer::ReLU(ReLULayer));
    net.add_layer(linear(
        arr2(&[[1.0_f32, -0.5, 0.75, -1.0]]),
        Some(arr1(&[0.5_f32])),
    ));
    net
}

fn anti_correlated_graph() -> GraphNetwork {
    chain(vec![
        ("linear1", linear(ndarray::Array2::eye(1), None)),
        ("relu1", Layer::ReLU(ReLULayer)),
        (
            "linear2",
            linear(
                arr2(&[[1.0_f32], [-1.0_f32]]),
                Some(arr1(&[0.5_f32, 0.5_f32])),
            ),
        ),
    ])
}

fn identity_graph() -> GraphNetwork {
    chain(vec![("out", linear(arr2(&[[1.0_f32]]), None))])
}

fn bx(lo: &[f32], hi: &[f32]) -> BoundedTensor {
    BoundedTensor::new(arr1(lo).into_dyn(), arr1(hi).into_dyn()).expect("valid input box")
}

/// Everything a verdict consumer reads, bit-exact; wall time excluded.
fn fingerprint(label: &str, result: &ny_core::Result<BetaCrownResult>) -> String {
    match result {
        Err(error) => format!("{label}: Err({error})"),
        Ok(r) => {
            let bounds = r.output_bounds.as_ref().map(|b| {
                let lo: Vec<u32> = b.lower().iter().map(|v| v.to_bits()).collect();
                let hi: Vec<u32> = b.upper().iter().map(|v| v.to_bits()).collect();
                (lo, hi)
            });
            format!(
                "{label}: status={:?} explored={} verified={} depth={} cuts={} bounds={:?}",
                r.result,
                r.domains_explored,
                r.domains_verified,
                r.max_depth_reached,
                r.cuts_generated,
                bounds
            )
        }
    }
}

fn base_config() -> BetaCrownConfig {
    BetaCrownConfig {
        enable_cuts: false,
        enable_pgd_attack: false,
        timeout: Duration::from_secs(30),
        ..Default::default()
    }
}

/// One deterministic run of every graph BaB family the census instruments.
fn run_all_families() -> Vec<String> {
    let mut prints = Vec::new();

    // ReLU split, branchy (no alpha), the deadline-parity configuration.
    let config = BetaCrownConfig {
        verify_upper_bound: true,
        use_alpha_crown: false,
        max_domains: 2,
        max_depth: 10,
        batch_size: 1,
        ..base_config()
    };
    prints.push(fingerprint(
        "relu_split_branchy",
        &BetaCrownVerifier::new(config).verify_graph_relu_split(
            &simple_relu_graph(),
            &bx(&[-1.0, -1.0], &[1.0, 1.0]),
            &[1.0],
            0.5,
        ),
    ));

    // ReLU split with the root alpha, BaBSR branching and beta, several rounds.
    for (label, threshold) in [("relu_split_deep_a", -0.4_f32), ("relu_split_deep_b", 0.1)] {
        let config = BetaCrownConfig {
            use_alpha_crown: true,
            branching_heuristic: BranchingHeuristic::BoundImpact,
            max_domains: 64,
            max_depth: 12,
            batch_size: 2,
            ..base_config()
        };
        prints.push(fingerprint(
            label,
            &BetaCrownVerifier::new(config).verify_graph_relu_split(
                &deeper_relu_graph(),
                &bx(&[-1.0, -1.0], &[1.0, 1.0]),
                &[1.0],
                threshold,
            ),
        ));
    }

    // Multi-objective ReLU split (disjunctive).
    let config = BetaCrownConfig {
        use_alpha_crown: false,
        max_domains: 100,
        max_depth: 10,
        batch_size: 1,
        ..base_config()
    };
    prints.push(fingerprint(
        "multi_objective",
        &BetaCrownVerifier::new(config).verify_graph_relu_split_multi_objective(
            &anti_correlated_graph(),
            &bx(&[-1.0], &[1.0]),
            &[vec![-1.0_f32, 0.0], vec![0.0, -1.0]],
            &[-0.55, -0.55],
        ),
    ));

    // Multi-objective ReLU split on the deeper net (branch selection + children).
    let config = BetaCrownConfig {
        use_alpha_crown: true,
        max_domains: 64,
        max_depth: 12,
        batch_size: 2,
        ..base_config()
    };
    prints.push(fingerprint(
        "multi_objective_deep",
        &BetaCrownVerifier::new(config).verify_graph_relu_split_multi_objective(
            &deeper_relu_graph(),
            &bx(&[-1.0, -1.0], &[1.0, 1.0]),
            &[vec![1.0_f32], vec![-1.0_f32]],
            &[-0.4, -2.5],
        ),
    ));

    // Input split.
    let config = BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::InputSplit,
        use_alpha_crown: false,
        use_crown_ibp: false,
        enable_relaxed_clip: false,
        input_split_ibp_enhancement: false,
        max_domains: 64,
        max_depth: 1,
        batch_size: 4,
        reorder_bab: false,
        ..base_config()
    };
    prints.push(fingerprint(
        "input_split",
        &BetaCrownVerifier::new(config).verify_graph_input_split(
            &identity_graph(),
            &bx(&[-1.0], &[1.0]),
            &[1.0],
            0.4,
        ),
    ));
    // Sequential engine (verify_impl): ReLU split, then input split.
    for (label, heuristic) in [
        ("sequential_relu_split", BranchingHeuristic::BoundImpact),
        ("sequential_input_split", BranchingHeuristic::InputSplit),
    ] {
        let config = BetaCrownConfig {
            branching_heuristic: heuristic,
            use_alpha_crown: false,
            max_domains: 64,
            max_depth: 10,
            batch_size: 2,
            ..base_config()
        };
        prints.push(fingerprint(
            label,
            &BetaCrownVerifier::new(config).verify(
                &deeper_relu_sequential(),
                &bx(&[-1.0, -1.0], &[1.0, 1.0]),
                -0.4,
            ),
        ));
    }
    let config = BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::InputSplit,
        max_domains: 256,
        max_depth: 8,
        batch_size: 4,
        ..base_config()
    };
    prints.push(fingerprint(
        "input_split_deep",
        &BetaCrownVerifier::new(config).verify_graph_input_split(
            &deeper_relu_graph(),
            &bx(&[-1.0, -1.0], &[1.0, 1.0]),
            &[1.0],
            -0.4,
        ),
    ));

    // review(cost-census): ROOT-CLOSING runs, the only ones whose result
    // carries `output_bounds` (the root bound's bits): verified at the root
    // (threshold far below any reachable output) and violated at the root
    // (far above), on the graph ReLU split with and without the root alpha,
    // the multi-objective split, the graph input split and the sequential
    // engine.
    for (label, alpha, threshold) in [
        ("root_verified_alpha", true, -10.0_f32),
        ("root_verified_plain", false, -10.0),
        ("root_violated_alpha", true, 10.0),
    ] {
        let config = BetaCrownConfig {
            use_alpha_crown: alpha,
            branching_heuristic: BranchingHeuristic::BoundImpact,
            max_domains: 64,
            max_depth: 12,
            batch_size: 2,
            ..base_config()
        };
        prints.push(fingerprint(
            label,
            &BetaCrownVerifier::new(config).verify_graph_relu_split(
                &deeper_relu_graph(),
                &bx(&[-1.0, -1.0], &[1.0, 1.0]),
                &[1.0],
                threshold,
            ),
        ));
    }
    let config = BetaCrownConfig {
        use_alpha_crown: true,
        max_domains: 64,
        max_depth: 12,
        batch_size: 2,
        ..base_config()
    };
    prints.push(fingerprint(
        "multi_objective_root_verified",
        &BetaCrownVerifier::new(config).verify_graph_relu_split_multi_objective(
            &deeper_relu_graph(),
            &bx(&[-1.0, -1.0], &[1.0, 1.0]),
            &[vec![1.0_f32], vec![-1.0_f32]],
            &[-10.0, -10.0],
        ),
    ));
    let config = BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::InputSplit,
        max_domains: 64,
        max_depth: 8,
        batch_size: 4,
        ..base_config()
    };
    prints.push(fingerprint(
        "input_split_root_verified",
        &BetaCrownVerifier::new(config).verify_graph_input_split(
            &deeper_relu_graph(),
            &bx(&[-1.0, -1.0], &[1.0, 1.0]),
            &[1.0],
            -10.0,
        ),
    ));
    let config = BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::BoundImpact,
        use_alpha_crown: true,
        max_domains: 64,
        max_depth: 10,
        batch_size: 2,
        ..base_config()
    };
    prints.push(fingerprint(
        "sequential_root_verified",
        &BetaCrownVerifier::new(config).verify(
            &deeper_relu_sequential(),
            &bx(&[-1.0, -1.0], &[1.0, 1.0]),
            -10.0,
        ),
    ));
    prints
}

fn enters(snapshot: &PhaseCostSnapshot, phase: &str) -> u64 {
    snapshot
        .phases
        .iter()
        .find(|row| row.phase == phase)
        .map_or(0, |row| row.enters)
}

fn counter(snapshot: &PhaseCostSnapshot, name: &str) -> u64 {
    snapshot.counters.get(name).copied().unwrap_or(0)
}

/// The BaB phases every family above passes through.
const ENGAGED: [&str; 4] = ["bab_loop", "root_bounds", "child_bound", "branch_select"];

#[test]
fn phase_cost_levels_leave_every_bab_verdict_bit_identical_and_coarse_engages() {
    // off first (the reference), then the default, then detail, then the
    // default again (a repeat of the reference level catches run-to-run drift
    // that would otherwise read as a level effect).
    let levels = [
        DetailLevel::Off,
        DetailLevel::Coarse,
        DetailLevel::Detail,
        DetailLevel::Off,
        DetailLevel::Coarse,
    ];
    let mut runs = Vec::new();
    for level in levels {
        phase_cost::set_level(level);
        let before = phase_cost::snapshot(false);
        let prints = run_all_families();
        let after = phase_cost::snapshot(false);
        runs.push((level, prints, before, after));
    }
    phase_cost::set_level(phase_cost::DEFAULT_LEVEL);

    let reference = &runs[0].1;
    for (i, line) in reference.iter().enumerate() {
        eprintln!("PHASE_COST_FINGERPRINT {i} {line}");
    }
    assert!(
        reference.iter().all(|line| !line.contains("Err(")),
        "a reference run failed, so the comparison would be vacuous: {reference:#?}"
    );
    assert!(
        reference.iter().any(|line| !line.contains("explored=0 ")),
        "no family explored a domain: the BaB phases were never exercised"
    );
    // review(cost-census): the bound bits are the part of the fingerprint a
    // subtle level-dependent perturbation moves; without them the comparison
    // is status + saturated counts only.
    let with_bounds = reference
        .iter()
        .filter(|line| !line.contains("bounds=None"))
        .count();
    assert!(
        with_bounds >= 2,
        "only {with_bounds} fingerprint(s) carry output-bound bits: {reference:#?}"
    );
    for (level, prints, _, _) in &runs[1..] {
        assert_eq!(
            prints, reference,
            "phase-cost level {level:?} changed a verdict fingerprint"
        );
    }

    for (level, _, before, after) in &runs {
        for phase in ENGAGED {
            let delta = enters(after, phase) - enters(before, phase);
            match level {
                DetailLevel::Off => {
                    assert_eq!(delta, 0, "level off recorded {delta} `{phase}` entries")
                }
                _ => assert!(
                    delta > 0,
                    "level {level:?}: `{phase}` was never entered -- the census is dark"
                ),
            }
        }
        let explored = counter(after, "domains_explored") - counter(before, "domains_explored");
        match level {
            DetailLevel::Off => assert_eq!(explored, 0),
            _ => assert!(explored > 0, "level {level:?}: no domains counted"),
        }
        eprintln!(
            "PHASE_COST_ENGAGED level={} {}",
            level.token(),
            ENGAGED
                .iter()
                .map(|p| format!("{p}=+{}", enters(after, p) - enters(before, p)))
                .collect::<Vec<_>>()
                .join(" ")
        );
    }
    // review(cost-census): `detail` records the FINE sites `coarse` leaves out
    // (the per-domain input-split selection of the sequential engine), so its
    // `branch_select` count must exceed the default level's on the same runs.
    let delta = |i: usize, phase: &str| enters(&runs[i].3, phase) - enters(&runs[i].2, phase);
    assert!(
        delta(2, "branch_select") > delta(1, "branch_select"),
        "detail recorded no fine `branch_select` site beyond coarse ({} vs {})",
        delta(2, "branch_select"),
        delta(1, "branch_select")
    );
    let last = &runs.last().expect("runs").3;
    assert_eq!(last.unbalanced, 0, "a guard was dropped out of order");
    eprintln!("PHASE_COST_SUMMARY {}", last.summary_line());
}
