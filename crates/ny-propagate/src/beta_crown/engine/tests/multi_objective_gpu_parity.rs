// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Parity tests: multi-objective GPU batched path vs CPU sequential path.
//!
//! The multi-objective verify loop uses the GPU batched path when
//! `engine.is_some() && batch_size > 1 && !conjunctive`.
//! These tests ensure the GPU batched path produces results consistent
//! with the CPU sequential baseline (engine=None, batch_size=1).
//!
//! Part of #3397 (GPU CROWN backward).
//! Part of #3872 (cuts-enabled GPU batched coverage).

use super::prelude::*;
use ny_core::NaiveCpuGemmEngine;

/// Build a 4-input, 2-output graph with two ReLU layers.
///
/// Architecture: x(4) → Linear(4→4) → ReLU → Linear(4→4) → ReLU → Linear(4→2) → [Y₀, Y₁]
///
/// This network has 8 ReLU neurons (two layers × 4 neurons), so BaB can split
/// meaningfully. Input bounds [-1,1]⁴ ensure at least some neurons are unstable
/// at the root domain.
fn build_multi_relu_graph_for_gpu_parity() -> (GraphNetwork, BoundedTensor) {
    let w1 = Array2::from_shape_vec(
        (4, 4),
        vec![
            0.5, -0.3, 0.2, 0.1, -0.4, 0.6, -0.1, 0.3, 0.3, 0.2, -0.5, 0.4, -0.1, 0.4, 0.3, -0.6,
        ],
    )
    .unwrap();
    let linear1 = LinearLayer::new(w1, None).unwrap();

    let w2 = Array2::from_shape_vec(
        (4, 4),
        vec![
            0.4, -0.2, 0.3, -0.1, -0.3, 0.5, 0.1, 0.2, 0.2, -0.4, 0.6, -0.3, -0.1, 0.3, -0.2, 0.5,
        ],
    )
    .unwrap();
    let linear2 = LinearLayer::new(w2, None).unwrap();

    let w3 =
        Array2::from_shape_vec((2, 4), vec![0.3, -0.2, 0.4, 0.1, -0.1, 0.5, -0.3, 0.2]).unwrap();
    let linear3 = LinearLayer::new(w3, None).unwrap();

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(linear2),
        vec!["relu1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "relu2",
        Layer::ReLU(ReLULayer),
        vec!["linear2".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear3",
        Layer::Linear(linear3),
        vec!["relu2".to_string()],
    ));
    graph.set_output("linear3");

    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, -1.0, -1.0, -1.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0, 1.0, 1.0]).into_dyn(),
    )
    .unwrap();

    (graph, input)
}

/// GPU batched path must produce the same verification status as CPU sequential.
///
/// Runs the same multi-objective disjunctive verification twice:
/// 1. CPU sequential: engine=None, batch_size=1
/// 2. GPU batched: engine=NaiveCpuGemmEngine, batch_size=4
///
/// Both must agree on the final status (Verified, Unknown, or Timeout).
/// The GPU batched path is triggered by `engine.is_some() && batch_size > 1`.
///
/// This tests the GPU-batched processing path in verify.rs
/// (process_graph_domains_batched_gpu_multi_objective) which was previously
/// untested end-to-end through the full BaB loop.
#[ntest::timeout(60000)]
#[test]
fn test_gpu_batched_multi_objective_matches_cpu_sequential_3397() {
    let (graph, input) = build_multi_relu_graph_for_gpu_parity();

    // Tight thresholds that require BaB exploration but can be verified.
    // Y₀ = w3[0] @ relu2_out, Y₁ = w3[1] @ relu2_out.
    // With small weights and [-1,1] inputs, output range is roughly [-1, 1].
    // Threshold -0.8 checks "is lower bound > -0.8?" which should be verifiable
    // after a few BaB splits tighten the ReLU relaxations.
    let objectives = vec![vec![1.0_f32, 0.0], vec![0.0, 1.0_f32]];
    let thresholds = vec![-0.8_f32, -0.8];

    let timeout = Duration::from_secs(10);
    let max_domains = 500;
    let max_depth = 20;

    // Path 1: CPU sequential (baseline)
    let cpu_config = BetaCrownConfig {
        timeout,
        max_domains,
        max_depth,
        batch_size: 1,
        enable_cuts: false,
        ..Default::default()
    };
    let cpu_result = BetaCrownVerifier::new(cpu_config)
        .verify_graph_relu_split_multi_objective_with_engine(
            &graph,
            &input,
            &objectives,
            &thresholds,
            None,
            None,
        )
        .expect("CPU sequential path should not error");

    // Path 2: GPU batched (engine present, batch_size > 1)
    let gpu_config = BetaCrownConfig {
        timeout,
        max_domains,
        max_depth,
        batch_size: 4,
        enable_cuts: false,
        ..Default::default()
    };
    let engine = NaiveCpuGemmEngine;
    let gpu_result = BetaCrownVerifier::new(gpu_config)
        .verify_graph_relu_split_multi_objective_with_engine(
            &graph,
            &input,
            &objectives,
            &thresholds,
            Some(&engine),
            None,
        )
        .expect("GPU batched path should not error");

    // Both must produce the same verification status category.
    let cpu_status = status_category(&cpu_result.result);
    let gpu_status = status_category(&gpu_result.result);
    assert_eq!(
        cpu_status,
        gpu_status,
        "GPU batched path status ({:?}) must match CPU sequential ({:?}).\n\
         CPU: {:?}, explored={}, verified={}\n\
         GPU: {:?}, explored={}, verified={}",
        gpu_result.result,
        cpu_result.result,
        cpu_result.result,
        cpu_result.domains_explored,
        cpu_result.domains_verified,
        gpu_result.result,
        gpu_result.domains_explored,
        gpu_result.domains_verified,
    );
}

/// GPU batched path with alpha-CROWN must not diverge from CPU sequential.
///
/// Same parity test but with use_alpha_crown=true, which exercises a different
/// initial bounds computation path (alpha-CROWN DAG vs plain CROWN-IBP).
#[ntest::timeout(60000)]
#[test]
fn test_gpu_batched_multi_objective_alpha_crown_parity_3397() {
    let (graph, input) = build_multi_relu_graph_for_gpu_parity();

    let objectives = vec![vec![1.0_f32, 0.0], vec![0.0, 1.0_f32]];
    let thresholds = vec![-0.8_f32, -0.8];

    let timeout = Duration::from_secs(10);

    // CPU sequential with alpha-CROWN
    let cpu_config = BetaCrownConfig {
        timeout,
        max_domains: 500,
        max_depth: 20,
        batch_size: 1,
        use_alpha_crown: true,
        enable_cuts: false,
        ..Default::default()
    };
    let cpu_result = BetaCrownVerifier::new(cpu_config)
        .verify_graph_relu_split_multi_objective_with_engine(
            &graph,
            &input,
            &objectives,
            &thresholds,
            None,
            None,
        )
        .expect("CPU alpha-CROWN should not error");

    // GPU batched with alpha-CROWN
    let gpu_config = BetaCrownConfig {
        timeout,
        max_domains: 500,
        max_depth: 20,
        batch_size: 4,
        use_alpha_crown: true,
        enable_cuts: false,
        ..Default::default()
    };
    let engine = NaiveCpuGemmEngine;
    let gpu_result = BetaCrownVerifier::new(gpu_config)
        .verify_graph_relu_split_multi_objective_with_engine(
            &graph,
            &input,
            &objectives,
            &thresholds,
            Some(&engine),
            None,
        )
        .expect("GPU alpha-CROWN should not error");

    let cpu_status = status_category(&cpu_result.result);
    let gpu_status = status_category(&gpu_result.result);
    assert_eq!(
        cpu_status, gpu_status,
        "GPU+alpha status ({:?}) must match CPU+alpha ({:?})",
        gpu_result.result, cpu_result.result,
    );
}

/// Providing an engine with batch_size=1 should take the sequential path,
/// producing results identical to engine=None with batch_size=1.
///
/// This verifies that the engine parameter alone (without batch_size > 1)
/// does not accidentally trigger the GPU batched codepath.
#[ntest::timeout(60000)]
#[test]
fn test_engine_with_batch_1_uses_sequential_path_3397() {
    let (graph, input) = build_multi_relu_graph_for_gpu_parity();

    let objectives = vec![vec![1.0_f32, 0.0], vec![0.0, 1.0_f32]];
    let thresholds = vec![-0.8_f32, -0.8];

    let config = BetaCrownConfig {
        timeout: Duration::from_secs(10),
        max_domains: 500,
        max_depth: 20,
        batch_size: 1,
        enable_cuts: false,
        ..Default::default()
    };

    let no_engine_result = BetaCrownVerifier::new(config.clone())
        .verify_graph_relu_split_multi_objective_with_engine(
            &graph,
            &input,
            &objectives,
            &thresholds,
            None,
            None,
        )
        .expect("no-engine path should not error");

    let engine = NaiveCpuGemmEngine;
    let with_engine_result = BetaCrownVerifier::new(config)
        .verify_graph_relu_split_multi_objective_with_engine(
            &graph,
            &input,
            &objectives,
            &thresholds,
            Some(&engine),
            None,
        )
        .expect("engine+batch1 path should not error");

    // Same sequential path → same domain count and status.
    assert_eq!(
        status_category(&no_engine_result.result),
        status_category(&with_engine_result.result),
        "batch_size=1 should use sequential path regardless of engine"
    );
    assert_eq!(
        no_engine_result.domains_explored, with_engine_result.domains_explored,
        "same sequential path should explore same number of domains"
    );
}

/// GPU batched path with cuts-enabled must match CPU sequential with cuts-enabled.
///
/// Part of #3813: the GPU batched path now accepts a read-only cut pool and
/// applies existing cuts during CROWN backward propagation. This test verifies
/// that enabling cuts on the GPU batched path produces the same verification
/// status as the CPU sequential path with cuts enabled.
///
/// Acceptance criterion 1 from #3872: exercises the multi-objective GPU batched
/// path with an active GraphCutPool containing real cuts.
#[ntest::timeout(60000)]
#[test]
fn test_gpu_batched_multi_objective_cuts_parity_3872() {
    let (graph, input) = build_multi_relu_graph_for_gpu_parity();

    let objectives = vec![vec![1.0_f32, 0.0], vec![0.0, 1.0_f32]];
    // Threshold -0.2 forces BaB exploration, so cuts are generated from
    // verified sub-domains and applied during subsequent CROWN propagation.
    let thresholds = vec![-0.2_f32, -0.2];

    let timeout = Duration::from_secs(10);
    let max_domains = 1000;
    let max_depth = 20;

    // CPU sequential with cuts (baseline)
    let cpu_config = BetaCrownConfig {
        timeout,
        max_domains,
        max_depth,
        batch_size: 1,
        enable_cuts: true,
        ..Default::default()
    };
    let cpu_result = BetaCrownVerifier::new(cpu_config)
        .verify_graph_relu_split_multi_objective_with_engine(
            &graph,
            &input,
            &objectives,
            &thresholds,
            None,
            None,
        )
        .expect("CPU sequential+cuts should not error");

    // GPU batched with cuts
    let gpu_config = BetaCrownConfig {
        timeout,
        max_domains,
        max_depth,
        batch_size: 4,
        enable_cuts: true,
        ..Default::default()
    };
    let engine = NaiveCpuGemmEngine;
    let gpu_result = BetaCrownVerifier::new(gpu_config)
        .verify_graph_relu_split_multi_objective_with_engine(
            &graph,
            &input,
            &objectives,
            &thresholds,
            Some(&engine),
            None,
        )
        .expect("GPU batched+cuts should not error");

    let cpu_status = status_category(&cpu_result.result);
    let gpu_status = status_category(&gpu_result.result);
    assert_eq!(
        cpu_status,
        gpu_status,
        "GPU batched+cuts status ({:?}) must match CPU sequential+cuts ({:?}).\n\
         CPU: cuts_generated={}, explored={}, verified={}\n\
         GPU: cuts_generated={}, explored={}, verified={}",
        gpu_result.result,
        cpu_result.result,
        cpu_result.cuts_generated,
        cpu_result.domains_explored,
        cpu_result.domains_verified,
        gpu_result.cuts_generated,
        gpu_result.domains_explored,
        gpu_result.domains_verified,
    );
}

/// Run GPU batched multi-objective verification with given config overrides.
///
/// Uses threshold -0.2 to force BaB exploration beyond the root domain,
/// ensuring cuts are actually generated and applied when `enable_cuts=true`.
fn run_gpu_batched_multi_objective(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    enable_cuts: bool,
) -> crate::beta_crown::result::BetaCrownResult {
    let objectives = vec![vec![1.0_f32, 0.0], vec![0.0, 1.0_f32]];
    let thresholds = vec![-0.2_f32, -0.2];
    let config = BetaCrownConfig {
        timeout: Duration::from_secs(10),
        max_domains: 1000,
        max_depth: 20,
        batch_size: 4,
        enable_cuts,
        ..Default::default()
    };
    let engine = NaiveCpuGemmEngine;
    BetaCrownVerifier::new(config)
        .verify_graph_relu_split_multi_objective_with_engine(
            graph,
            input,
            &objectives,
            &thresholds,
            Some(&engine),
            None,
        )
        .expect("GPU batched multi-objective should not error")
}

/// Cuts-enabled GPU batched path must verify at least as well as cuts-disabled.
///
/// Cutting planes can only tighten CROWN relaxations, never loosen them.
/// If the cuts-disabled path verifies, the cuts-enabled path must also verify.
///
/// Acceptance criterion 2 from #3872: bounds with cuts are at least as tight
/// as bounds without cuts.
#[ntest::timeout(60000)]
#[test]
fn test_gpu_batched_multi_objective_cuts_tighten_3872() {
    let (graph, input) = build_multi_relu_graph_for_gpu_parity();

    let no_cuts = run_gpu_batched_multi_objective(&graph, &input, false);
    let with_cuts = run_gpu_batched_multi_objective(&graph, &input, true);

    // Cuts can only tighten bounds, so the outcome must not degrade.
    // BaB exploration order may differ due to cut-tightened bounds changing
    // the priority queue ordering, so exact domain counts may vary.
    // Use status_rank() instead of assert_eq! because cuts may *improve*
    // the outcome (e.g., Unknown → Verified), which is correct behavior
    // that a strict equality check would incorrectly reject.
    assert!(
        status_rank(&with_cuts.result) >= status_rank(&no_cuts.result),
        "Cuts-enabled should not degrade verification outcome.\n\
         No-cuts: {:?} (rank={}), explored={}, verified={}\n\
         Cuts:    {:?} (rank={}), explored={}, verified={}, cuts_generated={}",
        no_cuts.result,
        status_rank(&no_cuts.result),
        no_cuts.domains_explored,
        no_cuts.domains_verified,
        with_cuts.result,
        status_rank(&with_cuts.result),
        with_cuts.domains_explored,
        with_cuts.domains_verified,
        with_cuts.cuts_generated,
    );
}

/// GPU batched path with cuts generates cuts from verified batch children.
///
/// The #3813 outer-loop code in verify.rs (lines 605-611) generates cuts from
/// verified children returned by the batched GPU processing. This test
/// verifies that code path is exercised: cuts_generated must be > 0 when
/// the BaB loop produces verified domains at depth >= min_cut_depth (2).
///
/// Acceptance criterion 3 from #3872: cut generation from verified batch
/// children in the outer loop.
#[ntest::timeout(60000)]
#[test]
fn test_gpu_batched_multi_objective_cut_generation_3872() {
    let (graph, input) = build_multi_relu_graph_for_gpu_parity();

    // Uses the shared helper (threshold -0.2, batch_size=4, enable_cuts=true)
    // which forces BaB exploration beyond the root domain. After 2+ splits,
    // sub-domains verify at depth >= min_cut_depth (2), triggering
    // GraphCutPool::add_from_verified_domain in the outer loop.
    let result = run_gpu_batched_multi_objective(&graph, &input, true);

    // The BaB loop should explore domains deep enough to generate cuts.
    // With 8 ReLU neurons and thresholds -0.2, BaB must split several
    // times before sub-domains verify, producing verified domains at
    // depth >= 2 that feed into GraphCutPool::add_from_verified_domain.
    assert!(
        result.cuts_generated > 0,
        "GPU batched+cuts must generate cuts from verified domains.\n\
         Status: {:?}, explored={}, verified={}, cuts_generated={}.\n\
         If cuts_generated is 0, the outer-loop cut generation code \
         (verify.rs #3813 change) was not exercised.",
        result.result,
        result.domains_explored,
        result.domains_verified,
        result.cuts_generated,
    );

    // Sanity: verify the property was actually explored (not trivially decided).
    assert!(
        result.domains_explored > 1,
        "BaB should explore more than the root domain, got explored={}",
        result.domains_explored,
    );
}

/// Classify BabVerificationStatus into a comparable category.
///
/// The exact reason strings in Unknown variants may differ between paths,
/// so we compare categories rather than exact equality.
fn status_category(status: &BabVerificationStatus) -> &'static str {
    match status {
        BabVerificationStatus::Verified => "Verified",
        BabVerificationStatus::Unknown { .. } => "Unknown",
        BabVerificationStatus::Timeout => "Timeout",
        BabVerificationStatus::Violated { .. } => "Violated",
        BabVerificationStatus::PotentialViolation => "PotentialViolation",
    }
}

/// Numeric rank for verification outcomes, higher = stronger result.
///
/// Used by `cuts_tighten` to assert that enabling cuts does not degrade
/// the verification outcome. Cuts can only tighten CROWN relaxations, so
/// the outcome with cuts should be at least as strong as without cuts.
///
/// Ranking: Verified (3) > Unknown/Timeout (2) > PotentialViolation (1) > Violated (0).
/// Unknown and Timeout are both "inconclusive" and ranked equally.
fn status_rank(status: &BabVerificationStatus) -> u8 {
    match status {
        BabVerificationStatus::Verified => 3,
        BabVerificationStatus::Unknown { .. } | BabVerificationStatus::Timeout => 2,
        BabVerificationStatus::PotentialViolation => 1,
        BabVerificationStatus::Violated { .. } => 0,
    }
}
