// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ny_core::NaiveCpuGemmEngine;
use ny_tensor::BoundedTensor;

use crate::beta_crown::config::BetaCrownConfig;
use crate::beta_crown::result::BabVerificationStatus;
use crate::beta_crown::BetaCrownVerifier;
use crate::layers::{FlattenLayer, Layer, LinearLayer, ReLULayer};
use crate::network::{GraphNetwork, GraphNode, Network};
use std::time::Instant;

/// Build a deep+wide Linear/ReLU graph whose root spec-CROWN output-bound
/// backward is dominated by a single very wide `Linear` GEMM (#4321).
///
/// Architecture (all Linear weights are small so bounds stay finite):
///   x[in_dim] -> [Linear(hidden->hidden)+ReLU]*depth -> Linear(hidden->out_dim)
/// with `in_dim == hidden`. The output `Linear(hidden -> out_dim)` is the first
/// node hit on the way back from the `num_specs`-row spec matrix, so its single
/// backward GEMM is `num_specs x out_dim x hidden` — the longest uninterrupted
/// op on the root path. Sized so that pass runs multiple seconds unbounded.
fn build_deep_wide_linear_graph(
    in_dim: usize,
    hidden: usize,
    out_dim: usize,
    depth: usize,
) -> (GraphNetwork, BoundedTensor) {
    assert_eq!(
        in_dim, hidden,
        "first layer maps in_dim->hidden as identity-ish"
    );
    let mut graph = GraphNetwork::new();
    let mut prev = crate::NETWORK_INPUT.to_string();

    let mk_weight = |rows: usize, cols: usize, salt: usize| {
        ndarray::Array2::<f32>::from_shape_fn((rows, cols), |(r, c)| {
            // Tiny, sign-varying coefficients keep IBP/CROWN bounds finite and
            // well-conditioned across many layers.
            0.01 * ((((r * 13 + c * 7 + salt) % 11) as f32) - 5.0)
        })
    };

    for layer_idx in 0..depth {
        let w = mk_weight(hidden, hidden, layer_idx);
        let b = ndarray::Array1::<f32>::from_elem(hidden, 0.001);
        let lin = LinearLayer::new(w, Some(b)).unwrap();
        let lname = format!("lin{layer_idx}");
        if layer_idx == 0 {
            graph.add_node(GraphNode::from_input(&lname, Layer::Linear(lin)));
        } else {
            graph.add_node(GraphNode::new(
                &lname,
                Layer::Linear(lin),
                vec![prev.clone()],
            ));
        }
        let rname = format!("relu{layer_idx}");
        graph.add_node(GraphNode::new(&rname, Layer::ReLU(ReLULayer), vec![lname]));
        prev = rname;
    }

    // Wide output classifier head: hidden -> out_dim.
    let w_out = mk_weight(out_dim, hidden, 9999);
    let b_out = ndarray::Array1::<f32>::from_elem(out_dim, 0.0);
    let lin_out = LinearLayer::new(w_out, Some(b_out)).unwrap();
    graph.add_node(GraphNode::new("out", Layer::Linear(lin_out), vec![prev]));
    graph.set_output("out");

    let input = BoundedTensor::new(
        ndarray::Array1::<f32>::from_elem(in_dim, -0.05).into_dyn(),
        ndarray::Array1::<f32>::from_elem(in_dim, 0.05).into_dyn(),
    )
    .unwrap();

    (graph, input)
}

/// Build a `num_specs`-row spec matrix as `objectives`/`thresholds` over the
/// `out_dim` outputs. Each row selects one output (cycling), with a threshold
/// so loose that no objective can be conclusively decided at the root — forcing
/// the verifier through the full root output-bound backward.
fn build_unverifiable_specs(num_specs: usize, out_dim: usize) -> (Vec<Vec<f32>>, Vec<f32>) {
    let objectives = (0..num_specs)
        .map(|k| {
            let mut row = vec![0.0_f32; out_dim];
            row[k % out_dim] = 1.0;
            row
        })
        .collect();
    // Threshold of -1e9 is below every reachable lower bound, so each objective
    // is "verified" trivially — but we keep PGD/BaB off the table by leaving the
    // expensive root pass as the only work. (Verdict is not what we assert here;
    // we assert prompt self-termination under a short deadline.)
    let thresholds = vec![-1e9_f32; num_specs];
    (objectives, thresholds)
}

/// The DD-zonotope root lane runs before the shared graph bootstrap and can
/// issue Verified directly. A cut-authority request must be rejected at the
/// multi-objective ingress before that otherwise-valid fast verdict.
#[ntest::timeout(10000)]
#[test]
fn dd_zonotope_prebootstrap_early_verified_rejects_cut_authority() {
    let _box_guard = crate::dd_zonotope::certified_box::test_lock();
    crate::dd_zonotope::certified_box::reset_for_test();

    ny_test_utils::env::with_env_edits(|env| {
        env.set("NY_DD_ZONOTOPE", "1");
        env.set("NY_DD_ZONOTOPE_MIN_INPUT", "1");

        // y = flatten(x) + 2 on x ∈ [-1,1], shaped CHW for the DD detector.
        // Its certified lower margin is 1 > 0.
        let mut network = Network::new();
        network.add_layer(Layer::Flatten(FlattenLayer::new(0)));
        network.add_layer(Layer::Linear(
            LinearLayer::new(ndarray::arr2(&[[1.0_f32]]), Some(ndarray::arr1(&[2.0_f32])))
                .expect("one-dimensional affine head"),
        ));
        let graph = GraphNetwork::from_sequential(&network).expect("test graph");
        let input = BoundedTensor::new(
            ndarray::ArrayD::from_elem(ndarray::IxDyn(&[1, 1, 1]), -1.0_f32),
            ndarray::ArrayD::from_elem(ndarray::IxDyn(&[1, 1, 1]), 1.0_f32),
        )
        .expect("CHW input box");
        crate::dd_zonotope::certified_box::register(
            &[-1.0_f32],
            &[1.0_f32],
            crate::dd_zonotope::certified_box::ExactBox {
                lower: vec![-1.0],
                upper: vec![1.0],
                center_hi: vec![0.0],
                center_lo: vec![0.0],
                center_err: vec![0.0],
                half_width: vec![1.0],
            },
        );
        let objectives = vec![vec![1.0_f32]];
        let thresholds = vec![0.0_f32];

        let dd = super::dd_zono_root::run_dd_zono_root(
            &graph,
            &input,
            &objectives,
            None,
            &BetaCrownConfig::default(),
        )
        .expect("fixture must be admitted by the pre-bootstrap DD-zonotope");
        assert!(
            dd.margin.lower_with_safety(
                0,
                crate::dd_zonotope::DdZonoConfig::from_env().safety_factor
            ) > 0.0,
            "fixture must independently establish a DD-zonotope early-Verified verdict"
        );

        let verifier = BetaCrownVerifier::new(BetaCrownConfig {
            enable_cuts: true,
            ..Default::default()
        });
        let error = verifier
            .verify_graph_relu_split_multi_objective(&graph, &input, &objectives, &thresholds)
            .expect_err("DD would verify, but cut authority must reject at ingress");
        assert!(
            error
                .to_string()
                .contains("cut proof authority is quarantined"),
            "expected quarantine error, got {error}"
        );
    });

    crate::dd_zonotope::certified_box::reset_for_test();
}

/// Regression (#4321): a deep+wide Linear/ReLU graph with many output specs must
/// RELIABLY self-terminate at its deadline instead of running a single
/// uninterrupted root output-bound GEMM past the budget (which, in production,
/// gets the process externally killed with no JSON verdict).
///
/// Before the fix, the `Linear` CROWN backward GEMM ran with no internal
/// deadline checkpoint, so a wide classifier-head backward through a deep stack
/// overran the verifier's own timeout. With the deadline threaded and the GEMM
/// row-chunked, the root pass aborts (DeadlineExceeded -> sound IBP fallback ->
/// graceful Timeout/Unknown verdict) well within `deadline + margin`.
///
/// Soundness: a Timeout/Unknown on abort never claims Verified.
#[ntest::timeout(60000)]
#[test]
fn test_deep_wide_linear_root_self_terminates_at_deadline_4321() {
    // Sized so the root pass cannot finish within the short deadline below; the
    // deadline-aware chunks must cut it off promptly.
    let (graph, input) = build_deep_wide_linear_graph(512, 512, 200, 12);
    let (objectives, thresholds) = build_unverifiable_specs(200, 200);

    let deadline_secs = 1u64;
    // Leave enough scheduler slack for loaded CI hosts while remaining well
    // below the measured unbounded root pass this fixture is sized to catch.
    let margin = std::time::Duration::from_secs(8);
    let deadline = Instant::now() + std::time::Duration::from_secs(deadline_secs);

    let config = BetaCrownConfig {
        // Config timeout is generous; the wall-clock `deadline` arg drives budgets.
        timeout: std::time::Duration::from_mins(10),
        use_alpha_crown: false,
        batch_size: 1,
        max_domains: 100_000,
        ..Default::default()
    };

    let start = Instant::now();
    let result = BetaCrownVerifier::new(config)
        .verify_graph_relu_split_multi_objective_with_engine(
            &graph,
            &input,
            &objectives,
            &thresholds,
            None,
            Some(deadline),
        )
        .expect(
            "verify must return a verdict (not error) even when the deadline cuts the root pass",
        );
    let elapsed = start.elapsed();

    // Primary assertion: the verifier returned PROMPTLY rather than running the
    // full unbounded root pass. This is what prevents the external kill.
    assert!(
        elapsed < std::time::Duration::from_secs(deadline_secs) + margin,
        "#4321 root pass must self-terminate near the deadline: elapsed={:?} \
         (deadline={}s + margin={:?})",
        elapsed,
        deadline_secs,
        margin,
    );

    // Soundness: an aborted root pass must never claim Verified for a property
    // that was not actually proved — Timeout/Unknown are the only sound abort
    // verdicts. (Verified is acceptable ONLY if it genuinely finished; with these
    // loose thresholds and a 1s cut, a real proof is not expected.)
    assert!(
        result.result != BabVerificationStatus::Verified,
        "#4321 aborted root pass must not report Verified, got {:?} after {:?}",
        result.result,
        elapsed
    );
}

/// Build a 1-input, 2-output network with one ReLU for conjunctive testing.
///
/// Architecture: x -> Linear(1→1, identity) -> ReLU -> Linear(1→2) -> [Y₀, Y₁]
///
/// Y₀ = max(x, 0) + 0.5, Y₁ = -max(x, 0) + 0.5. Sum = 1.0 (constant).
/// Root CROWN bounds: Y₀ ∈ [0.5, 1.5], Y₁ ∈ [-0.5, 0.5].
///
/// Reference: designs/2026-03-05-joint-conjunctive-bab.md Phase 2 step 4
fn build_single_relu_anti_correlated_graph() -> (GraphNetwork, BoundedTensor) {
    let linear1 = LinearLayer::new(ndarray::Array2::eye(1), None).unwrap();
    let w2 = ndarray::arr2(&[[1.0_f32], [-1.0]]);
    let b2 = ndarray::arr1(&[0.5_f32, 0.5]);
    let linear2 = LinearLayer::new(w2, Some(b2)).unwrap();

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
    graph.set_output("linear2");

    let input = BoundedTensor::new(
        ndarray::arr1(&[-1.0_f32]).into_dyn(),
        ndarray::arr1(&[1.0_f32]).into_dyn(),
    )
    .unwrap();

    (graph, input)
}

#[ntest::timeout(15000)]
#[test]
fn owned_multi_objective_ingress_matches_borrowed_root_result() {
    assert_eq!(
        super::finalized_root_handoff::take_handoff_constructions_for_test(),
        0
    );
    let (graph, input) = build_single_relu_anti_correlated_graph();
    let objectives = vec![vec![1.0_f32, 0.0], vec![0.0_f32, 1.0]];
    let thresholds = vec![-1.0_f32, -1.0];
    let config = BetaCrownConfig {
        use_alpha_crown: false,
        use_crown_ibp: false,
        enable_pgd_attack: false,
        enable_cuts: false,
        batch_size: 1,
        ..Default::default()
    };

    let borrowed = BetaCrownVerifier::new(config.clone())
        .verify_graph_relu_split_multi_objective_with_engine(
            &graph,
            &input,
            &objectives,
            &thresholds,
            None,
            None,
        )
        .expect("borrowed verifier result");
    assert_eq!(
        super::finalized_root_handoff::take_handoff_constructions_for_test(),
        0,
        "a borrowed root-finished call must not construct owned custody"
    );
    let owned = BetaCrownVerifier::new(config)
        .verify_graph_relu_split_multi_objective_owned_with_engine(
            &graph,
            &input,
            crate::OwnedSignNormalizedObjectiveSet::new(objectives, thresholds),
            None,
            None,
        )
        .expect("owned verifier result");
    assert_eq!(
        super::finalized_root_handoff::take_handoff_constructions_for_test(),
        0,
        "an owned root-finished call must not construct continuing custody"
    );

    assert_eq!(owned.result, borrowed.result);
    assert_eq!(owned.domains_explored, borrowed.domains_explored);
    assert_eq!(owned.max_depth_reached, borrowed.max_depth_reached);
    assert_eq!(owned.cuts_generated, borrowed.cuts_generated);
    assert_eq!(owned.domains_verified, borrowed.domains_verified);
    match (&owned.output_bounds, &borrowed.output_bounds) {
        (Some(owned), Some(borrowed)) => {
            assert_eq!(owned.shape(), borrowed.shape());
            assert!(owned
                .lower()
                .iter()
                .zip(borrowed.lower())
                .all(|(left, right)| left.to_bits() == right.to_bits()));
            assert!(owned
                .upper()
                .iter()
                .zip(borrowed.upper())
                .all(|(left, right)| left.to_bits() == right.to_bits()));
        }
        (None, None) => {}
        _ => panic!("owned and borrowed results must carry the same output bounds"),
    }
}

#[ntest::timeout(15000)]
#[test]
fn owned_continuing_ingress_crosses_one_handoff_and_preserves_parity() {
    assert_eq!(
        super::finalized_root_handoff::take_handoff_constructions_for_test(),
        0
    );
    let (graph, input) = build_single_relu_anti_correlated_graph();
    // Both identical rows remain unresolved at the root: Y0 spans [0.5, 1.5]
    // against 0.75, so this fixture must cross the finalized-root Continue seam.
    let objectives = vec![vec![1.0_f32, 0.0], vec![1.0_f32, 0.0]];
    let thresholds = vec![0.75_f32, 0.75];
    let config = BetaCrownConfig {
        timeout: std::time::Duration::from_secs(5),
        use_alpha_crown: false,
        use_crown_ibp: false,
        enable_pgd_attack: false,
        enable_cuts: false,
        batch_size: 1,
        ..Default::default()
    };

    let borrowed = BetaCrownVerifier::new(config.clone())
        .verify_graph_relu_split_multi_objective_with_engine(
            &graph,
            &input,
            &objectives,
            &thresholds,
            None,
            None,
        )
        .expect("borrowed continuing result");
    assert_eq!(
        super::finalized_root_handoff::take_handoff_constructions_for_test(),
        0,
        "borrowed Continue must bypass the owned handoff"
    );

    let owned = BetaCrownVerifier::new(config)
        .verify_graph_relu_split_multi_objective_owned_with_engine(
            &graph,
            &input,
            crate::OwnedSignNormalizedObjectiveSet::new(objectives, thresholds),
            None,
            None,
        )
        .expect("owned continuing result");
    assert_eq!(
        super::finalized_root_handoff::take_handoff_constructions_for_test(),
        1,
        "owned Continue must cross exactly one finalized-root handoff"
    );

    assert_eq!(owned.result, borrowed.result);
    assert_eq!(owned.domains_explored, borrowed.domains_explored);
    assert_eq!(owned.max_depth_reached, borrowed.max_depth_reached);
    assert_eq!(owned.cuts_generated, borrowed.cuts_generated);
    assert_eq!(owned.domains_verified, borrowed.domains_verified);
    match (&owned.output_bounds, &borrowed.output_bounds) {
        (Some(owned), Some(borrowed)) => {
            assert_eq!(owned.shape(), borrowed.shape());
            assert!(owned
                .lower()
                .iter()
                .zip(borrowed.lower())
                .all(|(left, right)| left.to_bits() == right.to_bits()));
            assert!(owned
                .upper()
                .iter()
                .zip(borrowed.upper())
                .all(|(left, right)| left.to_bits() == right.to_bits()));
        }
        (None, None) => {}
        _ => panic!("owned and borrowed continuing results must preserve output bounds"),
    }
}

#[ntest::timeout(15000)]
#[test]
fn owned_terminal_timeout_never_constructs_continuing_handoff() {
    assert_eq!(
        super::finalized_root_handoff::take_handoff_constructions_for_test(),
        0
    );
    let (graph, input) = build_single_relu_anti_correlated_graph();
    let expired = Instant::now()
        .checked_sub(std::time::Duration::from_millis(1))
        .expect("Instant subtraction");
    let result = BetaCrownVerifier::new(BetaCrownConfig {
        timeout: std::time::Duration::from_secs(5),
        use_alpha_crown: false,
        use_crown_ibp: false,
        enable_pgd_attack: false,
        enable_cuts: false,
        batch_size: 1,
        ..Default::default()
    })
    .verify_graph_relu_split_multi_objective_owned_with_engine(
        &graph,
        &input,
        crate::OwnedSignNormalizedObjectiveSet::new(
            vec![vec![1.0_f32, 0.0], vec![1.0_f32, 0.0]],
            vec![0.75_f32, 0.75],
        ),
        None,
        Some(expired),
    )
    .expect("expired root authority returns an ordinary terminal result");

    assert!(matches!(
        result.result,
        BabVerificationStatus::Timeout | BabVerificationStatus::Unknown { .. }
    ));
    assert_eq!(
        super::finalized_root_handoff::take_handoff_constructions_for_test(),
        0,
        "a terminal timeout must drop custody before the Continue-only handoff"
    );
}

#[test]
fn owned_multi_objective_ingress_preserves_borrowed_validation() {
    let (graph, input) = build_single_relu_anti_correlated_graph();
    let objectives = vec![vec![1.0_f32, 0.0], vec![0.0_f32, 1.0]];
    let thresholds = vec![0.0_f32];
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());

    let borrowed_error = verifier
        .verify_graph_relu_split_multi_objective_with_engine(
            &graph,
            &input,
            &objectives,
            &thresholds,
            None,
            None,
        )
        .expect_err("borrowed mismatch must fail");
    let owned_error = verifier
        .verify_graph_relu_split_multi_objective_owned_with_engine(
            &graph,
            &input,
            crate::OwnedSignNormalizedObjectiveSet::new(objectives, thresholds),
            None,
            None,
        )
        .expect_err("owned mismatch must fail");

    assert_eq!(owned_error.to_string(), borrowed_error.to_string());
}

fn assert_output_bounds_match_ibp(actual: &BoundedTensor, expected: &BoundedTensor) {
    assert_eq!(
        actual.lower().iter().copied().collect::<Vec<_>>(),
        expected.lower().iter().copied().collect::<Vec<_>>(),
        "expired warmup fallback must reuse direct IBP lower bounds"
    );
    assert_eq!(
        actual.upper().iter().copied().collect::<Vec<_>>(),
        expected.upper().iter().copied().collect::<Vec<_>>(),
        "expired warmup fallback must reuse direct IBP upper bounds"
    );
}

fn expired_deadline_bootstrap_for_fallback(
    verifier: &BetaCrownVerifier,
    graph: &GraphNetwork,
    input: &BoundedTensor,
) -> crate::beta_crown::engine::graph::shared::init::GraphBabBootstrap {
    use std::time::{Duration, Instant};

    use crate::beta_crown::engine::graph::shared::init::compute_graph_bab_bootstrap;

    let mut bootstrap = compute_graph_bab_bootstrap(graph, input, &verifier.config, None, None)
        .expect("warmup bootstrap should build before its phase checkpoint expires");
    bootstrap.alpha_config.deadline = Some(
        Instant::now()
            .checked_sub(Duration::from_millis(1))
            .unwrap(),
    );
    // Force the exact `Err(_) => fallback CROWN output` branch by making the
    // spec-guided request miss its required pre-activation cache.
    bootstrap.initial_node_bounds.clear();
    bootstrap
}

#[ntest::timeout(10000)]
#[test]
fn test_root_spec_cache_captures_branch_node_rows_3813() {
    let (graph, input) = build_single_relu_anti_correlated_graph();
    let node_bounds = graph
        .collect_node_bounds(&input)
        .expect("toy graph node bounds should collect");
    let spec_matrix = ndarray::arr2(&[[1.0_f32, 0.0], [0.0, 1.0_f32]]);

    let (_bounds, cache) = graph
        .propagate_crown_with_specs_and_node_bounds_and_cache_and_deadline(
            &input,
            &spec_matrix,
            None,
            &node_bounds,
            None,
        )
        .expect("spec-guided cache capture should succeed on the toy graph");
    let cache = cache.expect("spec-guided root pass should capture per-node lA cache");

    assert!(
        cache.linear_bounds("relu1").is_some(),
        "root cache must contain the branch node so children can warm-start there"
    );

    let per_objective = cache
        .split_multi_row(2)
        .expect("two-objective spec cache should split into per-objective rows");
    assert_eq!(per_objective.len(), 2);
    assert!(
        per_objective
            .iter()
            .all(|objective_cache| objective_cache.linear_bounds("relu1").is_some()),
        "each split cache must preserve the branch-node rows"
    );
}

#[test]
fn unchanged_root_boxes_keep_historical_spec_request() {
    use crate::beta_crown::engine::graph::shared::init::compute_graph_bab_bootstrap;
    use crate::network::SpecCrownRequest;

    let (graph, input) = build_single_relu_anti_correlated_graph();
    let objectives = vec![vec![1.0_f32, 0.0], vec![0.0, 1.0_f32]];
    let spec_matrix = ndarray::arr2(&[[1.0_f32, 0.0], [0.0, 1.0_f32]]);
    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        use_alpha_crown: false,
        ..Default::default()
    });
    let bootstrap = compute_graph_bab_bootstrap(&graph, &input, &verifier.config, None, None)
        .expect("toy bootstrap should succeed");

    // This is the exact pre-change request. With no intermediate endpoint
    // tightened, the production helper must preserve its row bounds and cache.
    let (legacy_bounds, legacy_cache) = SpecCrownRequest::new(&graph, &input, &spec_matrix, None)
        .node_bounds(&bootstrap.initial_node_bounds)
        .alpha_state_opt(bootstrap.root_alpha_state.as_ref())
        .deadline_opt(bootstrap.alpha_config.deadline)
        .truncate_after_opt(verifier.config.crown_backward_layers)
        .capture_cache()
        .run_with_cache()
        .expect("legacy spec request should succeed");
    let expected: Vec<(f32, f32)> = legacy_bounds
        .lower()
        .iter()
        .zip(legacy_bounds.upper())
        .map(|(&lower, &upper)| (lower, upper))
        .collect();

    let evaluation = super::root::compute_root_objective_bounds(
        &verifier,
        &graph,
        &input,
        &objectives,
        &[1.0e9, 1.0e9],
        false,
        None,
        &bootstrap,
        None,
        None,
        false,
        false,
    )
    .expect("unchanged-box root evaluation should succeed");

    assert_eq!(evaluation.initial_obj_bounds, expected);
    assert_eq!(
        evaluation.root_spec_cache.as_ref().map(|cache| cache.len()),
        legacy_cache.as_ref().map(|cache| cache.len())
    );
}

/// Conjunctive mode returns Verified when any single objective is verified at root.
///
/// Network: Y₀ = max(x,0)+0.5 ∈ [0.5,1.5], Y₁ = -max(x,0)+0.5 ∈ [-0.5,0.5].
/// Y₀ + Y₁ = 1.0, so "Y₀ > 0.55 AND Y₁ > 0.55" requires sum > 1.1 > 1 → impossible.
///
/// Objectives (negations): [-1,0] threshold -0.55 checks -Y₀ > -0.55 (Y₀ < 0.55).
///                          [0,-1] threshold -0.55 checks -Y₁ > -0.55 (Y₁ < 0.55).
///
/// Root CROWN: Obj2 lower = -upper(Y₁) = -0.5 > -0.55 → Obj2 verified.
/// Conjunctive any_verified → Verified at root. No BaB needed.
///
/// Reference: alpha-beta-CROWN stop_criterion_batch_any (auto_LiRPA/utils.py:107-113)
#[ntest::timeout(10000)]
#[test]
fn test_conjunctive_root_any_verified_returns_verified_3334() {
    let (graph, input) = build_single_relu_anti_correlated_graph();

    let objectives = vec![vec![-1.0_f32, 0.0], vec![0.0, -1.0_f32]];
    let thresholds = vec![-0.55_f32, -0.55];

    let config = BetaCrownConfig {
        timeout: std::time::Duration::from_secs(5),
        max_domains: 100,
        max_depth: 10,
        batch_size: 1,
        ..Default::default()
    };

    let result = BetaCrownVerifier::new(config)
        .verify_graph_relu_split_multi_objective_conjunctive_with_engine(
            &graph,
            &input,
            &objectives,
            &thresholds,
            None,
            None,
        )
        .expect("conjunctive BaB should complete without error");

    assert!(
        matches!(result.result, BabVerificationStatus::Verified),
        "conjunctive mode should return Verified when any objective verified at root, got {:?}",
        result.result
    );
    assert_eq!(
        result.domains_explored, 1,
        "expected root-only verification (1 domain explored)"
    );
}

/// Disjunctive mode does NOT return Verified when only one objective is verified at root.
///
/// Same network and objectives as the conjunctive test. Disjunctive requires ALL
/// objectives verified, so it enters BaB and returns Unknown (active subdomain
/// has loose CROWN bounds that can't verify Obj1).
#[ntest::timeout(10000)]
#[test]
fn test_disjunctive_root_partial_verified_enters_bab_3334() {
    let (graph, input) = build_single_relu_anti_correlated_graph();

    let objectives = vec![vec![-1.0_f32, 0.0], vec![0.0, -1.0_f32]];
    let thresholds = vec![-0.55_f32, -0.55];

    let config = BetaCrownConfig {
        timeout: std::time::Duration::from_secs(5),
        max_domains: 100,
        max_depth: 10,
        batch_size: 1,
        ..Default::default()
    };

    let result = BetaCrownVerifier::new(config)
        .verify_graph_relu_split_multi_objective_with_engine(
            &graph,
            &input,
            &objectives,
            &thresholds,
            None,
            None,
        )
        .expect("disjunctive BaB should complete without error");

    assert!(
        matches!(result.result, BabVerificationStatus::Unknown { .. }),
        "disjunctive mode should return Unknown (only 1 of 2 objectives verified at root), got {:?}",
        result.result
    );
    assert!(
        result.domains_explored > 1,
        "disjunctive should enter BaB (expected >1 domains), got {}",
        result.domains_explored
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_disjunctive_engine_batch_path_matches_no_engine_4398() {
    let (graph, input) = build_single_relu_anti_correlated_graph();

    let objectives = vec![vec![-1.0_f32, 0.0], vec![0.0, -1.0_f32]];
    let thresholds = vec![-0.55_f32, -0.55];

    let config = BetaCrownConfig {
        timeout: std::time::Duration::from_secs(5),
        max_domains: 100,
        max_depth: 10,
        batch_size: 4,
        ..Default::default()
    };

    let verifier = BetaCrownVerifier::new(config);
    let cpu_engine = NaiveCpuGemmEngine;

    let baseline = verifier
        .verify_graph_relu_split_multi_objective_with_engine(
            &graph,
            &input,
            &objectives,
            &thresholds,
            None,
            None,
        )
        .expect("baseline multi-objective BaB should complete without error");
    let batched = verifier
        .verify_graph_relu_split_multi_objective_with_engine(
            &graph,
            &input,
            &objectives,
            &thresholds,
            Some(&cpu_engine),
            None,
        )
        .expect("engine-backed multi-objective BaB should complete without error");

    assert_eq!(
        std::mem::discriminant(&batched.result),
        std::mem::discriminant(&baseline.result),
        "shared batch executor should preserve the multi-objective result kind: baseline={:?}, batched={:?}",
        baseline.result,
        batched.result
    );
    assert_eq!(
        batched.domains_explored, baseline.domains_explored,
        "shared batch executor should preserve explored-domain accounting"
    );
    assert_eq!(
        batched.domains_verified, baseline.domains_verified,
        "shared batch executor should preserve verified-domain accounting"
    );
}

/// Conjunctive all_violated at root returns Unknown, not Verified.
///
/// All objectives have upper < threshold at root → all constraints might hold →
/// conjunction might hold → NOT safe. Conjunctive mode returns Unknown.
#[ntest::timeout(10000)]
#[test]
fn test_conjunctive_all_violated_root_returns_unknown_3334() {
    let (graph, input) = build_single_relu_anti_correlated_graph();

    let objectives = vec![vec![1.0_f32, 0.0], vec![0.0, 1.0_f32]];
    let thresholds = vec![2.0_f32, 2.0];

    let config = BetaCrownConfig {
        timeout: std::time::Duration::from_secs(5),
        max_domains: 100,
        max_depth: 10,
        batch_size: 1,
        ..Default::default()
    };

    let result = BetaCrownVerifier::new(config)
        .verify_graph_relu_split_multi_objective_conjunctive_with_engine(
            &graph,
            &input,
            &objectives,
            &thresholds,
            None,
            None,
        )
        .expect("conjunctive BaB should complete without error");

    assert!(
        matches!(result.result, BabVerificationStatus::Unknown { .. }),
        "conjunctive mode should return Unknown when all violated at root, got {:?}",
        result.result
    );
}

/// Regression for #2266: empty objectives must be rejected, not trivially verified.
#[ntest::timeout(5000)]
#[test]
fn test_empty_objectives_returns_error_2266() {
    let config = BetaCrownConfig::default();
    let verifier = BetaCrownVerifier::new(config);

    let weight = ndarray::Array2::from_shape_vec((2, 2), vec![1.0f32, 0.0, 0.0, 1.0]).unwrap();
    let layer = LinearLayer::new(weight, None).unwrap();
    let mut network = Network::new();
    network.add_layer(Layer::Linear(layer));
    let graph = GraphNetwork::from_sequential(&network).unwrap();

    let input = BoundedTensor::new(
        ndarray::arr1(&[-1.0f32, -1.0]).into_dyn(),
        ndarray::arr1(&[1.0f32, 1.0]).into_dyn(),
    )
    .unwrap();

    let result = verifier.verify_graph_relu_split_multi_objective(&graph, &input, &[], &[]);
    assert!(
        result.is_err(),
        "Empty objectives must be rejected, not trivially verified"
    );
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("empty"),
        "Error should mention empty objectives, got: {err_msg}"
    );
}

/// Graph multi-objective stopping rules are intentionally lower-bound-only.
/// Before the ingress guard, this fixture reached the hard-coded
/// `lower > threshold` root close and returned a false `Verified`: in upper
/// mode the requested proof is instead `upper < threshold`, while the exact
/// constant output is 1 and the threshold is 0.
#[ntest::timeout(5000)]
#[test]
fn upper_mode_constant_counterexample_is_rejected_before_false_root_verified() {
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(ndarray::arr2(&[[0.0_f32]]), Some(ndarray::arr1(&[1.0_f32])))
            .expect("constant affine output"),
    ));
    let graph = GraphNetwork::from_sequential(&network).expect("constant graph");
    let input = BoundedTensor::new(
        ndarray::arr1(&[0.0_f32]).into_dyn(),
        ndarray::arr1(&[0.0_f32]).into_dyn(),
    )
    .expect("point input");
    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        verify_upper_bound: true,
        ..Default::default()
    });

    let error = verifier
        .verify_graph_relu_split_multi_objective(&graph, &input, &[vec![1.0]], &[0.0])
        .expect_err("upper mode must be refused before lower-bound root authority");
    assert!(
        error
            .to_string()
            .contains("requires sign-normalized lower-bound objectives"),
        "unexpected refusal: {error}"
    );
}

/// #3383: mismatched objectives/thresholds lengths must return error, not silently truncate.
///
/// .zip() truncates to the shorter iterator. Without the entry-point guard,
/// disjunctive mode would skip unchecked objectives (conservative but wrong),
/// and conjunctive mode could be unsound (all_violated on truncated set).
#[ntest::timeout(5000)]
#[test]
fn test_objectives_thresholds_length_mismatch_returns_error_3383() {
    let config = BetaCrownConfig::default();
    let verifier = BetaCrownVerifier::new(config);

    let weight = ndarray::Array2::from_shape_vec((2, 2), vec![1.0f32, 0.0, 0.0, 1.0]).unwrap();
    let layer = LinearLayer::new(weight, None).unwrap();
    let mut network = Network::new();
    network.add_layer(Layer::Linear(layer));
    let graph = GraphNetwork::from_sequential(&network).unwrap();

    let input = BoundedTensor::new(
        ndarray::arr1(&[-1.0f32, -1.0]).into_dyn(),
        ndarray::arr1(&[1.0f32, 1.0]).into_dyn(),
    )
    .unwrap();

    let objectives = vec![vec![1.0f32, 0.0], vec![0.0, 1.0f32]];
    let thresholds = vec![0.0f32];

    let result =
        verifier.verify_graph_relu_split_multi_objective(&graph, &input, &objectives, &thresholds);
    assert!(
        result.is_err(),
        "objectives/thresholds length mismatch must return Err, got {:?}",
        result
    );
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("mismatch"),
        "Error should mention mismatch, got: {err_msg}"
    );

    let result = verifier.verify_graph_relu_split_multi_objective_conjunctive_with_engine(
        &graph,
        &input,
        &objectives,
        &thresholds,
        None,
        None,
    );
    assert!(
        result.is_err(),
        "conjunctive: objectives/thresholds length mismatch must return Err, got {:?}",
        result
    );
}

/// Build a multi-ReLU network that creates a deep BaB tree.
///
/// Architecture: x(4) -> Linear(4→4) -> ReLU -> Linear(4→4) -> ReLU -> Linear(4→2) -> [Y₀, Y₁]
///
/// Input bounds: [-1, 1]⁴. With 8 ReLU neurons across two layers, the BaB tree
/// has up to 2⁸ = 256 leaf domains. Large enough to keep BaB busy but small enough
/// to construct quickly in a unit test.
fn build_multi_relu_graph() -> (GraphNetwork, BoundedTensor) {
    let w1 = ndarray::Array2::from_shape_vec(
        (4, 4),
        vec![
            0.5, -0.3, 0.2, 0.1, -0.4, 0.6, -0.1, 0.3, 0.3, 0.2, -0.5, 0.4, -0.1, 0.4, 0.3, -0.6,
        ],
    )
    .unwrap();
    let linear1 = LinearLayer::new(w1, None).unwrap();

    let w2 = ndarray::Array2::from_shape_vec(
        (4, 4),
        vec![
            0.4, -0.2, 0.3, -0.1, -0.3, 0.5, 0.1, 0.2, 0.2, -0.4, 0.6, -0.3, -0.1, 0.3, -0.2, 0.5,
        ],
    )
    .unwrap();
    let linear2 = LinearLayer::new(w2, None).unwrap();

    let w3 =
        ndarray::Array2::from_shape_vec((2, 4), vec![0.3, -0.2, 0.4, 0.1, -0.1, 0.5, -0.3, 0.2])
            .unwrap();
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
        ndarray::arr1(&[-1.0_f32, -1.0, -1.0, -1.0]).into_dyn(),
        ndarray::arr1(&[1.0_f32, 1.0, 1.0, 1.0]).into_dyn(),
    )
    .unwrap();

    (graph, input)
}

/// Regression test for #3388: multi-objective conjunctive BaB must respect timeout.
///
/// Without the per-domain deadline check in sequential processing, the BaB loop
/// only checks the timeout between batches. For large networks with many
/// objectives, a single batch of sequential CROWN passes can exceed the timeout
/// budget, causing the process to hang indefinitely.
///
/// This test sets a very short timeout (1s) and verifies the BaB loop returns
/// Timeout or completes within a reasonable wall-clock time (8s).
#[ntest::timeout(10000)]
#[test]
fn test_conjunctive_bab_respects_timeout_3388() {
    let (graph, input) = build_multi_relu_graph();

    let objectives = vec![vec![1.0_f32, 0.0], vec![0.0, 1.0_f32]];
    let thresholds = vec![100.0_f32, 100.0];

    let config = BetaCrownConfig {
        timeout: std::time::Duration::from_secs(1),
        max_domains: 100_000,
        max_depth: 100,
        batch_size: 64,
        ..Default::default()
    };

    let start = Instant::now();
    let result = BetaCrownVerifier::new(config)
        .verify_graph_relu_split_multi_objective_conjunctive_with_engine(
            &graph,
            &input,
            &objectives,
            &thresholds,
            None,
            None,
        )
        .expect("conjunctive BaB should complete without error");

    let elapsed = start.elapsed();

    assert!(
        matches!(
            result.result,
            BabVerificationStatus::Timeout | BabVerificationStatus::Unknown { .. }
        ),
        "expected Timeout or Unknown with impossible thresholds, got {:?}",
        result.result
    );
    assert!(
        elapsed.as_secs() < 8,
        "BaB should respect 1s timeout (took {:.1}s) — per-domain deadline check missing?",
        elapsed.as_secs_f64()
    );
}

/// Regression test for #3388: disjunctive multi-objective BaB also respects timeout.
///
/// Same scenario as the conjunctive test but with disjunctive semantics.
/// Disjunctive mode also uses the sequential path for small networks
/// (batch_size=1 or no GPU engine).
#[ntest::timeout(10000)]
#[test]
fn test_disjunctive_bab_respects_timeout_3388() {
    let (graph, input) = build_multi_relu_graph();

    let objectives = vec![vec![1.0_f32, 0.0], vec![0.0, 1.0_f32]];
    let thresholds = vec![100.0_f32, 100.0];

    let config = BetaCrownConfig {
        timeout: std::time::Duration::from_secs(1),
        max_domains: 100_000,
        max_depth: 100,
        batch_size: 64,
        ..Default::default()
    };

    let start = Instant::now();
    let result = BetaCrownVerifier::new(config)
        .verify_graph_relu_split_multi_objective_with_engine(
            &graph,
            &input,
            &objectives,
            &thresholds,
            None,
            None,
        )
        .expect("disjunctive BaB should complete without error");

    let elapsed = start.elapsed();

    assert!(
        matches!(
            result.result,
            BabVerificationStatus::Timeout | BabVerificationStatus::Unknown { .. }
        ),
        "expected Timeout or Unknown with impossible thresholds, got {:?}",
        result.result
    );
    assert!(
        elapsed.as_secs() < 8,
        "BaB should respect 1s timeout (took {:.1}s)",
        elapsed.as_secs_f64()
    );
}

/// Regression for #4260: multi-objective root fallback must respect the warmup
/// deadline from `#4095`. With `initial_bounds_fraction: 0.0`, the capped
/// deadline expires instantly. The non-alpha fallback branches must use
/// `bootstrap.alpha_config.deadline` (the capped value), not `None`.
///
/// Without the fix, the fallback calls
/// `crown_backward_with_relaxation_and_deadline_and_truncation(..., None, ...)`
/// which runs uncapped root CROWN. With the fix, the expired deadline triggers
/// an early bail and the result matches IBP-level bounds.
#[ntest::timeout(10000)]
#[test]
fn test_multi_objective_root_fallback_respects_warmup_deadline_4260() {
    use std::time::Duration;

    let (graph, input) = build_single_relu_anti_correlated_graph();
    let objectives = vec![vec![1.0_f32, 0.0], vec![0.0, 1.0_f32]];
    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        timeout: Duration::from_secs(2),
        use_alpha_crown: false,
        ..Default::default()
    });
    let bootstrap = expired_deadline_bootstrap_for_fallback(&verifier, &graph, &input);

    // The local warmup checkpoint is expired, but the outer verifier still
    // owns a short live budget.  The root phase may rebase only onto that
    // explicit authority, and its typed Dense-ReLU refusal must use the local
    // deadline-aware IBP fallback rather than uncapped CROWN.
    let global_deadline = Some(Instant::now() + Duration::from_secs(1));
    let evaluation = super::root::compute_root_objective_bounds(
        &verifier,
        &graph,
        &input,
        &objectives,
        &[1.0e9, 1.0e9],
        false,
        None,
        &bootstrap,
        global_deadline,
        None,
        false,
        false,
    )
    .expect("spec-guided failure should fall back to bounded root-output CROWN");

    let ibp_output = graph
        .propagate_ibp(&input)
        .expect("direct IBP output should succeed on the toy graph");
    let expected_obj_bounds = BetaCrownVerifier::objective_bounds_multi(&ibp_output, &objectives)
        .expect("IBP objective bounds should succeed on the toy graph");

    assert!(
        evaluation.root_spec_cache.is_none(),
        "fallback branch should not capture a spec-guided root cache"
    );
    assert_output_bounds_match_ibp(&evaluation.initial_output, &ibp_output);
    assert!(
        evaluation.initial_obj_bounds == expected_obj_bounds,
        "expired warmup fallback must match IBP objective bounds: actual={:?} expected={:?}",
        evaluation.initial_obj_bounds,
        expected_obj_bounds
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_multi_objective_root_fallback_expired_global_authority_is_terminal_4260() {
    use std::time::{Duration, Instant};

    let (graph, input) = build_single_relu_anti_correlated_graph();
    let objectives = vec![vec![1.0_f32, 0.0], vec![0.0, 1.0_f32]];
    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        timeout: Duration::from_secs(2),
        use_alpha_crown: false,
        ..Default::default()
    });
    let bootstrap = expired_deadline_bootstrap_for_fallback(&verifier, &graph, &input);
    let expired = Instant::now()
        .checked_sub(Duration::from_millis(1))
        .expect("representable expired deadline");

    super::root::reset_root_objective_spec_build_count_for_test();
    let error = match super::root::compute_root_objective_bounds(
        &verifier,
        &graph,
        &input,
        &objectives,
        &[1.0e9, 1.0e9],
        false,
        None,
        &bootstrap,
        Some(expired),
        None,
        false,
        false,
    ) {
        Err(error) => error,
        Ok(_) => panic!("an exhausted hard authority must not launch a fallback pass"),
    };
    assert!(
        error.is_deadline_exceeded(),
        "expected terminal DeadlineExceeded, got {error:?}"
    );
    assert_eq!(
        super::root::root_objective_spec_build_count_for_test(),
        0,
        "expired global authority must refuse before dense-spec allocation"
    );
}

// ===========================================================================
// Verdict-preservation tests for upstream-bound inheritance.
//
// The constrained forward pass reuses parent-domain intermediate bounds for
// nodes provably unaffected by a BaB split (see
// `compute_constrained_forward_bounds`). The reused bounds are identical to a
// full recomputation by construction (the equivalence is asserted element-wise
// in `constraints::tests::upstream_cache`), so end-to-end verdicts must be
// unchanged. These tests pin the verdicts on a graph whose exact output bounds
// are known, exercising the BaB child path that performs the caching.
// ===========================================================================

/// Known-VERIFIABLE property still verifies with upstream-bound caching active.
///
/// Network: Y₀ = max(x,0)+0.5 ∈ [0.5, 1.5]. Objective `[-1,0]` with threshold
/// `-1.6` checks `-Y₀ > -1.6` ⇔ `Y₀ < 1.6`, which holds for all x (Y₀ ≤ 1.5).
/// Verifying the second objective `[0,-1]` (-Y₁ > -1.6 ⇔ Y₁ < 1.6; Y₁ ≤ 0.5)
/// also holds. Disjunctive mode requires ALL objectives → Verified.
#[ntest::timeout(10000)]
#[test]
fn test_verdict_preserved_verifiable_with_upstream_cache() {
    let (graph, input) = build_single_relu_anti_correlated_graph();

    let objectives = vec![vec![-1.0_f32, 0.0], vec![0.0, -1.0_f32]];
    let thresholds = vec![-1.6_f32, -1.6];

    let config = BetaCrownConfig {
        timeout: std::time::Duration::from_secs(5),
        max_domains: 1000,
        max_depth: 20,
        batch_size: 1,
        ..Default::default()
    };

    let result = BetaCrownVerifier::new(config)
        .verify_graph_relu_split_multi_objective_with_engine(
            &graph,
            &input,
            &objectives,
            &thresholds,
            None,
            None,
        )
        .expect("disjunctive BaB should complete without error");

    assert!(
        matches!(result.result, BabVerificationStatus::Verified),
        "known-verifiable property must remain Verified with upstream caching, got {:?}",
        result.result
    );
}

/// Known-UNVERIFIABLE property still does NOT verify with upstream caching.
///
/// Objective `[-1,0]` with threshold `-0.4` checks `-Y₀ > -0.4` ⇔ `Y₀ < 0.4`.
/// Y₀ = max(x,0)+0.5 ≥ 0.5 for all x, so at e.g. x = -1 (Y₀ = 0.5) the property
/// `-Y₀ > -0.4` is false → the property does not hold. The multi-objective BaB
/// engine reports this as a conclusive non-`Verified` verdict (`Unknown` with a
/// definite "cannot be verified" reason, or `Violated`). The soundness-critical
/// guarantee is that upstream-bound caching must NEVER flip such a property to
/// `Verified`.
#[ntest::timeout(10000)]
#[test]
fn test_verdict_preserved_unverifiable_with_upstream_cache() {
    let (graph, input) = build_single_relu_anti_correlated_graph();

    let objectives = vec![vec![-1.0_f32, 0.0]];
    let thresholds = vec![-0.4_f32];

    let config = BetaCrownConfig {
        timeout: std::time::Duration::from_secs(5),
        max_domains: 1000,
        max_depth: 20,
        batch_size: 1,
        ..Default::default()
    };

    let result = BetaCrownVerifier::new(config)
        .verify_graph_relu_split_multi_objective_with_engine(
            &graph,
            &input,
            &objectives,
            &thresholds,
            None,
            None,
        )
        .expect("disjunctive BaB should complete without error");

    assert!(
        !matches!(result.result, BabVerificationStatus::Verified),
        "property that provably does not hold must NEVER become Verified with \
         upstream caching, got {:?}",
        result.result
    );
}

/// Verdict preservation on a deeper (two-ReLU) graph that genuinely enters BaB,
/// exercising the constrained forward pass — and thus upstream-bound reuse — on
/// real child domains. Generous thresholds make every objective trivially
/// verifiable so the verdict is a deterministic Verified.
#[ntest::timeout(10000)]
#[test]
fn test_verdict_preserved_multi_relu_bab_verifies_with_upstream_cache() {
    let (graph, input) = build_multi_relu_graph();

    // Outputs of this small network are comfortably within [-100, 100]; the
    // objectives `+Y_k` with threshold -100 are verified for all inputs.
    let objectives = vec![vec![1.0_f32, 0.0], vec![0.0, 1.0_f32]];
    let thresholds = vec![-100.0_f32, -100.0];

    let config = BetaCrownConfig {
        timeout: std::time::Duration::from_secs(5),
        max_domains: 1000,
        max_depth: 20,
        batch_size: 1,
        ..Default::default()
    };

    let result = BetaCrownVerifier::new(config)
        .verify_graph_relu_split_multi_objective_with_engine(
            &graph,
            &input,
            &objectives,
            &thresholds,
            None,
            None,
        )
        .expect("disjunctive BaB should complete without error");

    assert!(
        matches!(result.result, BabVerificationStatus::Verified),
        "trivially-verifiable two-ReLU property must remain Verified, got {:?}",
        result.result
    );
}
