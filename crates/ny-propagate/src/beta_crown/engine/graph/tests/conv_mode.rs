// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::beta_crown::{BetaCrownConfig, BetaCrownVerifier, ConvMode, GraphPrecomputedBounds};
use crate::bounds::patches::{patches_to_dense_call_sites, reset_patches_to_dense_call_count};
use crate::layers::{Conv2dLayer, FlattenLayer, LinearLayer, ReLULayer};
use crate::{BoundedTensor, GraphNetwork, Layer, Network};
use ndarray::{arr1, arr2, ArrayD, IxDyn};

fn build_conv_classifier_graph_3813() -> (GraphNetwork, BoundedTensor) {
    let kernel =
        ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 2]), vec![0.5_f32, -0.25, 0.75, 0.4]).unwrap();
    let conv = Conv2dLayer::with_input_shape(kernel, Some(arr1(&[0.1_f32])), (1, 1), (0, 0), 4, 4)
        .unwrap();
    let linear = LinearLayer::new(
        arr2(&[
            [0.25_f32, -0.5, 0.75, 0.1, 0.0, 0.5, -0.2, 0.4, 0.3],
            [-0.4, 0.3, 0.2, -0.6, 0.5, -0.1, 0.7, -0.2, 0.15],
        ]),
        Some(arr1(&[0.05_f32, -0.1])),
    )
    .unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Conv2d(conv));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Flatten(FlattenLayer::new(0)));
    network.add_layer(Layer::Linear(linear));

    let graph = GraphNetwork::from_sequential(&network).unwrap();
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, 4, 4]), -0.2_f32),
        ArrayD::from_elem(IxDyn(&[1, 4, 4]), 0.6_f32),
    )
    .unwrap();

    (graph, input)
}

fn assert_runtime_avoids_global_patches_reentry_3813(call_sites: &[String], context: &str) {
    let forbidden = [
        "network/graph_crown/propagation.rs",
        "network/graph_crown/spec_propagation/",
        "network/graph_alpha/backward/",
        "network/graph_alpha/bounds/target_backward_patches.rs",
    ];
    let unexpected = call_sites
        .iter()
        .filter(|site| forbidden.iter().any(|needle| site.contains(needle)))
        .cloned()
        .collect::<Vec<_>>();
    assert!(
        unexpected.is_empty(),
        "#3813 matrix mode should not re-enter patches outside constrained backward in {context}; unexpected patches->dense sites={unexpected:?}, all_sites={call_sites:?}",
    );
}

#[test]
fn configured_graph_for_crown_uses_explicit_matrix_without_cuts_3813() {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        enable_cuts: false,
        conv_mode: ConvMode::Matrix,
        ..Default::default()
    });

    let configured = verifier.configured_graph_for_crown(&GraphNetwork::new());
    assert!(
        !configured.use_patches_mode,
        "#3813: explicit matrix mode must disable graph patches without cut authority"
    );
}

#[test]
fn configured_graph_for_crown_preserves_explicit_patches_override_3813() {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        enable_cuts: false,
        conv_mode: ConvMode::Patches,
        ..Default::default()
    });

    let configured = verifier.configured_graph_for_crown(&GraphNetwork::new());
    assert!(
        configured.use_patches_mode,
        "#3813: explicit patches mode must survive verifier graph cloning"
    );
}

#[test]
fn configured_graph_scopes_degradation_logs_per_verification() {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let source = GraphNetwork::new();

    // Source/model scope is independent of every configured verification.
    assert_eq!(
        source
            .crown_degradation_warning_log_receipt()
            .map(|receipt| receipt.occurrence),
        Some(1)
    );
    let first = verifier.configured_graph_for_crown(&source);
    assert_eq!(
        first
            .crown_degradation_warning_log_receipt()
            .map(|receipt| receipt.occurrence),
        Some(1)
    );

    // A BaB domain clone shares the configured verification's Arc/counter.
    let domain_clone = first.clone();
    assert_eq!(
        domain_clone
            .crown_degradation_warning_log_receipt()
            .map(|receipt| receipt.occurrence),
        Some(2)
    );
    assert_eq!(first.crown_degradation_warning_log_receipt(), None);

    // Reusing the same verifier/model later receives a fresh first diagnostic,
    // without resetting either the source or the first verification's Arc.
    let later = verifier.configured_graph_for_crown(&source);
    assert_eq!(
        later
            .crown_degradation_warning_log_receipt()
            .map(|receipt| receipt.occurrence),
        Some(1)
    );
    assert_eq!(
        source
            .crown_degradation_warning_log_receipt()
            .map(|receipt| receipt.occurrence),
        Some(2)
    );
    assert_eq!(
        first
            .crown_degradation_warning_log_receipt()
            .map(|receipt| receipt.occurrence),
        Some(4)
    );
}

#[ntest::timeout(10000)]
#[test]
fn multi_objective_relu_split_uses_configured_matrix_mode_3813() {
    let (graph, input) = build_conv_classifier_graph_3813();
    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        enable_cuts: false,
        conv_mode: ConvMode::Matrix,
        use_alpha_crown: false,
        timeout: std::time::Duration::from_secs(1),
        max_domains: 1,
        max_depth: 0,
        batch_size: 1,
        ..Default::default()
    });
    let objectives = vec![vec![1.0_f32, 0.0], vec![0.0, 1.0_f32]];
    let thresholds = vec![10.0_f32, 10.0];

    reset_patches_to_dense_call_count();
    verifier
        .verify_graph_relu_split_multi_objective_with_engine(
            &graph,
            &input,
            &objectives,
            &thresholds,
            None,
            None,
        )
        .expect("#3813 multi-objective relu-split should complete on the toy conv graph");

    assert_runtime_avoids_global_patches_reentry_3813(
        &patches_to_dense_call_sites(),
        "graph multi-objective relu split",
    );
}

#[ntest::timeout(10000)]
#[test]
fn input_split_uses_explicit_matrix_mode_3813() {
    let (graph, input) = build_conv_classifier_graph_3813();
    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        conv_mode: ConvMode::Matrix,
        use_alpha_crown: false,
        timeout: std::time::Duration::from_secs(1),
        max_domains: 1,
        max_depth: 0,
        batch_size: 1,
        ..Default::default()
    });
    let objective = [1.0_f32, -1.0_f32];

    reset_patches_to_dense_call_count();
    verifier
        .verify_graph_input_split(&graph, &input, &objective, 0.0)
        .expect("#3813 input-split should complete on the toy conv graph");

    assert_runtime_avoids_global_patches_reentry_3813(
        &patches_to_dense_call_sites(),
        "graph input split",
    );
}

#[ntest::timeout(10000)]
#[test]
fn multi_objective_input_split_uses_explicit_matrix_mode_3813() {
    let (graph, input) = build_conv_classifier_graph_3813();
    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        conv_mode: ConvMode::Matrix,
        use_alpha_crown: false,
        timeout: std::time::Duration::from_secs(1),
        max_domains: 1,
        max_depth: 0,
        batch_size: 1,
        ..Default::default()
    });
    let objectives = vec![vec![1.0_f32, 0.0], vec![0.0, 1.0_f32]];
    let thresholds = vec![10.0_f32, 10.0];

    reset_patches_to_dense_call_count();
    verifier
        .verify_graph_input_split_multi_objective_conjunctive(
            &graph,
            &input,
            &objectives,
            &thresholds,
            None,
            None,
        )
        .expect("#3813 multi-objective input-split should complete on the toy conv graph");

    assert_runtime_avoids_global_patches_reentry_3813(
        &patches_to_dense_call_sites(),
        "graph multi-objective input split",
    );
}

#[ntest::timeout(10000)]
#[test]
fn relu_split_uses_configured_matrix_mode_3813() {
    let (graph, input) = build_conv_classifier_graph_3813();
    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        enable_cuts: false,
        conv_mode: ConvMode::Matrix,
        use_alpha_crown: false,
        timeout: std::time::Duration::from_secs(1),
        max_domains: 1,
        max_depth: 0,
        batch_size: 1,
        ..Default::default()
    });
    let objective = [1.0_f32, -1.0_f32];

    reset_patches_to_dense_call_count();
    verifier
        .verify_graph_relu_split(&graph, &input, &objective, 0.0)
        .expect("#3813 relu-split should complete on the toy conv graph");

    assert_runtime_avoids_global_patches_reentry_3813(
        &patches_to_dense_call_sites(),
        "graph relu split",
    );
}

#[ntest::timeout(10000)]
#[test]
fn relu_split_with_bounds_uses_configured_matrix_mode_3813() {
    let (graph, input) = build_conv_classifier_graph_3813();
    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        enable_cuts: false,
        conv_mode: ConvMode::Matrix,
        use_alpha_crown: false,
        timeout: std::time::Duration::from_secs(1),
        max_domains: 1,
        max_depth: 0,
        batch_size: 1,
        ..Default::default()
    });
    let objective = [1.0_f32, -1.0_f32];

    reset_patches_to_dense_call_count();
    let (node_bounds, output_bounds) = verifier
        .compute_initial_graph_bounds(&graph, &input, None)
        .expect("#3813 initial graph bounds should complete on the toy conv graph");
    let precomputed = GraphPrecomputedBounds::new(&node_bounds, &output_bounds);
    verifier
        .verify_graph_relu_split_with_bounds(&graph, &input, &objective, 0.0, &precomputed)
        .expect("#3813 relu-split-with-bounds should complete on the toy conv graph");

    assert_runtime_avoids_global_patches_reentry_3813(
        &patches_to_dense_call_sites(),
        "graph relu split with precomputed bounds",
    );
}

#[ntest::timeout(10000)]
#[test]
fn gpu_domain_list_uses_configured_matrix_mode_3813() {
    let (graph, input) = build_conv_classifier_graph_3813();
    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        enable_cuts: false,
        conv_mode: ConvMode::Matrix,
        use_alpha_crown: false,
        timeout: std::time::Duration::from_secs(1),
        max_domains: 1,
        max_depth: 0,
        batch_size: 1,
        ..Default::default()
    });
    let objective = [1.0_f32, -1.0_f32];

    reset_patches_to_dense_call_count();
    verifier
        .verify_graph_gpu_domain_list(&graph, &input, &objective, 0.0, None, None)
        .expect("#3813 gpu-domain-list path should complete on the toy conv graph");

    assert_runtime_avoids_global_patches_reentry_3813(
        &patches_to_dense_call_sites(),
        "graph gpu domain list",
    );
}

/// #w5-bab-throughput: `configured_graph_for_crown` must ADOPT the source
/// graph's certified forward-linear reference map. `Clone` resets the cache and
/// `set_use_patches_mode` invalidates it, so before the fix every verify entry
/// repaid the full O(L) certified pass (~25s on cifar100) for a map already
/// computed upstream (e.g. warmed on the CLI graph during the attack phase).
///
/// Oracle: warm the cache on the source, then ask the CLONE for the cached map
/// under an ALREADY-EXPIRED deadline — success is only possible via a cache
/// hit (a cold collection aborts with DeadlineExceeded before the first node).
#[test]
fn configured_graph_for_crown_adopts_forward_linear_cache_w5() {
    // Serialized + gate pinned: the forward-linear cache key is salted with
    // the dark ConvTranspose surface gate, so an unsynchronized mid-test env
    // flip would turn the expected cache hit into a cold build (which the
    // expired deadline then refuses).
    crate::tests::with_serialized_env_vars(
        &[("NY_NO_FORWARD_LINEAR_CONV_TRANSPOSE_REF", "1")],
        configured_graph_for_crown_adopts_forward_linear_cache_w5_body,
    );
}

fn configured_graph_for_crown_adopts_forward_linear_cache_w5_body() {
    let (graph, input) = build_conv_classifier_graph_3813();

    // Warm the source cache with a generous deadline.
    let warm = graph
        .collect_forward_linear_bounds_dag_cached(
            &input,
            None,
            // The cold-build admission floor is 30s.  Passing exactly 30s is
            // inherently racy because the callee samples `Instant::now()`
            // after this deadline is constructed.
            Some(std::time::Instant::now() + std::time::Duration::from_mins(1)),
        )
        .expect("source forward-linear collection should succeed on the toy conv graph");

    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let configured = verifier.configured_graph_for_crown(&graph);

    let expired = std::time::Instant::now()
        .checked_sub(std::time::Duration::from_millis(1))
        .unwrap();
    let adopted = configured
        .collect_forward_linear_bounds_dag_cached(&input, None, Some(expired))
        .expect(
            "#w5: the configured clone must serve the adopted cache without recomputing \
             (a cold collection under an expired deadline aborts)",
        );

    // Same input key, same graph semantics — the adopted map must be the same map.
    assert!(
        std::sync::Arc::ptr_eq(&warm, &adopted),
        "#w5: adopted cache must share the source's map (Arc identity)"
    );
}
