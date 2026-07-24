// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::time::{Duration, Instant};

use ndarray::{arr1, arr2};
use ny_core::{NyError, Result};

use super::*;
use crate::batched_domain::DomainMetadata;
use crate::beta_crown::branching::BranchingHeuristic;
use crate::beta_crown::config::BetaCrownConfig;
use crate::beta_crown::engine::graph::forward_mode_test_support::{
    assert_bounds_close_4354, build_forward_mode_graph_fixture_4354,
    expected_forward_root_output_4354, plain_graph_crown_output_4354,
};
use crate::beta_crown::engine::graph::input_split::shared::compute_crown_or_ibp_bounds;
use crate::beta_crown::nonlinear_branching::NonlinearBranchingConfig;
use crate::bounds::LinearBounds;
use crate::layers::{GELULayer, LinearLayer};
use crate::network::GraphNode;
use crate::{Layer, ReLULayer};

fn simple_graph_network() -> GraphNetwork {
    let linear1 = LinearLayer::new(arr2(&[[1.0_f32, -1.0_f32], [0.5_f32, 1.0_f32]]), None)
        .expect("valid first linear layer");
    let linear2 = LinearLayer::new(arr2(&[[1.0_f32, 1.0_f32]]), None).expect("valid output layer");

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
    graph
}

fn genbab_graph_network() -> GraphNetwork {
    let linear1 = LinearLayer::new(arr2(&[[1.0_f32]]), None).expect("valid first linear layer");
    let linear2 = LinearLayer::new(arr2(&[[1.0_f32]]), None).expect("valid second linear layer");

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "gelu1",
        Layer::GELU(GELULayer::default()),
        vec!["relu1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(linear2),
        vec!["gelu1".to_string()],
    ));
    graph.set_output("linear2");
    graph
}

fn simple_input_bounds() -> BoundedTensor {
    BoundedTensor::new(
        arr1(&[-1.0_f32, 0.0_f32]).into_dyn(),
        arr1(&[1.0_f32, 1.5_f32]).into_dyn(),
    )
    .expect("valid input bounds")
}

fn test_config() -> BetaCrownConfig {
    BetaCrownConfig {
        verify_upper_bound: true,
        use_alpha_crown: false,
        batch_size: 4,
        timeout: Duration::from_secs(5),
        ..Default::default()
    }
}

fn assert_linear_bounds_eq(actual: LinearBounds, expected: LinearBounds) {
    let (actual_lower_a, actual_lower_b, actual_upper_a, actual_upper_b) = actual.into_parts();
    let (expected_lower_a, expected_lower_b, expected_upper_a, expected_upper_b) =
        expected.into_parts();
    assert_eq!(actual_lower_a, expected_lower_a);
    assert_eq!(actual_lower_b, expected_lower_b);
    assert_eq!(actual_upper_a, expected_upper_a);
    assert_eq!(actual_upper_b, expected_upper_b);
}

#[ntest::timeout(5000)]
#[test]
fn test_cache_restore_input_split_linear_bounds_round_trip_3089() -> Result<()> {
    let linear_bounds = LinearBounds::new(
        arr2(&[[1.0_f32, -2.0_f32]]),
        arr1(&[0.25_f32]),
        arr2(&[[3.0_f32, 4.0_f32]]),
        arr1(&[-0.5_f32]),
    )?;
    let mut metadata = DomainMetadata::root(-1.0, 1.0)?;
    metadata.cached_la = Some(Arc::new(cache_input_split_linear_bounds(&linear_bounds)));

    let restored = restore_input_split_linear_bounds(&metadata)
        .expect("input-split cache should restore the stored linear bounds");

    assert_linear_bounds_eq(restored, linear_bounds);
    Ok(())
}

#[ntest::timeout(5000)]
#[test]
fn test_compute_initial_bounds_rejects_empty_objective_3089() {
    let graph = simple_graph_network();
    let input = simple_input_bounds();
    let config = test_config();

    let err = match compute_initial_bounds(&graph, &input, &[], &config, None, None, None, false) {
        Ok(_) => panic!("empty spec objective should not produce a usable root domain"),
        Err(err) => err,
    };

    match err {
        NyError::InvalidSpec(message) => {
            assert!(
                message.contains("empty output tensor"),
                "expected explicit empty-output guard, got: {message}"
            );
        }
        NyError::ShapeMismatch { expected, got } => {
            assert_eq!(
                expected,
                vec![1],
                "empty objective should fail the spec shape contract"
            );
            assert_eq!(
                got,
                vec![0],
                "empty objective should report the missing objective row"
            );
        }
        other => panic!("expected InvalidSpec for empty objective, got {other:?}"),
    }
}

#[ntest::timeout(5000)]
#[test]
fn test_compute_initial_bounds_input_split_bootstrap_matches_scalar_output_3089() -> Result<()> {
    let graph = simple_graph_network();
    let input = simple_input_bounds();
    let config = BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::InputSplit,
        ..test_config()
    };

    let init = compute_initial_bounds(&graph, &input, &[1.0], &config, None, None, None, true)?;
    let bootstrap = init
        .input_split_bootstrap
        .as_ref()
        .expect("input-split init should produce reusable bootstrap state");

    assert_eq!(init.initial_output.lower()[[0]], init.root_lower);
    assert_eq!(init.initial_output.upper()[[0]], init.root_upper);
    assert_eq!(
        bootstrap.spec_matrix,
        arr2(&[[1.0_f32]]),
        "input-split bootstrap must retain the exact root spec matrix for deferred parent/child bounds"
    );
    assert!(
        bootstrap.root_linear_bounds.is_some(),
        "simple CROWN input-split init should keep the root linear bounds for warm-start"
    );

    Ok(())
}

#[ntest::timeout(5000)]
#[test]
fn test_compute_initial_bounds_input_split_forward_mode_reuses_root_bounds_4354() -> Result<()> {
    let graph = simple_graph_network();
    let input = simple_input_bounds();
    let config = BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::InputSplit,
        use_alpha_crown: false,
        use_forward_bounds: true,
        ..test_config()
    };
    let forward_node_bounds = graph.collect_forward_linear_bounds_dag_with_engine(&input, None)?;
    let spec_matrix = arr2(&[[1.0_f32]]);
    let (expected_bounds, _expected_linear) = compute_crown_or_ibp_bounds(
        &graph,
        &input,
        &spec_matrix,
        None,
        Some(&forward_node_bounds),
        None,
        None,
        None,
        config.crown_backward_layers,
        config.input_split_ibp_enhancement,
    )?;

    let init = compute_initial_bounds(&graph, &input, &[1.0], &config, None, None, None, true)?;
    let bootstrap = init
        .input_split_bootstrap
        .as_ref()
        .expect("input-split init should produce reusable bootstrap state");
    let cached_bounds = bootstrap
        .fixed_node_bounds
        .as_ref()
        .expect("forward+crown input split should keep reusable root node bounds");
    let cached_relu = cached_bounds
        .get("relu1")
        .expect("cached forward bounds should include the branch node");
    let expected_relu = forward_node_bounds
        .get("relu1")
        .expect("forward-linear bounds should include the branch node");

    assert_eq!(
        cached_relu.lower().iter().copied().collect::<Vec<_>>(),
        expected_relu.lower().iter().copied().collect::<Vec<_>>(),
        "GPU input split should cache the forward-linear root lower bounds"
    );
    assert_eq!(
        cached_relu.upper().iter().copied().collect::<Vec<_>>(),
        expected_relu.upper().iter().copied().collect::<Vec<_>>(),
        "GPU input split should cache the forward-linear root upper bounds"
    );
    assert_eq!(
        init.root_lower,
        expected_bounds.lower()[[0]],
        "GPU input split root lower bound should use the cached forward-linear intermediates"
    );
    assert_eq!(
        init.root_upper,
        expected_bounds.upper()[[0]],
        "GPU input split root upper bound should use the cached forward-linear intermediates"
    );
    assert!(
        bootstrap.root_linear_bounds.is_some(),
        "forward+crown input split should still capture root linear bounds for split scoring"
    );

    Ok(())
}

#[ntest::timeout(5000)]
#[test]
fn test_compute_initial_bounds_input_split_forward_mode_skips_expired_deadline_4398() -> Result<()>
{
    let graph = simple_graph_network();
    let input = simple_input_bounds();
    let config = BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::InputSplit,
        use_alpha_crown: false,
        use_forward_bounds: true,
        ..test_config()
    };
    let expired_deadline = Some(
        Instant::now()
            .checked_sub(Duration::from_millis(1))
            .unwrap(),
    );
    let spec_matrix = arr2(&[[1.0_f32]]);
    let (expected_bounds, expected_linear) = compute_crown_or_ibp_bounds(
        &graph,
        &input,
        &spec_matrix,
        None,
        None,
        None,
        None,
        expired_deadline,
        config.crown_backward_layers,
        config.input_split_ibp_enhancement,
    )?;

    let init = compute_initial_bounds(
        &graph,
        &input,
        &[1.0],
        &config,
        None,
        expired_deadline,
        None,
        true,
    )?;
    let bootstrap = init
        .input_split_bootstrap
        .as_ref()
        .expect("input-split init should produce reusable bootstrap state");

    assert!(
        bootstrap.fixed_node_bounds.is_none(),
        "expired forward-linear warmup should skip reusable node bounds instead of aborting DomainList init"
    );
    assert!(
        init.initial_node_bounds.is_empty(),
        "skipped forward-linear warmup should not publish stale node bounds into the root domain state"
    );
    assert_eq!(
        init.root_lower,
        expected_bounds.lower()[[0]],
        "expired forward-linear warmup should fall back to the same conservative scalar lower bound as the shared input-split helper"
    );
    assert_eq!(
        init.root_upper,
        expected_bounds.upper()[[0]],
        "expired forward-linear warmup should fall back to the same conservative scalar upper bound as the shared input-split helper"
    );
    assert_eq!(
        bootstrap.root_linear_bounds.is_some(),
        expected_linear.is_some(),
        "DomainList input split should preserve the fallback linear-bounds contract from compute_crown_or_ibp_bounds"
    );

    Ok(())
}

#[ntest::timeout(5000)]
#[test]
fn test_compute_initial_bounds_non_input_split_forward_mode_uses_forward_output_4354() -> Result<()>
{
    let (graph, input) = build_forward_mode_graph_fixture_4354();
    let config = BetaCrownConfig {
        use_alpha_crown: false,
        use_forward_bounds: true,
        ..test_config()
    };
    let expected_output = expected_forward_root_output_4354(&graph, &input)?;
    let plain_output = plain_graph_crown_output_4354(&graph, &input, config.crown_backward_layers)?;

    assert!(
        plain_output
            .lower()
            .iter()
            .zip(expected_output.lower().iter())
            .chain(
                plain_output
                    .upper()
                    .iter()
                    .zip(expected_output.upper().iter())
            )
            .any(|(plain, expected)| (plain - expected).abs() > 1e-6),
        "fixture must distinguish forward+crown output from plain DAG-CROWN so the routing regression stays observable"
    );

    let init = compute_initial_bounds(
        &graph,
        &input,
        &[1.0, 0.0],
        &config,
        None,
        None,
        None,
        false,
    )?;

    assert_bounds_close_4354(
        &init.initial_output,
        &expected_output,
        "gpu_bab non-input-split forward output",
    );
    Ok(())
}

#[ntest::timeout(5000)]
#[test]
fn test_create_domain_list_non_input_split_preserves_layer_storage_3089() -> Result<()> {
    let graph = simple_graph_network();
    let input = simple_input_bounds();
    let config = test_config();

    let init = compute_initial_bounds(&graph, &input, &[1.0], &config, None, None, None, false)?;
    let setup = crate::beta_crown::engine::graph::shared::setup::build_graph_bab_setup(
        &graph,
        &init.initial_node_bounds,
    );
    let mut expected_layer_names: Vec<String> = init.initial_node_bounds.keys().cloned().collect();
    expected_layer_names.sort();

    let (mut domain_list, layer_names) =
        create_domain_list(&init, &input, &graph, &config, false, &setup)?;
    let picked = domain_list.pick_out(1)?;

    assert_eq!(layer_names, expected_layer_names);
    assert_eq!(picked.batch_size, 1);
    assert_eq!(picked.global_lbs, vec![init.root_lower]);
    assert_eq!(picked.global_ubs, vec![init.root_upper]);
    assert_eq!(picked.layer_lowers.len(), expected_layer_names.len());
    assert_eq!(picked.layer_uppers.len(), expected_layer_names.len());
    for layer_name in &expected_layer_names {
        let lower = picked
            .layer_lowers
            .get(layer_name)
            .expect("configured layer lower bounds should be stored");
        let upper = picked
            .layer_uppers
            .get(layer_name)
            .expect("configured layer upper bounds should be stored");
        assert_eq!(
            lower.shape()[0],
            1,
            "root layer lower bounds keep batch dim"
        );
        assert_eq!(
            upper.shape()[0],
            1,
            "root layer upper bounds keep batch dim"
        );
    }
    assert!(
        picked.metadata[0].cached_la().is_none(),
        "non-input-split roots should not seed cached linear bounds"
    );
    assert!(
        picked.metadata[0].alpha_state().is_some(),
        "root domain should carry initialized alpha state"
    );

    Ok(())
}

#[ntest::timeout(5000)]
#[test]
fn test_create_domain_list_input_split_preserves_cached_linear_bounds_3089() -> Result<()> {
    let graph = simple_graph_network();
    let input = simple_input_bounds();
    let config = BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::InputSplit,
        ..test_config()
    };

    let init = compute_initial_bounds(&graph, &input, &[1.0], &config, None, None, None, true)?;
    let expected_linear = init
        .input_split_bootstrap
        .as_ref()
        .and_then(|bootstrap| bootstrap.root_linear_bounds.clone())
        .expect("input-split init should keep a root linear cache");
    let setup = crate::beta_crown::engine::graph::shared::setup::build_graph_bab_setup(
        &graph,
        &init.initial_node_bounds,
    );

    let (mut domain_list, layer_names) =
        create_domain_list(&init, &input, &graph, &config, true, &setup)?;
    let picked = domain_list.pick_out(1)?;
    let restored = restore_input_split_linear_bounds(&picked.metadata[0])
        .expect("input-split root metadata should retain cached linear bounds");

    assert!(
        !layer_names.is_empty(),
        "callers still need sorted graph layer names"
    );
    assert_eq!(picked.batch_size, 1);
    assert!(
        picked.layer_lowers.is_empty() && picked.layer_uppers.is_empty(),
        "input-split DomainList roots should avoid per-layer storage"
    );
    assert_eq!(picked.global_lbs, vec![init.root_lower]);
    assert_eq!(picked.global_ubs, vec![init.root_upper]);
    assert!(
        picked.metadata[0].alpha_state().is_some(),
        "root domain should preserve alpha initialization alongside cached lA"
    );
    assert_linear_bounds_eq(restored, expected_linear);

    Ok(())
}

#[ntest::timeout(5000)]
#[test]
fn test_build_setup_context_collects_relu_pre_map_and_genbab_nodes_3089() {
    let graph = genbab_graph_network();
    let relu_nodes = vec!["relu1".to_string()];
    let config = BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::GenBaB(NonlinearBranchingConfig::default()),
        ..Default::default()
    };

    let setup = build_setup_context(&graph, &config, relu_nodes.clone());
    let mut nonlinear_nodes = setup.nonlinear_nodes.clone();
    nonlinear_nodes.sort();

    assert_eq!(&setup.relu_nodes, &relu_nodes);
    assert_eq!(
        setup.relu_pre_map.get("relu1"),
        Some(&"linear1".to_string())
    );
    assert!(
        setup.genbab_instance.is_some(),
        "setup should produce a genbab_instance for gelu1"
    );
    assert_eq!(
        nonlinear_nodes,
        vec!["gelu1".to_string(), "relu1".to_string()]
    );
}

#[ntest::timeout(5000)]
#[test]
fn test_create_domain_list_root_warm_start_uses_bounds_when_beta_disabled_3089() -> Result<()> {
    let graph = simple_graph_network();
    let input = simple_input_bounds();
    let config = test_config();
    let init = compute_initial_bounds(&graph, &input, &[1.0], &config, None, None, None, false)?;
    let setup = crate::beta_crown::engine::graph::shared::setup::build_graph_bab_setup(
        &graph,
        &init.initial_node_bounds,
    );

    let (mut domain_list, _layer_names) =
        create_domain_list(&init, &input, &graph, &config, false, &setup)?;
    let picked = domain_list.pick_out(1)?;
    let alpha_state = picked.metadata[0]
        .alpha_state()
        .expect("root domain should always carry alpha state");

    assert!(
        !alpha_state.neurons.is_empty() || !alpha_state.upper_neurons.is_empty(),
        "beta_iterations=0 should still initialize heuristic graph alpha state from bounds"
    );

    Ok(())
}
