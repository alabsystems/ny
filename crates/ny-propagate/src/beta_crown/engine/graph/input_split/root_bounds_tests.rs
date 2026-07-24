// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::{Conv2dLayer, GraphNode, Layer, LinearLayer, ReLULayer};
use ndarray::{arr1, arr2, ArrayD, IxDyn};
use ny_test_utils::{env::lock_env, CountingGemmEngine};
use std::collections::BTreeMap;
use std::time::Duration;

fn restart_graph() -> (GraphNetwork, BoundedTensor) {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "linear1",
        Layer::Linear(
            LinearLayer::new(
                arr2(&[[1.0_f32, -0.5], [0.4, 0.8], [-0.9, 0.3]]),
                Some(arr1(&[0.1_f32, -0.2, 0.05])),
            )
            .expect("valid linear1"),
        ),
    ));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "out",
        Layer::Linear(
            LinearLayer::new(arr2(&[[0.7_f32, -0.4, 0.6]]), Some(arr1(&[0.03])))
                .expect("valid output linear"),
        ),
        vec!["relu1".to_string()],
    ));
    graph.set_output("out");
    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, -0.75]).into_dyn(),
        arr1(&[1.0_f32, 0.9]).into_dyn(),
    )
    .expect("valid root box");
    (graph, input)
}

fn deterministic_alpha_config() -> BetaCrownConfig {
    let mut config = BetaCrownConfig {
        use_alpha_crown: true,
        use_forward_bounds: false,
        use_crown_ibp: false,
        ..BetaCrownConfig::default()
    };
    config.alpha_config.iterations = 1;
    config.alpha_config.early_stop_patience = 1;
    config.alpha_config.fix_interm_bounds = false;
    config.alpha_config.adaptive_skip = false;
    config.alpha_config.gradient_method = GradientMethod::AnalyticChain;
    config
}

fn assert_f32_map_bits_eq(
    left: &BTreeMap<String, ndarray::Array1<f32>>,
    right: &BTreeMap<String, ndarray::Array1<f32>>,
    label: &str,
) {
    assert_eq!(
        left.keys().collect::<Vec<_>>(),
        right.keys().collect::<Vec<_>>()
    );
    for (name, left_values) in left {
        let right_values = &right[name];
        assert_eq!(
            left_values.shape(),
            right_values.shape(),
            "{label} {name} shape"
        );
        assert_eq!(
            left_values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            right_values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "{label} {name} bits"
        );
    }
}

fn assert_alpha_state_bits_eq(left: &GraphAlphaState, right: &GraphAlphaState) {
    for (label, left_map, right_map) in [
        ("alphas", &left.alphas, &right.alphas),
        ("alphas_upper", &left.alphas_upper, &right.alphas_upper),
        ("velocity", &left.velocity, &right.velocity),
        ("adam_m", &left.adam_m, &right.adam_m),
        ("adam_v", &left.adam_v, &right.adam_v),
        (
            "velocity_upper",
            &left.velocity_upper,
            &right.velocity_upper,
        ),
        ("adam_m_upper", &left.adam_m_upper, &right.adam_m_upper),
        ("adam_v_upper", &left.adam_v_upper, &right.adam_v_upper),
    ] {
        assert_f32_map_bits_eq(left_map, right_map, label);
    }
    assert_eq!(left.unstable_mask, right.unstable_mask);
    assert_eq!(left.spatial_shapes, right.spatial_shapes);
    // This ReLU-only fixture has no SPSA-supplement alpha bundles.
    assert!(left.monotone_s_shaped_alphas.is_empty());
    assert!(right.monotone_s_shaped_alphas.is_empty());
    assert!(left.sqrt_alphas.is_empty());
    assert!(right.sqrt_alphas.is_empty());
    assert!(left.reciprocal_alphas.is_empty());
    assert!(right.reciprocal_alphas.is_empty());
    let left_ineligible = left
        .gpu_suffix_ineligible
        .read()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone();
    let right_ineligible = right
        .gpu_suffix_ineligible
        .read()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone();
    assert_eq!(left_ineligible, right_ineligible);
    assert!(
        !Arc::ptr_eq(&left.gpu_suffix_ineligible, &right.gpu_suffix_ineligible),
        "cached alpha state must detach mutable runtime-only caches"
    );
}

fn assert_root_value_bits_eq(left: &RootBoundsValue, right: &RootBoundsValue) {
    match (&left.0, &right.0) {
        (Some(left_map), Some(right_map)) => {
            assert_eq!(left_map.len(), right_map.len());
            for (name, left_bounds) in left_map {
                let right_bounds = &right_map[name];
                assert_eq!(left_bounds.shape(), right_bounds.shape(), "{name} shape");
                assert_eq!(
                    left_bounds
                        .lower()
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>(),
                    right_bounds
                        .lower()
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>(),
                    "{name} lower bits"
                );
                assert_eq!(
                    left_bounds
                        .upper()
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>(),
                    right_bounds
                        .upper()
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>(),
                    "{name} upper bits"
                );
            }
        }
        (None, None) => {}
        _ => panic!("root bound map option mismatch"),
    }
    match (&left.1, &right.1) {
        (Some(left_state), Some(right_state)) => {
            assert_alpha_state_bits_eq(left_state, right_state)
        }
        (None, None) => {}
        _ => panic!("root alpha state option mismatch"),
    }
}

#[test]
fn second_restart_hits_and_skips_collection_bit_exact_20260721() {
    // Block the repository's serialized test env writers while the exact
    // option fingerprint is sampled twice.
    let _env_lock = lock_env();
    let (graph, input) = restart_graph();
    let config = deterministic_alpha_config();
    let engine = CountingGemmEngine::new();
    let overall_deadline = Some(Instant::now() + Duration::from_secs(30));
    let cache = InputSplitRootBoundsCache::new(
        disjunctive_spec_identity(&[vec![1.0]], &[0.0], &[1]),
        overall_deadline,
    );

    let ibp = graph
        .collect_node_bounds_with_engine(&input, Some(&engine))
        .expect("reference IBP collection");
    let calls_before_first = engine.gemm_calls();
    let first = collect_input_split_root_node_bounds(
        &graph,
        &input,
        &config,
        Some(&engine),
        overall_deadline,
        "test restart #1",
        Some((&cache, overall_deadline)),
    )
    .expect("first restart root collection");
    let calls_after_first = engine.gemm_calls();
    assert!(
        calls_after_first > calls_before_first,
        "producer must exercise GEMM beyond the reference collection"
    );

    let second = collect_input_split_root_node_bounds(
        &graph,
        &input,
        &config,
        Some(&engine),
        overall_deadline.map(|d| {
            d.checked_sub(Duration::from_secs(1))
                .expect("test deadline must permit a one-second offset")
        }),
        "test restart #2",
        Some((&cache, overall_deadline)),
    )
    .expect("second restart cache hit");

    assert_eq!(cache.collections(), 1, "restart #2 must skip collection");
    assert_eq!(cache.misses(), 1, "only the cold lookup may miss");
    assert_eq!(cache.hits(), 1, "restart #2 must hit");
    assert_eq!(
        engine.gemm_calls(),
        calls_after_first,
        "restart #2 cache hit must issue no GEMM"
    );
    assert_root_value_bits_eq(&first, &second);

    // The producer's map is an intersection with its certified reference
    // map; the cached clone must preserve that shrink-only relation.
    for (name, cached) in second.0.as_ref().expect("cached root map") {
        let Some(reference) = ibp.get(name) else {
            continue;
        };
        for ((&cached_l, &cached_u), (&ibp_l, &ibp_u)) in cached
            .lower()
            .iter()
            .zip(cached.upper().iter())
            .zip(reference.lower().iter().zip(reference.upper().iter()))
        {
            assert!(cached_l >= ibp_l, "{name}: cached lower widened");
            assert!(cached_u <= ibp_u, "{name}: cached upper widened");
        }
    }
}

#[test]
fn exact_key_misses_on_ulp_option_input_engine_and_graph_identity_20260721() {
    let _env_lock = lock_env();
    let (graph, input) = restart_graph();
    let config = deterministic_alpha_config();
    let cache =
        InputSplitRootBoundsCache::new(disjunctive_spec_identity(&[vec![1.0]], &[0.0], &[1]), None);
    let engine_a = CountingGemmEngine::new();
    let engine_b = CountingGemmEngine::new();
    let base = root_bounds_cache_key(&cache, &graph, &input, &config, Some(&engine_a))
        .expect("deterministic base key");

    let graph_clone = graph.clone();
    let cloned_key = root_bounds_cache_key(&cache, &graph_clone, &input, &config, Some(&engine_a))
        .expect("same-graph clone key");
    assert!(
        base == cloned_key,
        "an exact graph clone must retain identity"
    );

    let (foreign_graph, _) = restart_graph();
    let foreign = root_bounds_cache_key(&cache, &foreign_graph, &input, &config, Some(&engine_a))
        .expect("foreign graph key");
    assert!(base != foreign, "same-shaped foreign graph must miss");

    let mut ulp_input = input.clone();
    let upper = arr1(&[f32::from_bits(1.0_f32.to_bits() + 1), 0.9_f32]).into_dyn();
    ulp_input = BoundedTensor::new(ulp_input.lower().clone(), upper).expect("ULP root box");
    let input_ulp = root_bounds_cache_key(&cache, &graph, &ulp_input, &config, Some(&engine_a))
        .expect("ULP input key");
    assert!(base != input_ulp, "one input ULP must miss");

    let mut option_ulp_config = config.clone();
    option_ulp_config.alpha_config.learning_rate =
        f32::from_bits(option_ulp_config.alpha_config.learning_rate.to_bits() + 1);
    let option_ulp =
        root_bounds_cache_key(&cache, &graph, &input, &option_ulp_config, Some(&engine_a))
            .expect("ULP option key");
    assert!(base != option_ulp, "one config ULP must miss");

    let other_engine = root_bounds_cache_key(&cache, &graph, &input, &config, Some(&engine_b))
        .expect("other engine key");
    assert!(base != other_engine, "a different engine object must miss");
}

#[test]
fn cache_scope_requires_exact_spec_deadline_and_is_inherited_20260721() {
    let objectives = vec![vec![1.0_f32, -0.0]];
    let thresholds = vec![0.25_f32];
    let clause_sizes = vec![1usize];
    let deadline = Some(Instant::now() + Duration::from_secs(10));
    let parent = crate::BetaCrownVerifier::new(BetaCrownConfig::default())
        .with_fresh_disjunctive_restart_root_cache(
            &objectives,
            &thresholds,
            &clause_sizes,
            deadline,
        );
    let restart_1 = parent.with_config_from(BetaCrownConfig::default());
    let restart_2 = parent.with_config_from(BetaCrownConfig::default());
    let cache_1 = restart_1
        .disjunctive_restart_root_cache(&objectives, &thresholds, &clause_sizes, deadline)
        .expect("restart #1 cache");
    let cache_2 = restart_2
        .disjunctive_restart_root_cache(&objectives, &thresholds, &clause_sizes, deadline)
        .expect("restart #2 inherited cache");
    assert!(
        std::ptr::eq(cache_1, cache_2),
        "restarts must share one call-local cache"
    );

    let mut spec_ulp = objectives.clone();
    spec_ulp[0][0] = f32::from_bits(spec_ulp[0][0].to_bits() + 1);
    assert!(
        restart_2
            .disjunctive_restart_root_cache(&spec_ulp, &thresholds, &clause_sizes, deadline,)
            .is_none(),
        "one objective ULP must reject the cache"
    );
    assert!(
        restart_2
            .disjunctive_restart_root_cache(
                &objectives,
                &thresholds,
                &clause_sizes,
                deadline.map(|d| d + Duration::from_nanos(1)),
            )
            .is_none(),
        "a different absolute deadline must reject the cache"
    );
    assert_ne!(
        disjunctive_spec_identity(&objectives, &thresholds, &clause_sizes),
        disjunctive_spec_identity(&spec_ulp, &thresholds, &clause_sizes),
        "spec identity must retain every f32 bit"
    );
}

#[test]
fn rng_consuming_root_alpha_fails_closed_to_no_cache_20260721() {
    let (graph, input) = restart_graph();
    let mut config = deterministic_alpha_config();
    config.alpha_config.gradient_method = GradientMethod::Spsa;
    let cache = InputSplitRootBoundsCache::new(Vec::new(), None);
    assert!(
        root_bounds_cache_key(&cache, &graph, &input, &config, None).is_none(),
        "restart-seeded SPSA must bypass root reuse"
    );
}

#[test]
fn unsupported_forward_image_configuration_falls_back_without_aborting() {
    let kernel = ArrayD::from_shape_vec(IxDyn(&[2, 1, 1, 1]), vec![1.0_f32, 1.0]).unwrap();
    let conv = Conv2dLayer::with_input_shape_full(kernel, None, (1, 1), (0, 0), 2, 1, 1).unwrap();
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("conv", Layer::Conv2d(conv)));
    graph.set_output("conv");
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 1, 1]), vec![-1.0_f32, -2.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2, 1, 1]), vec![1.0_f32, 2.0]).unwrap(),
    )
    .unwrap();
    let config = BetaCrownConfig {
        use_alpha_crown: false,
        use_forward_bounds: true,
        ..BetaCrownConfig::default()
    };

    let (bounds, alpha) = collect_input_split_root_node_bounds(
        &graph,
        &input,
        &config,
        None,
        None,
        "forward fail-closed test",
        None,
    )
    .unwrap();
    assert!(bounds.is_none());
    assert!(alpha.is_none());
}
