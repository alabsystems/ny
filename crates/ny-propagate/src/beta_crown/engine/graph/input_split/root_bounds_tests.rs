// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::{
    Conv2dLayer, ConvTranspose2dLayer, GraphNode, Layer, LinearLayer, ReLULayer, TanhLayer,
};
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

fn typed_cgan_restart_graph() -> (GraphNetwork, BoundedTensor) {
    let transpose = ConvTranspose2dLayer::with_input_shape(
        ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 2]), vec![1.0_f32, -0.5, 0.25, 0.75])
            .expect("transpose kernel"),
        Some(arr1(&[0.1_f32])),
        (1, 1),
        (0, 0),
        2,
        2,
    )
    .expect("conv transpose");
    let conv = Conv2dLayer::with_input_shape(
        ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 1]), vec![0.75_f32]).expect("conv kernel"),
        Some(arr1(&[-0.2_f32])),
        (1, 1),
        (0, 0),
        3,
        3,
    )
    .expect("conv");
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "convt",
        Layer::ConvTranspose2d(transpose),
    ));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["convt".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "conv",
        Layer::Conv2d(conv),
        vec!["relu".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "out",
        Layer::ReLU(ReLULayer),
        vec!["conv".to_string()],
    ));
    graph.set_output("out");
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, 2, 2]), -1.0_f32),
        ArrayD::from_elem(IxDyn(&[1, 2, 2]), 1.0_f32),
    )
    .expect("input");
    (graph, input)
}

fn typed_cgan_tanh_restart_graph() -> (GraphNetwork, BoundedTensor) {
    let (mut graph, input) = typed_cgan_restart_graph();
    graph.add_node(GraphNode::new(
        "tanh",
        Layer::Tanh(TanhLayer),
        vec!["conv".to_string()],
    ));
    let post = Conv2dLayer::with_input_shape(
        ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 1]), vec![-0.75_f32]).expect("post-Tanh kernel"),
        Some(arr1(&[0.05_f32])),
        (1, 1),
        (0, 0),
        3,
        3,
    )
    .expect("post-Tanh conv");
    let output = graph.nodes.get_mut("out").expect("output node");
    output.layer = Layer::Conv2d(post);
    output.inputs = vec!["tanh".to_string()];
    graph.set_output("out");
    (graph, input)
}

fn tanh_alpha_bits(state: &GraphAlphaState) -> Vec<u32> {
    let alpha = state
        .monotone_s_shaped_alphas
        .get("tanh")
        .expect("Tanh alpha state");
    [
        &alpha.tp_pos,
        &alpha.tp_neg,
        &alpha.tp_both_lower,
        &alpha.tp_both_upper,
    ]
    .into_iter()
    .flat_map(|params| {
        params
            .lower_path
            .iter()
            .chain(params.upper_path.iter())
            .map(|value| value.to_bits())
    })
    .collect()
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

#[ntest::timeout(10000)]
#[test]
fn typed_cgan_complete_second_grouped_restart_reuses_single_transaction() {
    use crate::network::CganCompleteCollectionEntryCounter;

    ny_test_utils::env::with_env_edits(|env| {
        for key in [
            "NY_NO_FORWARD_LINEAR_REF",
            "NY_NO_FORWARD_LINEAR_CONV_TRANSPOSE_REF",
            "NY_CROWN_IBP_SPARSE_RELU_ROWS",
            "NY_CROWN_IBP_DOWNSTREAM_RESWEEP",
            "NY_CROWN_DEADLINE_CHUNK_SALVAGE",
            "NY_RNG_SEED",
        ] {
            env.remove(key);
        }
        let (graph, input) = typed_cgan_restart_graph();
        let mut config = BetaCrownConfig {
            use_alpha_crown: true,
            ..BetaCrownConfig::default()
        };
        config.alpha_config.iterations = 0;
        config.alpha_config.gradient_method = GradientMethod::AnalyticChain;
        config.alpha_config.fix_interm_bounds = true;
        config.alpha_config.adaptive_skip = false;
        config.alpha_config.cgan_complete_crown_ibp_root = true;

        let cache = InputSplitRootBoundsCache::new(
            disjunctive_spec_identity(&[vec![1.0; 9]], &[0.0], &[1]),
            None,
        );
        let entries = CganCompleteCollectionEntryCounter::start();
        let first = collect_input_split_root_node_bounds(
            &graph,
            &input,
            &config,
            None,
            None,
            "typed restart #1",
            Some((&cache, None)),
        )
        .expect("first typed restart");
        let second = collect_input_split_root_node_bounds(
            &graph,
            &input,
            &config,
            None,
            None,
            "typed restart #2",
            Some((&cache, None)),
        )
        .expect("second typed restart");

        assert_eq!(entries.entries(), 1);
        assert_eq!(cache.collections(), 1);
        assert_eq!(cache.hits(), 1);
        assert_root_value_bits_eq(&first, &second);
    });
}

#[ntest::timeout(10000)]
#[test]
fn typed_cgan_tanh_restarts_reuse_only_map_and_recompute_seeded_alpha() {
    use crate::network::CganCompleteCollectionEntryCounter;

    ny_test_utils::env::with_env_edits(|env| {
        for key in [
            "NY_NO_FORWARD_LINEAR_REF",
            "NY_NO_FORWARD_LINEAR_CONV_TRANSPOSE_REF",
            "NY_CROWN_IBP_SPARSE_RELU_ROWS",
            "NY_CROWN_IBP_DOWNSTREAM_RESWEEP",
            "NY_CROWN_DEADLINE_CHUNK_SALVAGE",
            "NY_RNG_SEED",
        ] {
            env.remove(key);
        }
        let (graph, input) = typed_cgan_tanh_restart_graph();
        let mut config = BetaCrownConfig {
            use_alpha_crown: true,
            ..BetaCrownConfig::default()
        };
        config.alpha_config.iterations = 3;
        config.alpha_config.spsa_samples = 1;
        config.alpha_config.early_stop_patience = usize::MAX;
        config.alpha_config.gradient_method = GradientMethod::AnalyticChain;
        config.alpha_config.fix_interm_bounds = true;
        config.alpha_config.adaptive_skip = false;
        config.alpha_config.adaptive_skip_pilot = false;
        config.alpha_config.cgan_complete_crown_ibp_root = true;

        let cache = InputSplitRootBoundsCache::new(
            disjunctive_spec_identity(&[vec![1.0; 9]], &[0.0], &[1]),
            None,
        );
        assert!(
            root_bounds_cache_key(&cache, &graph, &input, &config, None).is_none(),
            "an iteration-bearing Tanh root must not cache the whole alpha result"
        );
        assert!(
            typed_reference_map_cache_key(&cache, &graph, &input, &config, None).is_some(),
            "the deterministic typed reference map should remain cacheable"
        );

        let entries = CganCompleteCollectionEntryCounter::start();
        let first = {
            let _seed = crate::set_rng_restart_offset(0);
            collect_input_split_root_node_bounds(
                &graph,
                &input,
                &config,
                None,
                None,
                "typed Tanh restart #1",
                Some((&cache, None)),
            )
            .expect("first typed Tanh restart")
        };
        let ordinary_error = graph
            .collect_forward_linear_bounds_dag_cached(&input, None, None)
            .expect_err("typed Tanh cache state must not expand an ordinary request");
        assert!(matches!(
            ordinary_error,
            NyError::UnsupportedConfiguration(_)
        ));
        let second = {
            let _seed = crate::set_rng_restart_offset(1);
            collect_input_split_root_node_bounds(
                &graph,
                &input,
                &config,
                None,
                None,
                "typed Tanh restart #2",
                Some((&cache, None)),
            )
            .expect("second typed Tanh restart")
        };

        assert_eq!(
            entries.entries(),
            1,
            "both restarts must share one complete typed-map transaction"
        );
        assert_eq!(
            cache.collections(),
            0,
            "the whole-result cache must remain disabled for seeded Tanh alpha"
        );
        assert_eq!(cache.typed_reference_collections(), 1);
        assert_eq!(cache.typed_reference_hits(), 1);

        let first_alpha = first.1.as_ref().expect("first Tanh alpha state");
        let second_alpha = second.1.as_ref().expect("second Tanh alpha state");
        assert_ne!(
            tanh_alpha_bits(first_alpha),
            tanh_alpha_bits(second_alpha),
            "restart-specific Tanh alpha initialization must be recomputed"
        );
    });
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

    let mut refresh_ulp_config = config.clone();
    refresh_ulp_config.alpha_config.reference_refresh_fraction = f32::from_bits(
        refresh_ulp_config
            .alpha_config
            .reference_refresh_fraction
            .to_bits()
            + 1,
    );
    let refresh_ulp =
        root_bounds_cache_key(&cache, &graph, &input, &refresh_ulp_config, Some(&engine_a))
            .expect("refresh-fraction ULP option key");
    assert!(
        base != refresh_ulp,
        "one refresh-fraction config ULP must miss"
    );

    let mut deadline_fallback_config = config.clone();
    deadline_fallback_config
        .alpha_config
        .forward_linear_deadline_fallback_to_ibp = true;
    let deadline_fallback = root_bounds_cache_key(
        &cache,
        &graph,
        &input,
        &deadline_fallback_config,
        Some(&engine_a),
    )
    .expect("forward-linear deadline-fallback option key");
    assert!(
        base != deadline_fallback,
        "changing the forward-linear deadline-fallback policy must miss"
    );

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
