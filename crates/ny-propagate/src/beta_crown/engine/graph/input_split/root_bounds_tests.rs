// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::{
    Conv2dLayer, ConvTranspose2dLayer, GraphNode, Layer, LinearLayer, ReLULayer, TanhLayer,
};
use ndarray::{arr1, arr2, ArrayD, IxDyn};
use ny_core::{GpuCrownBackward, GpuCrownLayer, GpuCrownResult};
use ny_test_utils::{env::lock_env, CountingGemmEngine};
use std::cell::Cell;
use std::collections::BTreeMap;
use std::time::Duration;

struct RootJointFactoryCapabilityMock {
    sound: bool,
    bounded_joint: bool,
    sound_fold_max_rows: usize,
}

impl GemmEngine for RootJointFactoryCapabilityMock {
    fn gemm_f32(&self, m: usize, _k: usize, n: usize, _a: &[f32], _b: &[f32]) -> Result<Vec<f32>> {
        Ok(vec![0.0; m * n])
    }

    fn as_gpu_crown_backward(&self) -> Option<&dyn GpuCrownBackward> {
        Some(self)
    }
}

impl GpuCrownBackward for RootJointFactoryCapabilityMock {
    fn crown_backward_gpu(
        &self,
        _layers: &[GpuCrownLayer],
        _spec: &[f32],
        num_specs: usize,
        _input_lower: &[f32],
        _input_upper: &[f32],
    ) -> Result<GpuCrownResult> {
        Ok(GpuCrownResult {
            lower_bounds: vec![0.0; num_specs],
            upper_bounds: vec![0.0; num_specs],
        })
    }

    fn provides_sound_gpu_crown(&self) -> bool {
        self.sound
    }

    fn provides_deadline_bounded_joint_alpha_gradient_resident(&self) -> bool {
        self.bounded_joint
    }

    fn deadline_bounded_resnet_sound_max_rows(&self) -> usize {
        self.sound_fold_max_rows
    }
}

#[test]
fn root_joint_deadline_engine_none_reaches_qualified_factory_at_bounded_k() {
    // Exact scored recipe preconditions: M=1, D=1, K<=8. The production
    // caller evaluates the absent local engine before entering the same lazy
    // factory branch exercised here.
    let configured_master = true;
    let configured_deadline_ascent = true;
    let configured_max_sel = ny_core::DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS;
    assert!(
        configured_master
            && configured_deadline_ascent
            && configured_max_sel <= ny_core::DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS
    );

    let engine: Option<&dyn GemmEngine> = None;
    let local = engine.and_then(|candidate| {
        crate::sound_gpu_gate::with_root_joint_deadline_gpu(candidate, |_| 1usize)
    });
    let factory = RootJointFactoryCapabilityMock {
        sound: true,
        bounded_joint: true,
        sound_fold_max_rows: ny_core::DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS,
    };
    let factory_lookups = Cell::new(0usize);
    let tightener_calls = Cell::new(0usize);
    let dispatch = finish_root_joint_deadline_dispatch(local, || {
        factory_lookups.set(factory_lookups.get() + 1);
        Ok(Some(crate::sound_gpu_gate::with_root_joint_deadline_gpu(
            &factory,
            |selected| {
                tightener_calls.set(tightener_calls.get() + 1);
                selected.sound_fold_max_rows()
            },
        )))
    });

    assert_eq!(factory_lookups.get(), 1);
    assert_eq!(tightener_calls.get(), 1);
    assert_eq!(
        dispatch.terminal_receipt(),
        (
            1,
            crate::execution_telemetry::RootJointIntermAlphaOutcome::Completed
        )
    );
    assert!(matches!(
        dispatch,
        RootJointDeadlineDispatch::Completed(rows)
            if rows == ny_core::DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS
    ));
}

#[test]
fn root_joint_deadline_missing_exact_capability_is_sound_no_op() {
    let unqualified = RootJointFactoryCapabilityMock {
        sound: true,
        bounded_joint: false,
        sound_fold_max_rows: ny_core::DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS,
    };
    let tightener_calls = Cell::new(0usize);
    let dispatch = finish_root_joint_deadline_dispatch(None::<usize>, || {
        Ok(Some(crate::sound_gpu_gate::with_root_joint_deadline_gpu(
            &unqualified,
            |_| {
                tightener_calls.set(tightener_calls.get() + 1);
                7usize
            },
        )))
    });

    assert_eq!(tightener_calls.get(), 0, "no bound producer may run");
    assert_eq!(
        dispatch.terminal_receipt(),
        (
            0,
            crate::execution_telemetry::RootJointIntermAlphaOutcome::CapabilityUnavailable
        )
    );
    assert!(matches!(
        dispatch,
        RootJointDeadlineDispatch::CapabilityUnavailable
    ));
}

#[test]
fn root_joint_deadline_local_completion_owns_the_only_attempt() {
    let dispatch = finish_root_joint_deadline_dispatch(Some(23usize), || {
        panic!("a completed local attempt must not observe the factory")
    });
    assert_eq!(
        dispatch.terminal_receipt(),
        (
            1,
            crate::execution_telemetry::RootJointIntermAlphaOutcome::Completed
        )
    );
    assert!(matches!(dispatch, RootJointDeadlineDispatch::Completed(23)));
}

#[test]
fn root_joint_deadline_factory_refusal_records_zero_attempts() {
    let dispatch = finish_root_joint_deadline_dispatch::<usize>(None, || {
        Err(NyError::DeadlineExceeded(
            "test factory deadline expired".to_string(),
        ))
    });
    assert_eq!(
        dispatch.terminal_receipt(),
        (
            0,
            crate::execution_telemetry::RootJointIntermAlphaOutcome::ExecutionRefused
        )
    );
    assert!(matches!(
        dispatch,
        RootJointDeadlineDispatch::ExecutionRefused(NyError::DeadlineExceeded(_))
    ));
}

#[test]
fn root_joint_scored_authority_caps_the_historical_grace_request() {
    let now = Instant::now();
    let requested = Duration::from_mins(1);
    let caller = now + Duration::from_secs(20);
    let overall = now + Duration::from_secs(9);
    let later_scored_wall = now + Duration::from_secs(30);

    assert_eq!(
        input_split_root_authority_deadline(Some(caller), Some(overall)),
        Some(overall),
        "a stricter scored wall must beat a longer effort/caller horizon"
    );
    assert_eq!(
        input_split_root_authority_deadline(Some(caller), None),
        Some(caller),
        "ordinary scored routes carry their caller deadline without a cache"
    );
    assert_eq!(
        input_split_root_authority_deadline(Some(caller), Some(later_scored_wall)),
        Some(caller),
        "a route-local BaB boundary must preserve its post-BaB reserve"
    );
    assert_eq!(
        input_split_root_authority_deadline(None, Some(overall)),
        Some(overall),
        "an explicit scored wall is authoritative even without a caller horizon"
    );
    assert_eq!(
        input_split_root_authority_deadline(None, None),
        None,
        "only a genuinely standalone call retains unbounded research grace"
    );

    assert_eq!(
        root_joint_pass_deadline(now, requested, Some(overall)),
        Some(overall),
        "a typed scored treatment may not extend the overall instance wall"
    );
    assert_eq!(
        root_joint_pass_deadline(now, requested, None),
        Some(now + requested),
        "standalone research calls retain the historical explicit grace"
    );

    let expired = now.checked_sub(Duration::from_secs(1)).unwrap();
    assert_eq!(
        root_joint_pass_deadline(now, requested, Some(expired)),
        Some(expired),
        "an expired overall authority must stay expired"
    );
    assert_eq!(
        root_joint_pass_deadline(now, Duration::MAX, None),
        None,
        "an overflowing unbounded research request fails closed"
    );
}

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
        overall_deadline,
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
        None,
        "forward fail-closed test",
        None,
    )
    .unwrap();
    assert!(bounds.is_none());
    assert!(alpha.is_none());
}

// ---------------------------------------------------------------------------
// #crown-root-map: the `crown` arm publishing the complete cGAN cascade map
// ---------------------------------------------------------------------------

/// The shipped cgan_2023 shape on the `crown` arm: no alpha and no
/// forward-linear authority, the complete cascade armed, and BOTH publication
/// guards up — with the consumer key left at its dark default.
fn crown_arm_typed_cgan_config() -> BetaCrownConfig {
    let mut config = BetaCrownConfig {
        use_alpha_crown: false,
        use_forward_bounds: false,
        input_split_ibp_enhancement: true,
        input_split_stacked_rebound: true,
        ..BetaCrownConfig::default()
    };
    config.alpha_config.fix_interm_bounds = true;
    config.alpha_config.cgan_complete_crown_ibp_root = true;
    config
}

fn assert_bounds_map_bits_eq(
    left: &HashMap<String, BoundedTensor>,
    right: &HashMap<String, BoundedTensor>,
    label: &str,
) {
    let mut left_names: Vec<&String> = left.keys().collect();
    left_names.sort();
    let mut right_names: Vec<&String> = right.keys().collect();
    right_names.sort();
    assert_eq!(left_names, right_names, "{label}: node set");
    for (name, left_bounds) in left {
        let right_bounds = &right[name];
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
            "{label}: {name} lower bits"
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
            "{label}: {name} upper bits"
        );
    }
}

/// Byte-identical default: with the key unset the `crown` arm returns the
/// historical `(None, None)` AND never enters the cascade, so an unarmed
/// scored run pays nothing for the mechanism existing.
#[ntest::timeout(10000)]
#[test]
fn crown_arm_without_crown_root_map_key_publishes_nothing_and_pays_nothing() {
    use crate::network::CganCompleteCollectionEntryCounter;

    // The typed cascade's routing reads NY_* runtime options. The harness
    // environment carries none; holding the env lock keeps any concurrent
    // editor from changing that mid-test.
    let _env_lock = lock_env();
    let (graph, input) = typed_cgan_restart_graph();
    let config = crown_arm_typed_cgan_config();
    assert!(
        !config.alpha_config.cgan_crown_root_map,
        "the consumer key must default dark"
    );
    let exec_order = graph.exec_order().expect("exec order");
    assert!(
        graph.cgan_complete_crown_ibp_root_eligible(&config.alpha_config, exec_order),
        "fixture must be typed-eligible so the refusal is the key's, not the graph's"
    );

    let entries = CganCompleteCollectionEntryCounter::start();
    let (bounds, alpha) = collect_input_split_root_node_bounds(
        &graph,
        &input,
        &config,
        None,
        None,
        None,
        "crown arm dark",
        None,
    )
    .expect("the crown arm never fails the run");
    assert!(bounds.is_none(), "dark default must publish no root map");
    assert!(alpha.is_none());
    assert_eq!(
        entries.entries(),
        0,
        "dark default must not enter the complete cascade"
    );
}

/// Armed under both guards, the `crown` arm publishes exactly the map the
/// alpha arm's lifted Step 1 produces (same collector, same L2 contract), bit
/// for bit, with no alpha state, in one cascade transaction — and that map is
/// shrink-only against plain IBP, which is what makes publishing it a
/// tightening under the two guards.
#[ntest::timeout(10000)]
#[test]
fn crown_arm_publishes_the_alpha_arms_step1_map_under_both_guards() {
    use crate::network::CganCompleteCollectionEntryCounter;

    // The typed cascade's routing reads NY_* runtime options. The harness
    // environment carries none; holding the env lock keeps any concurrent
    // editor from changing that mid-test.
    let _env_lock = lock_env();
    let (graph, input) = typed_cgan_restart_graph();
    let mut config = crown_arm_typed_cgan_config();
    config.alpha_config.cgan_crown_root_map = true;

    let entries = CganCompleteCollectionEntryCounter::start();
    let (bounds, alpha) = collect_input_split_root_node_bounds(
        &graph,
        &input,
        &config,
        None,
        None,
        None,
        "crown arm armed",
        None,
    )
    .expect("the crown arm never fails the run");
    let bounds = bounds.expect("armed crown arm publishes the cascade map");
    assert!(alpha.is_none(), "the crown arm carries no alpha authority");
    assert!(!bounds.is_empty());
    assert_eq!(
        entries.entries(),
        1,
        "exactly one complete-cascade transaction"
    );

    let exec_order = graph.exec_order().expect("exec order");
    let (expected, source) = {
        let _l2_lever_off = crate::l2_lever_gate::L2LeverGuard::disabled();
        graph
            .collect_alpha_reference_bounds_with_engine_and_source(
                &input,
                &config.alpha_config,
                None,
                exec_order,
            )
            .expect("reference collection")
    };
    assert!(
        source.is_cgan_complete_crown_ibp(),
        "fixture must route through the typed cascade: {source:?}"
    );
    assert_eq!(entries.entries(), 2);
    assert_bounds_map_bits_eq(&bounds, &expected, "crown arm vs alpha Step 1");

    let ibp = graph
        .collect_node_bounds_with_engine(&input, None)
        .expect("reference IBP collection");
    for (name, published) in &bounds {
        let Some(reference) = ibp.get(name) else {
            continue;
        };
        for ((&l, &u), (&ibp_l, &ibp_u)) in published
            .lower()
            .iter()
            .zip(published.upper().iter())
            .zip(reference.lower().iter().zip(reference.upper().iter()))
        {
            assert!(l >= ibp_l, "{name}: published lower widened past IBP");
            assert!(u <= ibp_u, "{name}: published upper widened past IBP");
        }
    }
}

/// The two publication guards are NOT implied by each other (see
/// `#root-map-two-guards`): with either down the armed key must refuse, and
/// must refuse BEFORE paying the cascade.
#[ntest::timeout(10000)]
#[test]
fn crown_arm_refuses_when_either_publication_guard_is_down() {
    use crate::network::CganCompleteCollectionEntryCounter;

    // The typed cascade's routing reads NY_* runtime options. The harness
    // environment carries none; holding the env lock keeps any concurrent
    // editor from changing that mid-test.
    let _env_lock = lock_env();
    let (graph, input) = typed_cgan_restart_graph();
    for (label, ibp_enhancement, stacked_rebound) in [
        ("stacked_rebound down", true, false),
        ("ibp_enhancement down", false, true),
        ("both guards down", false, false),
    ] {
        let mut config = crown_arm_typed_cgan_config();
        config.alpha_config.cgan_crown_root_map = true;
        config.input_split_ibp_enhancement = ibp_enhancement;
        config.input_split_stacked_rebound = stacked_rebound;

        let entries = CganCompleteCollectionEntryCounter::start();
        let (bounds, alpha) = collect_input_split_root_node_bounds(
            &graph, &input, &config, None, None, None, label, None,
        )
        .expect("a guard refusal never fails the run");
        assert!(
            bounds.is_none(),
            "{label}: a guard refusal must publish nothing"
        );
        assert!(alpha.is_none(), "{label}");
        assert_eq!(
            entries.entries(),
            0,
            "{label}: a guard refusal must not pay the cascade"
        );
    }
}

/// A grouped restart reuses the crown arm's published map exactly — one
/// cascade transaction, bit-identical maps. A no-alpha root is
/// restart-deterministic, so the WHOLE-RESULT tier is the one consulted (it is
/// keyed before the typed tier); the typed tier is still keyed for the armed
/// crown arm, and for nothing the arm would refuse.
#[ntest::timeout(10000)]
#[test]
fn crown_arm_second_restart_reuses_the_cascade_transaction() {
    use crate::network::CganCompleteCollectionEntryCounter;

    // The typed cascade's routing reads NY_* runtime options. The harness
    // environment carries none; holding the env lock keeps any concurrent
    // editor from changing that mid-test.
    let _env_lock = lock_env();
    let (graph, input) = typed_cgan_restart_graph();
    let mut config = crown_arm_typed_cgan_config();
    config.alpha_config.cgan_crown_root_map = true;

    let cache = InputSplitRootBoundsCache::new(
        disjunctive_spec_identity(&[vec![1.0; 9]], &[0.0], &[1]),
        None,
    );
    assert!(
        root_bounds_cache_key(&cache, &graph, &input, &config, None).is_some(),
        "a no-alpha root is restart-deterministic, so the whole-result tier applies"
    );
    assert!(
        typed_reference_map_cache_key(&cache, &graph, &input, &config, None).is_some(),
        "the armed crown arm also keys the deterministic typed-map tier"
    );
    let mut dark = config.clone();
    dark.alpha_config.cgan_crown_root_map = false;
    assert!(
        typed_reference_map_cache_key(&cache, &graph, &input, &dark, None).is_none(),
        "the dark crown arm has nothing to cache"
    );
    let mut guard_down = config.clone();
    guard_down.input_split_stacked_rebound = false;
    assert!(
        typed_reference_map_cache_key(&cache, &graph, &input, &guard_down, None).is_none(),
        "a guard refusal publishes nothing, so nothing of it may be cached"
    );

    let entries = CganCompleteCollectionEntryCounter::start();
    let first = collect_input_split_root_node_bounds(
        &graph,
        &input,
        &config,
        None,
        None,
        None,
        "crown restart #1",
        Some((&cache, None)),
    )
    .expect("first crown restart");
    let second = collect_input_split_root_node_bounds(
        &graph,
        &input,
        &config,
        None,
        None,
        None,
        "crown restart #2",
        Some((&cache, None)),
    )
    .expect("second crown restart");

    assert_eq!(
        entries.entries(),
        1,
        "both restarts must share one complete cascade transaction"
    );
    assert_eq!(cache.collections(), 1, "restart #2 must skip collection");
    assert_eq!(cache.hits(), 1, "restart #2 must hit the whole-result tier");
    assert!(first.0.is_some(), "the armed crown arm publishes the map");
    assert!(first.1.is_none(), "the crown arm carries no alpha state");
    assert_root_value_bits_eq(&first, &second);
}
