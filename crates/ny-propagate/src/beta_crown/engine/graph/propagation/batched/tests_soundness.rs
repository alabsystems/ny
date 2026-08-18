// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{arr1, arr2, Array2};
use ny_core::NaiveCpuGemmEngine;
use std::time::Duration;

use super::{
    interm_refine::HermeticSoundGpuCrownEngine, BatchedBackwardContext, BetaCrownVerifier,
};
use crate::batched_domain::BatchedDomains;
use crate::beta_crown::{BetaCrownConfig, GraphBabDomain};
use crate::{BoundedTensor, GraphNetwork, GraphNode, Layer, LinearLayer, ReLULayer};

const SAMPLE_TOLERANCE_NY: f32 = 1.0e-5;

fn build_single_relu_graph_for_batched_soundness_tests() -> GraphNetwork {
    let linear1 = LinearLayer::new(arr2(&[[1.0]]), Some(arr1(&[0.0]))).expect("valid linear1");
    let linear2 = LinearLayer::new(arr2(&[[1.0]]), Some(arr1(&[0.0]))).expect("valid linear2");

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

fn single_relu_graph_output_2769(x: f32) -> f32 {
    x.max(0.0)
}

fn assert_single_relu_graph_bounds_contain_samples_2769(bounds: &BoundedTensor) {
    let flat = bounds.flatten();
    let lower = flat.lower()[[0]];
    let upper = flat.upper()[[0]];

    for i in -10..=10 {
        let x = i as f32 / 10.0;
        let y = single_relu_graph_output_2769(x);
        assert!(
            y >= lower - SAMPLE_TOLERANCE_NY && y <= upper + SAMPLE_TOLERANCE_NY,
            "sample {x} -> {y} must stay within [{lower}, {upper}]",
        );
    }
}

fn root_graph_domain_2769(graph: &GraphNetwork) -> GraphBabDomain {
    let input =
        BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn()).expect("valid input");
    let initial_bounds = graph
        .collect_node_bounds(&input)
        .expect("graph bounds should collect");
    GraphBabDomain::root(initial_bounds, -10.0, 10.0, &input, false)
        .expect("root domain with finite bounds should not fail")
}

#[test]
fn batched_forward_adapter_preserves_expired_deadline() {
    let graph = build_single_relu_graph_for_batched_soundness_tests();
    let root = root_graph_domain_2769(&graph);
    let domains = vec![&root];
    let layer_names = vec!["relu1".to_string()];
    let batched =
        BatchedDomains::from_graph_domains(&domains, &layer_names).expect("batched domains");
    let ctx = BatchedBackwardContext::from_domains(&domains, &batched).expect("valid context");
    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        timeout: Duration::ZERO,
        ..Default::default()
    });

    let error = verifier
        .propagate_crown_batched_with_context(&graph, &ctx, &[1.0_f32], &NaiveCpuGemmEngine)
        .expect_err("expired constrained forward must refuse");
    assert!(
        error.is_deadline_exceeded(),
        "batched adapter must preserve DeadlineExceeded, got {error:?}"
    );
}

#[test]
fn batched_dense_spec_forward_adapter_preserves_expired_deadline() {
    let graph = build_single_relu_graph_for_batched_soundness_tests();
    let root = root_graph_domain_2769(&graph);
    let domains = vec![&root];
    let layer_names = vec!["relu1".to_string()];
    let batched =
        BatchedDomains::from_graph_domains(&domains, &layer_names).expect("batched domains");
    let ctx = BatchedBackwardContext::from_domains(&domains, &batched).expect("valid context");
    let spec_matrix = Array2::from_shape_vec((1, 1), vec![1.0_f32]).expect("valid spec matrix");
    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        timeout: Duration::ZERO,
        ..Default::default()
    });

    let error = verifier
        .propagate_crown_batched_with_context_specs(&graph, &ctx, &spec_matrix, &NaiveCpuGemmEngine)
        .expect_err("expired dense-spec forward must refuse");
    assert!(
        error.is_deadline_exceeded(),
        "dense-spec adapter must preserve DeadlineExceeded, got {error:?}"
    );
}

#[test]
fn test_concretize_batched_results_contains_samples_2769() {
    let graph = build_single_relu_graph_for_batched_soundness_tests();
    let root = root_graph_domain_2769(&graph);
    let domains = vec![&root];
    let layer_names = vec!["relu1".to_string()];
    let batched =
        BatchedDomains::from_graph_domains(&domains, &layer_names).expect("batched domains");
    let ctx = BatchedBackwardContext::from_domains(&domains, &batched).expect("valid context");
    let objective = vec![1.0_f32];

    let results = BetaCrownVerifier::new(BetaCrownConfig::default())
        .propagate_crown_batched_with_context(&graph, &ctx, &objective, &NaiveCpuGemmEngine)
        .expect("batched context propagation should succeed");

    assert_eq!(results.len(), 1, "expected one domain result");
    assert_single_relu_graph_bounds_contain_samples_2769(&results[0].0);
}

#[test]
fn test_propagate_crown_with_batched_domains_full_contains_samples_2769() {
    let graph = build_single_relu_graph_for_batched_soundness_tests();
    let root = root_graph_domain_2769(&graph);
    let domains = vec![&root];
    let layer_names = vec!["relu1".to_string()];
    let batched =
        BatchedDomains::from_graph_domains(&domains, &layer_names).expect("batched domains");
    let objective = vec![1.0_f32];

    let results = BetaCrownVerifier::new(BetaCrownConfig::default())
        .propagate_crown_with_batched_domains_full(
            &graph,
            &domains,
            &batched,
            &objective,
            &NaiveCpuGemmEngine,
        )
        .expect("full batched-domain propagation should succeed");

    let output_bounds = results[0]
        .as_ref()
        .expect("root domain should produce batched CROWN output");
    assert_single_relu_graph_bounds_contain_samples_2769(&output_bounds.0);
}

/// Dense-spec batched backward through `propagate_crown_batched_with_context_specs`.
/// Exercises the full dense-spec pipeline: forward pass, spec-seeded backward,
/// and concretize with `input_linear` preservation.
/// Part of #4116 — resolves dead-code warnings for staging API.
#[test]
fn test_batched_specs_contains_samples_and_preserves_input_linear_4116() {
    let graph = build_single_relu_graph_for_batched_soundness_tests();
    let root = root_graph_domain_2769(&graph);
    let domains = vec![&root];
    let layer_names = vec!["relu1".to_string()];
    let batched =
        BatchedDomains::from_graph_domains(&domains, &layer_names).expect("batched domains");
    let ctx = BatchedBackwardContext::from_domains(&domains, &batched).expect("valid context");

    // Single-row identity spec matrix equivalent to objective [1.0].
    let spec_matrix = Array2::from_shape_vec((1, 1), vec![1.0_f32]).expect("valid spec matrix");

    let results = BetaCrownVerifier::new(BetaCrownConfig::default())
        .propagate_crown_batched_with_context_specs(&graph, &ctx, &spec_matrix, &NaiveCpuGemmEngine)
        .expect("spec batched context propagation should succeed");

    assert_eq!(results.len(), 1, "expected one domain result");
    assert_single_relu_graph_bounds_contain_samples_2769(&results[0].output_bounds);

    // Dense-spec path must preserve input linear bounds when CROWN succeeds.
    assert!(
        results[0].input_linear.is_some(),
        "input_linear must be Some when CROWN backward succeeds without fallback",
    );
}

/// Dense-spec batched backward with lA capture.
/// Exercises `propagate_crown_batched_with_context_specs_capture_la`.
/// Part of #4116 — resolves dead-code for `BatchedSpecBackwardResult`.
#[test]
fn test_batched_specs_capture_la_returns_result_4116() {
    let graph = build_single_relu_graph_for_batched_soundness_tests();
    let root = root_graph_domain_2769(&graph);
    let domains = vec![&root];
    let layer_names = vec!["relu1".to_string()];
    let batched =
        BatchedDomains::from_graph_domains(&domains, &layer_names).expect("batched domains");
    let ctx = BatchedBackwardContext::from_domains(&domains, &batched).expect("valid context");

    let spec_matrix = Array2::from_shape_vec((1, 1), vec![1.0_f32]).expect("valid spec matrix");

    let result = BetaCrownVerifier::new(BetaCrownConfig::default())
        .propagate_crown_batched_with_context_specs_capture_la(
            &graph,
            &ctx,
            &spec_matrix,
            &NaiveCpuGemmEngine,
        )
        .expect("spec batched with lA capture should succeed");

    assert_eq!(result.results.len(), 1, "expected one domain result");
    assert_single_relu_graph_bounds_contain_samples_2769(&result.results[0].output_bounds);
}

/// #lsnc-shared-fwd parity: the input-split batched forward's SHARED-map fast
/// path must produce BIT-IDENTICAL certified output bounds to the historical
/// per-domain node-bounds clone path. Both legs run the production
/// `batched_forward_then_backward_specs` (via
/// `propagate_crown_batched_with_context_specs`); the ONLY difference is whether
/// every domain's `base_bounds` entry aliases ONE shared warmup map (fast path —
/// `std::ptr::eq` true ⇒ no per-domain clone) or points at a DISTINCT identical
/// clone (`std::ptr::eq` false ⇒ the historical per-domain clone fallback). The
/// input-split path emits CERTIFIED bounds, so with identical inputs any
/// deviation is a soundness bug: equality must be EXACT (bit-identical f32), not
/// approximate.
#[test]
fn test_input_split_shared_fwd_bit_identical_to_per_domain_clone() {
    use std::collections::HashMap;
    use std::sync::Arc;

    use crate::batched_domain::{BatchedDomainOptions, BatchedDomainsBuilder};
    use crate::beta_crown::branching::GraphSplitHistory;

    // 2 -> 3 (ReLU) -> 2 MLP with genuinely unstable hidden neurons over [-1,1]^2
    // (so the shared warmup pre-activation bounds actually drive ReLU relaxations).
    let linear1 = LinearLayer::new(
        arr2(&[[1.0, -1.0], [0.5, 0.5], [-1.0, 1.0]]),
        Some(arr1(&[0.0, 0.0, 0.0])),
    )
    .expect("valid linear1");
    let linear2 = LinearLayer::new(
        arr2(&[[1.0, 1.0, -1.0], [0.5, -1.0, 1.0]]),
        Some(arr1(&[0.0, 0.0])),
    )
    .expect("valid linear2");
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

    // Warmup reference bounds over the ROOT box — the "fix_interm_bounds" map
    // input-split shares (read-only) across every sub-box.
    let root = BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn())
        .expect("root box");
    let warmup: HashMap<String, BoundedTensor> = graph
        .collect_node_bounds(&root)
        .expect("warmup node bounds");
    let shared_arc: HashMap<String, Arc<BoundedTensor>> = warmup
        .iter()
        .map(|(k, v)| (k.clone(), Arc::new(v.clone())))
        .collect();

    // Three input sub-boxes of the root (input-split's disjoint children).
    let sub_boxes = [
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[0.0, 1.0]).into_dyn()).unwrap(),
        BoundedTensor::new(arr1(&[0.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap(),
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 0.0]).into_dyn()).unwrap(),
    ];
    let n = sub_boxes.len();

    let mut builder =
        BatchedDomainsBuilder::new_with_options(Vec::new(), BatchedDomainOptions::default());
    let empty_layer_bounds: HashMap<String, (ndarray::ArrayD<f32>, ndarray::ArrayD<f32>)> =
        HashMap::new();
    for b in &sub_boxes {
        builder.add_domain(
            &empty_layer_bounds,
            b.lower().clone(),
            b.upper().clone(),
            0.0,
            0.0,
            0,
            Vec::new(),
        );
    }
    let batched = builder.build().expect("batched domains");

    let empty_history = GraphSplitHistory::new();
    // 2x2 identity spec ⇒ compare both raw output dims per domain.
    let spec_matrix =
        Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).expect("valid spec matrix");
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());

    // Path A: SHARED map — every base_bounds entry aliases ONE map (ptr::eq true).
    // Relies on the default gate (NY_INPUT_SPLIT_SHARED_FWD unset ⇒ enabled).
    let ctx_shared = BatchedBackwardContext {
        batched: &batched,
        histories: vec![&empty_history; n],
        beta_states: vec![None; n],
        base_bounds: vec![Some(&shared_arc); n],
        delta_seeds: vec![None; n],
        alpha_states: vec![None; n],
        cached_la: vec![None; n],
        mul_binary_alphas: None,
    };
    let shared_results = verifier
        .propagate_crown_batched_with_context_specs(
            &graph,
            &ctx_shared,
            &spec_matrix,
            &NaiveCpuGemmEngine,
        )
        .expect("shared-fwd path should succeed");

    // Path B: per-domain DISTINCT clones (identical content, different pointers ⇒
    // ptr::eq false ⇒ the historical per-domain clone fallback).
    let per_domain_maps: Vec<HashMap<String, Arc<BoundedTensor>>> =
        (0..n).map(|_| shared_arc.clone()).collect();
    let ctx_clone = BatchedBackwardContext {
        batched: &batched,
        histories: vec![&empty_history; n],
        beta_states: vec![None; n],
        base_bounds: per_domain_maps.iter().map(Some).collect(),
        delta_seeds: vec![None; n],
        alpha_states: vec![None; n],
        cached_la: vec![None; n],
        mul_binary_alphas: None,
    };
    let clone_results = verifier
        .propagate_crown_batched_with_context_specs(
            &graph,
            &ctx_clone,
            &spec_matrix,
            &NaiveCpuGemmEngine,
        )
        .expect("per-domain-clone path should succeed");

    assert_eq!(shared_results.len(), n);
    assert_eq!(clone_results.len(), n);
    for i in 0..n {
        assert_eq!(
            shared_results[i].output_bounds.lower(),
            clone_results[i].output_bounds.lower(),
            "domain {i}: shared-fwd LOWER bounds must be bit-identical to the per-domain clone path",
        );
        assert_eq!(
            shared_results[i].output_bounds.upper(),
            clone_results[i].output_bounds.upper(),
            "domain {i}: shared-fwd UPPER bounds must be bit-identical to the per-domain clone path",
        );
    }
}

/// #lsnc-relu STEP 2 KERNEL PARITY: the DOMAIN-batched ReLU backward
/// (`ReLULayer::propagate_linear_multi_domain_relu`) must be BIT-IDENTICAL to the
/// historical per-domain scalar loop — i.e. to invoking, per domain,
/// `propagate_linear_with_alpha(..).0` (α present) or `propagate_linear_with_bounds`
/// (α absent), the exact two functions the ReLU dispatch arm calls.
///
/// The input-split path emits CERTIFIED bounds, so parity must be EXACT f32 bits on
/// the coefficients, the biases, AND the certified coefficient-error matrices — a
/// tighter batched bound would be a FALSE PROOF. This drives a deterministic LCG over
/// many batches covering: α (dual, lower≠upper) and heuristic domains; incoming
/// coefficient error present and absent; coefficients of both signs and exact zeros;
/// sign-STABLE (|a|>e) and sign-AMBIGUOUS (|a|≤e) error carries; and per-domain
/// pre-activation boxes that make each ReLU STABLE-active / STABLE-inactive / UNSTABLE
/// in DIFFERENT domains (so the box-dependent triangle slope genuinely varies per
/// domain — the reason ReLU cannot use a shared hull box).
#[test]
fn test_batched_relu_bit_identical_to_per_domain_kernel_lsnc_step2() {
    use crate::LinearBounds;
    use ndarray::{Array1, Array2};

    // Deterministic LCG in [0,1).
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut u01 = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 40) as f32) / ((1u64 << 24) as f32)
    };

    let relu = ReLULayer;
    let mut trials = 0usize;
    for shape_trial in 0..60 {
        let num_outputs = 1 + (shape_trial % 4); // 1..=4 spec rows
        let num_neurons = 1 + (shape_trial % 7); // 1..=7 neurons
        let n_domains = 3 + (shape_trial % 5); // 3..=7 domains per batch

        // Build the per-domain inputs the ReLU arm would build.
        let mut bounds_owned: Vec<LinearBounds> = Vec::with_capacity(n_domains);
        let mut pre_owned: Vec<BoundedTensor> = Vec::with_capacity(n_domains);
        let mut alpha_owned: Vec<Option<(Array1<f32>, Array1<f32>)>> =
            Vec::with_capacity(n_domains);

        for d in 0..n_domains {
            // Coefficients spanning [-2, 2] with a deliberate exact-zero stripe.
            let mk_coeff = |g: &mut dyn FnMut() -> f32| {
                Array2::from_shape_fn((num_outputs, num_neurons), |(j, i)| {
                    if (i + j + d) % 5 == 0 {
                        0.0
                    } else {
                        g() * 4.0 - 2.0
                    }
                })
            };
            let lower_a = mk_coeff(&mut u01);
            let upper_a = mk_coeff(&mut u01);
            let lower_b = Array1::from_shape_fn(num_outputs, |_| u01() * 2.0 - 1.0);
            let upper_b = Array1::from_shape_fn(num_outputs, |_| u01() * 2.0 - 1.0);
            let mut lb = LinearBounds::new(lower_a, lower_b, upper_a, upper_b)
                .expect("valid incoming bounds");

            // Attach certified coefficient error on ~half the domains. Magnitudes
            // straddle the coefficient magnitudes so BOTH sign-stable (|a|>e) and
            // sign-ambiguous (|a|≤e) carries are exercised.
            if d % 2 == 0 {
                let le = Array2::from_shape_fn((num_outputs, num_neurons), |_| u01() * 1.5);
                let ue = Array2::from_shape_fn((num_outputs, num_neurons), |_| u01() * 1.5);
                lb.set_coeff_err(le, ue);
            }
            bounds_owned.push(lb);

            // Per-domain pre-activation box: rotate which neurons are stable-active,
            // stable-inactive, or crossing so the relaxation differs across domains.
            let (mut lo, mut hi) = (
                Array1::<f32>::zeros(num_neurons),
                Array1::<f32>::zeros(num_neurons),
            );
            for i in 0..num_neurons {
                match (i + d) % 3 {
                    0 => {
                        // stable active: 0 <= l < u
                        lo[i] = u01() * 1.5;
                        hi[i] = lo[i] + 0.1 + u01();
                    }
                    1 => {
                        // stable inactive: l < u <= 0
                        hi[i] = -(u01() * 1.5);
                        lo[i] = hi[i] - 0.1 - u01();
                    }
                    _ => {
                        // crossing / unstable: l < 0 < u
                        lo[i] = -(0.1 + u01() * 2.0);
                        hi[i] = 0.1 + u01() * 2.0;
                    }
                }
            }
            pre_owned.push(
                BoundedTensor::new(lo.into_dyn(), hi.into_dyn()).expect("valid pre-activation"),
            );

            // Alpha present on ~2/3 of domains, with dual (lower≠upper) α ∈ [0,1].
            let alpha = if d % 3 != 0 {
                let al = Array1::from_shape_fn(num_neurons, |_| u01());
                let au = Array1::from_shape_fn(num_neurons, |_| u01());
                Some((al, au))
            } else {
                None
            };
            alpha_owned.push(alpha);
        }

        let bounds_refs: Vec<&LinearBounds> = bounds_owned.iter().collect();
        let pre_refs: Vec<&BoundedTensor> = pre_owned.iter().collect();

        // Batched path.
        let batched = relu
            .propagate_linear_multi_domain_relu(&bounds_refs, &pre_refs, &alpha_owned)
            .expect("batched relu ok")
            .expect("batched relu must not decline on finite contiguous inputs");
        assert_eq!(batched.len(), n_domains);

        // Reference per-domain scalar path.
        for d in 0..n_domains {
            let reference = match &alpha_owned[d] {
                Some((al, au)) => {
                    relu.propagate_linear_with_alpha(bounds_refs[d], pre_refs[d], al, Some(au))
                        .expect("alpha reference ok")
                        .0
                }
                None => relu
                    .propagate_linear_with_bounds(bounds_refs[d], pre_refs[d])
                    .expect("heuristic reference ok"),
            };
            let got = &batched[d];
            assert_eq!(
                got.lower_a(),
                reference.lower_a(),
                "shape_trial {shape_trial} domain {d}: lower_a differs (batched vs per-domain)"
            );
            assert_eq!(
                got.upper_a(),
                reference.upper_a(),
                "shape_trial {shape_trial} domain {d}: upper_a differs"
            );
            assert_eq!(
                got.lower_b(),
                reference.lower_b(),
                "shape_trial {shape_trial} domain {d}: lower_b differs"
            );
            assert_eq!(
                got.upper_b(),
                reference.upper_b(),
                "shape_trial {shape_trial} domain {d}: upper_b differs"
            );
            assert_eq!(
                got.lower_a_err(), reference.lower_a_err(),
                "shape_trial {shape_trial} domain {d}: lower_a_err differs (certified error must match bit-for-bit)"
            );
            assert_eq!(
                got.upper_a_err(),
                reference.upper_a_err(),
                "shape_trial {shape_trial} domain {d}: upper_a_err differs"
            );
            trials += 1;
        }
    }
    assert!(
        trials > 200,
        "expected a broad sweep, only compared {trials} domains"
    );
}

/// #lsnc-relu STEP 2 FULL-PIPELINE PARITY: driving the production input-split batched
/// spec backward (`propagate_crown_batched_with_context_specs`) with the batched ReLU
/// gate ON vs OFF must yield BIT-IDENTICAL certified output bounds, on a net with TWO
/// hidden ReLU layers and multiple input sub-boxes (so the certified coefficient error
/// composes across layers, and both stable and unstable ReLUs occur across the boxes).
#[test]
fn test_batched_relu_full_pipeline_bit_identical_multi_layer_lsnc_step2() {
    use std::collections::HashMap;
    use std::sync::Arc;

    use super::backward_core::force_batched_relu;
    use crate::batched_domain::{BatchedDomainOptions, BatchedDomainsBuilder};
    use crate::beta_crown::branching::GraphSplitHistory;

    // 2 -> 4 (ReLU) -> 3 (ReLU) -> 2 MLP with genuinely unstable hidden neurons.
    let linear1 = LinearLayer::new(
        arr2(&[[1.0, -1.0], [0.5, 0.9], [-1.0, 0.4], [0.3, -0.7]]),
        Some(arr1(&[0.1, -0.2, 0.0, 0.05])),
    )
    .expect("valid linear1");
    let linear2 = LinearLayer::new(
        arr2(&[
            [1.0, -0.5, 0.8, 0.2],
            [-0.6, 0.7, -0.3, 1.0],
            [0.4, 0.4, -0.9, -0.1],
        ]),
        Some(arr1(&[0.0, 0.1, -0.05])),
    )
    .expect("valid linear2");
    let linear3 = LinearLayer::new(
        arr2(&[[1.0, -1.0, 0.5], [0.3, 0.8, -0.6]]),
        Some(arr1(&[0.0, 0.0])),
    )
    .expect("valid linear3");
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

    let root = BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn())
        .expect("root box");
    let warmup: HashMap<String, BoundedTensor> = graph
        .collect_node_bounds(&root)
        .expect("warmup node bounds");
    let shared_arc: HashMap<String, Arc<BoundedTensor>> = warmup
        .iter()
        .map(|(k, v)| (k.clone(), Arc::new(v.clone())))
        .collect();

    // Six input sub-boxes tiling the root — each drives a different constrained
    // forward, so relu1/relu2 flip between stable and unstable across the boxes.
    let sub_boxes = [
        ([-1.0, -1.0], [0.0, 0.0]),
        ([0.0, -1.0], [1.0, 0.0]),
        ([-1.0, 0.0], [0.0, 1.0]),
        ([0.0, 0.0], [1.0, 1.0]),
        ([-0.5, -0.5], [0.5, 0.5]),
        ([-1.0, -1.0], [1.0, 1.0]),
    ];
    let n = sub_boxes.len();

    let mut builder =
        BatchedDomainsBuilder::new_with_options(Vec::new(), BatchedDomainOptions::default());
    let empty_layer_bounds: HashMap<String, (ndarray::ArrayD<f32>, ndarray::ArrayD<f32>)> =
        HashMap::new();
    for (lo, hi) in &sub_boxes {
        let b = BoundedTensor::new(arr1(lo).into_dyn(), arr1(hi).into_dyn()).unwrap();
        builder.add_domain(
            &empty_layer_bounds,
            b.lower().clone(),
            b.upper().clone(),
            0.0,
            0.0,
            0,
            Vec::new(),
        );
    }
    let batched = builder.build().expect("batched domains");
    let empty_history = GraphSplitHistory::new();
    let spec_matrix =
        Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).expect("valid spec matrix");
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());

    let run = || {
        let ctx = BatchedBackwardContext {
            batched: &batched,
            histories: vec![&empty_history; n],
            beta_states: vec![None; n],
            base_bounds: vec![Some(&shared_arc); n],
            delta_seeds: vec![None; n],
            alpha_states: vec![None; n],
            cached_la: vec![None; n],
            mul_binary_alphas: None,
        };
        verifier
            .propagate_crown_batched_with_context_specs(
                &graph,
                &ctx,
                &spec_matrix,
                &NaiveCpuGemmEngine,
            )
            .expect("batched spec propagation should succeed")
    };

    force_batched_relu(Some(true));
    let on = run();
    force_batched_relu(Some(false));
    let off = run();
    force_batched_relu(None); // restore env default

    assert_eq!(on.len(), n);
    assert_eq!(off.len(), n);
    for i in 0..n {
        assert_eq!(
            on[i].output_bounds.lower(),
            off[i].output_bounds.lower(),
            "domain {i}: batched-ReLU LOWER must be bit-identical to the per-domain loop",
        );
        assert_eq!(
            on[i].output_bounds.upper(),
            off[i].output_bounds.upper(),
            "domain {i}: batched-ReLU UPPER must be bit-identical to the per-domain loop",
        );
    }
}

/// #mo-beta-graft end-to-end enclosure on a PURE CONV chain with ReLU splits
/// (the metaroom shape): with `config.mo_beta_graft = true` and NO
/// `NY_BAB_CHAIN_WIDE`, the wide segment-lane ascent must engage (pure-chain
/// extraction forced for the ascent-only pass), the dense-spec bound must be
/// evaluated with the ascended β folded in, and the composed
/// elementwise-tightest bound must still ENCLOSE the true network values at
/// every premise-satisfying sample of each child subdomain.
#[test]
fn graft_composed_bound_encloses_true_forward_on_split_conv_chain_mo_graft() {
    use crate::beta_crown::branching::GraphNeuronConstraint;
    use crate::layers::Conv2dLayer;
    use ndarray::{Array, IxDyn};

    let device = HermeticSoundGpuCrownEngine::default();
    let engine: &dyn ny_core::GemmEngine = &device;
    assert!(
        engine
            .as_gpu_crown_backward()
            .is_some_and(|gpu| gpu.provides_sound_gpu_crown()),
        "hermetic graft enclosure test requires its injected sound CROWN backend"
    );

    // conv1 (1x1, w=1) -> relu1 -> conv2 (1x1, w=0.5, b=0.1) over a [1,2,2]
    // input: out_i = 0.5*relu(x_i) + 0.1 per pixel.
    let k_ident = Array::from_shape_vec(IxDyn(&[1, 1, 1, 1]), vec![1.0f32]).unwrap();
    let k_half = Array::from_shape_vec(IxDyn(&[1, 1, 1, 1]), vec![0.5f32]).unwrap();
    let conv1 = Conv2dLayer::new(k_ident, None, (1, 1), (0, 0)).expect("conv1");
    let conv2 = Conv2dLayer::new(k_half, Some(arr1(&[0.1f32])), (1, 1), (0, 0)).expect("conv2");
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("conv1", Layer::Conv2d(conv1)));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["conv1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "conv2",
        Layer::Conv2d(conv2),
        vec!["relu1".to_string()],
    ));
    graph.set_output("conv2");
    assert!(graph.has_conv_layers());

    let input = BoundedTensor::new(
        Array::from_shape_vec(IxDyn(&[1, 2, 2]), vec![-1.0f32; 4]).unwrap(),
        Array::from_shape_vec(IxDyn(&[1, 2, 2]), vec![1.0f32; 4]).unwrap(),
    )
    .expect("input box");
    let node_bounds = graph.collect_node_bounds(&input).expect("node bounds");
    let root = GraphBabDomain::root(node_bounds, -10.0, 10.0, &input, false).expect("root");

    // Production split machinery: two children of the relu1[0] split — the β
    // entries therefore correspond EXACTLY to each child's split constraint
    // (the graft soundness precondition).
    let mk_child = |is_active: bool| {
        root.with_constraint(
            &graph,
            GraphNeuronConstraint {
                node_name: "relu1".to_string(),
                neuron_idx: 0,
                is_active,
                score: 0.0,
            },
            false,
        )
        .expect("with_constraint")
        .expect("feasible child")
    };
    let active = mk_child(true);
    let inactive = mk_child(false);
    assert!(
        !active.beta_state.is_empty() && !inactive.beta_state.is_empty(),
        "split children must carry β entries for their premises"
    );
    let domains: Vec<&GraphBabDomain> = vec![&active, &inactive];
    let relu_names = vec!["relu1".to_string()];
    let batched = BatchedDomains::from_graph_domains(&domains, &relu_names).expect("batched");

    // Two spec rows over the 4 flattened outputs.
    let spec_matrix =
        Array2::from_shape_vec((2, 4), vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0])
            .expect("spec matrix");

    let thresholds = [0.0f32, 0.0];
    let row_verified = vec![vec![false, false], vec![false, false]];
    let eligible = [true, true];
    let depths = [1usize, 1];
    let beta_opt = super::GpuBetaOptSpec {
        thresholds: &thresholds,
        row_verified: &row_verified,
        eligible: &eligible,
        depths: &depths,
    };

    let run = |graft: bool| {
        let verifier = BetaCrownVerifier::new(BetaCrownConfig {
            mo_beta_graft: graft,
            ..Default::default()
        });
        verifier
            .propagate_crown_with_batched_domains_full_specs_beta_opt(
                &graph,
                &domains,
                &batched,
                &spec_matrix,
                engine,
                Some(&beta_opt),
            )
            .expect("dense-spec propagation")
    };
    let baseline = run(false);
    let out = run(true);
    assert_eq!(out.results.len(), 2);
    // The graft never warm-starts children with the wide-ascended β* (it
    // poisons descendants' dense passes — measured on metaroom 6cnn).
    assert!(
        out.optimized_betas.is_none(),
        "graft must not return β* for child warm-starting"
    );
    // Never-looser: the composed bound is baseline ∩ folded ∩ wide, so each
    // row must be at least as tight as the baseline dense bound.
    for (dom_idx, (g, b)) in out.results.iter().zip(baseline.results.iter()).enumerate() {
        let gf = g.output_bounds.flatten();
        let bf = b.output_bounds.flatten();
        for row in 0..gf.len() {
            assert!(
                gf.lower()[[row]] >= bf.lower()[[row]] - 1e-5,
                "domain {dom_idx} row {row}: graft lower {} regressed below baseline {}",
                gf.lower()[[row]],
                bf.lower()[[row]],
            );
            assert!(
                gf.upper()[[row]] <= bf.upper()[[row]] + 1e-5,
                "domain {dom_idx} row {row}: graft upper {} regressed above baseline {}",
                gf.upper()[[row]],
                bf.upper()[[row]],
            );
        }
    }

    // Enclosure: every premise-satisfying grid sample's spec values must lie
    // inside the composed bounds of its child domain.
    let spec_vals = |x: &[f32; 4]| {
        let y: Vec<f32> = x.iter().map(|&v| 0.5 * v.max(0.0) + 0.1).collect();
        [y[0], y[1] + y[2]]
    };
    for (dom_idx, premise_active) in [(0usize, true), (1usize, false)] {
        let bounds = out.results[dom_idx].output_bounds.flatten();
        let n = 4;
        for i0 in 0..=n {
            for i1 in 0..=n {
                for i2 in 0..=n {
                    let x0 = -1.0 + 2.0 * (i0 as f32) / (n as f32);
                    if premise_active != (x0 >= 0.0) {
                        continue;
                    }
                    let x = [
                        x0,
                        -1.0 + 2.0 * (i1 as f32) / (n as f32),
                        -1.0 + 2.0 * (i2 as f32) / (n as f32),
                        0.3,
                    ];
                    let vals = spec_vals(&x);
                    for (row, &v) in vals.iter().enumerate() {
                        let l = bounds.lower()[[row]];
                        let u = bounds.upper()[[row]];
                        assert!(
                            l - 1e-4 <= v && v <= u + 1e-4,
                            "domain {dom_idx} row {row}: value {v} at x={x:?} \
                             escapes composed bound [{l}, {u}]"
                        );
                    }
                }
            }
        }
    }
}

/// #metaroom-chain-wide differential soundness oracle (`NY_BAB_CHAIN_WIDE`).
///
/// The chain-wide lane routes PURE-CHAIN conv suffixes (`segments = [Chain(..)]`,
/// metaroom's 6cnn shape class) onto the wide batched GPU β lane, whose bound
/// REPLACES the historical dense node-by-node batched backward (the
/// `try_gpu_beta_batched_resnet_opt` early return in
/// `propagate_crown_batched_backward_core_specs`). This oracle drives BOTH lanes
/// on the SAME domains — identical bounds caches (built by the production
/// `compute_constrained_forward_bounds`), identical constrained inputs, identical
/// β states and α states — and asserts, per (fixture x domain x spec-row):
///
/// (a) SOUNDNESS (hard assert): the wide lane's bound must ENCLOSE the true
///     network spec values at every premise-satisfying sample of the domain
///     (points are evaluated through the production forward on a degenerate box,
///     so the reference is the network itself, not a re-implementation).
/// (b) TIGHTNESS DIAGNOSTIC: compare the wide and dense bounds row-by-row. The
///     wide replacement is intentionally opt-in because it is not universally
///     tighter; the default-off contract is covered at the gate definition.
///
/// Coverage: chain lengths 2/3/4 convs (1/2/3 ReLUs), a stride-2 conv, biased
/// and bias-free convs, domains without β (root), with β (single and double
/// `with_constraint` splits — the production split machinery, so β entries
/// correspond exactly to the premises), and with a populated per-domain α state.
///
/// The wide leg is invoked with `graft_pure_chain = true`, which feeds the SAME
/// `allow_pure_chain` boolean that `NY_BAB_CHAIN_WIDE=1` sets (`allow_pure_chain =
/// bab_chain_wide_enabled() || graft_pure_chain`) — so the routing under test is
/// exactly the gate's, without mutating process-global env (racy under the
/// parallel test harness). The injected authority adapter makes this contract
/// deterministic and hardware-independent.
///
/// Measured 2026-07-17 (debug, wgpu): soundness held everywhere while the wide
/// bound was looser on 17/26 rows. A never-looser assertion is therefore not a
/// valid property of this algorithm.
fn chain_wide_replacement_oracle_body() {
    use crate::beta_crown::branching::GraphNeuronConstraint;
    use crate::beta_crown::state::AlphaNeuronState;
    use crate::layers::Conv2dLayer;
    use ndarray::{Array, IxDyn};
    use std::collections::HashMap;

    // Env guards: these would silently change ONE leg's routing/caches and void
    // the differential (the oracle must compare the historical dense path against
    // the chain-wide bound on symmetric inputs).
    for (var, why) in [
        ("NY_BAB_CHAIN_WIDE", "the dense leg would route wide too"),
        (
            "NY_RESNET_BETA_GPU",
            "the wide lane could be force-disabled",
        ),
        (
            "NY_INTERM_REFINE",
            "the dense leg would consume refined caches",
        ),
        (
            "NY_BAB_RESNET_BATCHED",
            "forces one wide sub-path for both values",
        ),
    ] {
        assert!(
            std::env::var(var).is_err(),
            "live chain-wide oracle requires {var} to be unset ({why})"
        );
    }
    let device = HermeticSoundGpuCrownEngine::default();
    let engine: &dyn ny_core::GemmEngine = &device;
    assert!(
        engine
            .as_gpu_crown_backward()
            .is_some_and(|g| g.provides_sound_gpu_crown()),
        "chain-wide oracle requires its injected sound CROWN backend"
    );

    // Deterministic LCG (the ny-gpu differential-oracle pattern).
    let mut state: u64 = 0xC4A1_57AC_71F3;
    let mut rng = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    };

    /// One conv stage of a pure chain: (out_channels, kernel, stride, pad, bias?).
    struct Stage(usize, usize, usize, usize, bool);
    struct Fixture {
        name: &'static str,
        in_shape: [usize; 3],
        stages: Vec<Stage>,
        /// Number of `with_constraint` splits to stack on the deepest child.
        extra_splits: usize,
    }
    let fixtures = [
        Fixture {
            name: "chain2-1x1",
            in_shape: [1, 2, 2],
            stages: vec![Stage(1, 1, 1, 0, false), Stage(1, 1, 1, 0, true)],
            extra_splits: 0,
        },
        Fixture {
            name: "chain3-stride2",
            in_shape: [1, 4, 4],
            stages: vec![
                Stage(2, 3, 1, 1, true),
                Stage(2, 2, 2, 0, false),
                Stage(1, 1, 1, 0, true),
            ],
            extra_splits: 0,
        },
        Fixture {
            name: "chain4-deep",
            in_shape: [2, 3, 3],
            stages: vec![
                Stage(2, 3, 1, 1, false),
                Stage(2, 3, 1, 1, true),
                Stage(2, 3, 1, 1, false),
                Stage(2, 3, 1, 1, true),
            ],
            extra_splits: 1,
        },
    ];

    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let mut looser: Vec<String> = Vec::new();
    let mut compared_rows = 0usize;

    for fx in &fixtures {
        // --- Build the pure conv chain: conv -> relu -> conv -> ... -> conv ---
        let mut graph = GraphNetwork::new();
        let (mut c, mut h, mut w) = (fx.in_shape[0], fx.in_shape[1], fx.in_shape[2]);
        let mut relu_names: Vec<String> = Vec::new();
        let mut prev: Option<String> = None;
        let n_stages = fx.stages.len();
        for (si, Stage(oc, k, s, p, bias)) in fx.stages.iter().enumerate() {
            let kernel = Array::from_shape_vec(
                IxDyn(&[*oc, c, *k, *k]),
                (0..oc * c * k * k).map(|_| rng() * 0.35).collect(),
            )
            .expect("kernel");
            let b = bias.then(|| arr1(&(0..*oc).map(|_| rng() * 0.1).collect::<Vec<f32>>()));
            let conv = Conv2dLayer::new(kernel, b, (*s, *s), (*p, *p)).expect("conv");
            let cname = format!("conv{si}");
            match &prev {
                None => graph.add_node(GraphNode::from_input(&cname, Layer::Conv2d(conv))),
                Some(p) => {
                    graph.add_node(GraphNode::new(&cname, Layer::Conv2d(conv), vec![p.clone()]))
                }
            }
            c = *oc;
            h = (h + 2 * p - k) / s + 1;
            w = (w + 2 * p - k) / s + 1;
            prev = Some(cname.clone());
            if si + 1 < n_stages {
                let rname = format!("relu{si}");
                graph.add_node(GraphNode::new(
                    &rname,
                    Layer::ReLU(ReLULayer),
                    vec![cname.clone()],
                ));
                relu_names.push(rname.clone());
                prev = Some(rname);
            }
        }
        let output_name = prev.expect("output node");
        graph.set_output(&output_name);
        assert!(graph.has_conv_layers());
        let output_dim = c * h * w;

        let in_dim = fx.in_shape.iter().product::<usize>();
        let input = BoundedTensor::new(
            Array::from_shape_vec(IxDyn(&fx.in_shape), vec![-1.0f32; in_dim]).unwrap(),
            Array::from_shape_vec(IxDyn(&fx.in_shape), vec![1.0f32; in_dim]).unwrap(),
        )
        .expect("input box");

        // --- Domains: root (no β), split children (β), a two-split child, α child ---
        let node_bounds = graph.collect_node_bounds(&input).expect("node bounds");
        let root =
            GraphBabDomain::root(node_bounds, -100.0, 100.0, &input, false).expect("root domain");

        // First unstable neuron of a ReLU node, per the root pre-activation cache.
        let unstable_neuron = |d: &GraphBabDomain, rname: &str| -> Option<usize> {
            let pre = graph.nodes.get(rname)?.inputs.first()?.clone();
            let bt = d.node_bounds.get(&pre)?.flatten();
            (0..bt.len()).find(|&j| bt.lower()[[j]] < 0.0 && bt.upper()[[j]] > 0.0)
        };
        let split_relu = relu_names.last().expect("chain has a ReLU").clone();
        let j0 = unstable_neuron(&root, &split_relu)
            .expect("fixture bug: no unstable neuron on the split ReLU");
        let mk_child = |parent: &GraphBabDomain, rname: &str, j: usize, is_active: bool| {
            parent
                .with_constraint(
                    &graph,
                    GraphNeuronConstraint {
                        node_name: rname.to_string(),
                        neuron_idx: j,
                        is_active,
                        score: 0.0,
                    },
                    false,
                )
                .expect("with_constraint")
                .expect("feasible child")
        };
        let active = mk_child(&root, &split_relu, j0, true);
        let inactive = mk_child(&root, &split_relu, j0, false);
        assert!(
            !active.beta_state.is_empty() && !inactive.beta_state.is_empty(),
            "{}: split children must carry β entries",
            fx.name
        );
        // Premise list per domain, for the sampling leg: (relu, neuron, is_active).
        type Premise = (String, usize, bool);
        let mut doms: Vec<(GraphBabDomain, Vec<Premise>)> = vec![
            (root.clone(), vec![]),
            (active.clone(), vec![(split_relu.clone(), j0, true)]),
            (inactive.clone(), vec![(split_relu.clone(), j0, false)]),
        ];
        if fx.extra_splits > 0 {
            // Stack a second production split on the FIRST ReLU (deeper β state).
            let first_relu = relu_names.first().expect("relu").clone();
            let j1 = unstable_neuron(&active, &first_relu)
                .expect("fixture bug: no unstable neuron on the first ReLU");
            let deep = mk_child(&active, &first_relu, j1, false);
            doms.push((
                deep,
                vec![
                    (split_relu.clone(), j0, true),
                    (first_relu.clone(), j1, false),
                ],
            ));
        }
        // α-state child: the active child with per-neuron α ∈ [0,1] on every ReLU
        // (both lanes consume the SAME GraphDomainAlphaState — the wide lane via
        // build_alpha_bridge, the dense lane via build_alpha_array).
        let mut alpha_child = active.clone();
        for rname in &relu_names {
            let pre = graph
                .nodes
                .get(rname)
                .and_then(|n| n.inputs.first())
                .expect("relu input")
                .clone();
            let bt = alpha_child
                .node_bounds
                .get(&pre)
                .expect("pre bounds")
                .flatten();
            let mk = || {
                (0..bt.len())
                    .filter(|&j| bt.lower()[[j]] < 0.0 && bt.upper()[[j]] > 0.0)
                    .take(4)
                    .map(|j| (j, AlphaNeuronState::new(0.3)))
                    .collect::<rustc_hash::FxHashMap<_, _>>()
            };
            alpha_child.alpha_state.neurons.insert(rname.clone(), mk());
            alpha_child
                .alpha_state
                .upper_neurons
                .insert(rname.clone(), mk());
        }
        doms.push((alpha_child, vec![(split_relu.clone(), j0, true)]));

        // --- Shared inputs for BOTH legs: the production constrained forward ---
        let n_domains = doms.len();
        let mut caches: Vec<HashMap<String, std::sync::Arc<BoundedTensor>>> =
            Vec::with_capacity(n_domains);
        let mut cinputs: Vec<BoundedTensor> = Vec::with_capacity(n_domains);
        for (d, _) in &doms {
            let (cache, cin) = verifier
                .compute_constrained_forward_bounds(
                    &graph,
                    &d.input_bounds,
                    &d.history,
                    Some(&d.node_bounds),
                    None,
                )
                .expect("constrained forward");
            caches.push(cache);
            cinputs.push(cin);
        }
        // #lsnc-shared-fwd: the batched cores now borrow per-domain caches.
        let caches_ref: Vec<&HashMap<String, std::sync::Arc<BoundedTensor>>> =
            caches.iter().collect();
        let histories: Vec<&crate::beta_crown::branching::GraphSplitHistory> =
            doms.iter().map(|(d, _)| &d.history).collect();
        let beta_refs: Vec<Option<&crate::beta_crown::state::GraphBetaState>> =
            doms.iter().map(|(d, _)| Some(&d.beta_state)).collect();
        let alpha_refs: Vec<Option<&crate::beta_crown::state::GraphDomainAlphaState>> =
            doms.iter().map(|(d, _)| Some(&d.alpha_state)).collect();

        // Two random spec rows over the flattened output.
        let num_specs = 2usize;
        let seed_rows: Vec<f32> = (0..num_specs * output_dim).map(|_| rng()).collect();
        let spec = Array2::from_shape_vec((num_specs, output_dim), seed_rows.clone())
            .expect("spec matrix");

        // --- Dense leg: the historical node-by-node batched backward (the gate-OFF
        // route: allow_pure_chain=false refuses the chain, the core falls through). ---
        let plan = graph.dispatch_plan().expect("dispatch plan");
        let dense = verifier
            .propagate_crown_batched_backward_core_specs(
                &graph,
                n_domains,
                plan,
                &caches_ref,
                &cinputs,
                &histories,
                &beta_refs,
                &alpha_refs,
                &spec,
                engine,
                super::BatchedBackwardMode::Standard,
                None,
                None,
                false,
            )
            .expect("dense node-by-node backward");
        assert_eq!(dense.results.len(), n_domains);

        // --- Wide leg: the chain-wide routing (same allow_pure_chain boolean the
        // NY_BAB_CHAIN_WIDE=1 gate feeds), same caches/inputs/β/α. ---
        let wide = verifier
            .try_gpu_beta_batched_resnet_opt(
                &graph,
                &output_name,
                output_dim,
                &seed_rows,
                num_specs,
                n_domains,
                &caches_ref,
                &cinputs,
                &beta_refs,
                &alpha_refs,
                engine,
                "chain-wide-oracle",
                None,
                true, // the chain-wide routing under test
            )
            .unwrap_or_else(|| {
                // The decline tally turns "it refused" into "it refused HERE".
                // This oracle previously went vacuous on
                // `EntryNoSoundBackend` — an entry-level backend admission, not
                // the chain-segment extraction everyone assumed — and the bare
                // message cost a diagnosis to recover.
                panic!(
                    "{}: chain-wide lane refused a pure conv chain — oracle vacuous \
                     (extraction or GPU path regressed). declines: {:?}",
                    fx.name,
                    ny_core::wide_lane_telemetry::wide_lane_decline_tally()
                        .into_iter()
                        .filter(|(_, n)| *n > 0)
                        .collect::<Vec<_>>()
                )
            })
            .0;
        assert_eq!(wide.len(), n_domains);

        // --- (a) SOUNDNESS: premise-filtered sampling through the production
        // forward (degenerate box ⇒ exact point evaluation). ---
        for (di, (_, premises)) in doms.iter().enumerate() {
            let cin = cinputs[di].flatten();
            let dim = cin.len();
            let wf = wide[di].flatten();
            let df = dense.results[di].output_bounds.flatten();
            let mut kept = 0usize;
            for _ in 0..160 {
                let x: Vec<f32> = (0..dim)
                    .map(|k| {
                        let (l, u) = (cin.lower()[[k]], cin.upper()[[k]]);
                        l + (rng() * 0.5 + 0.5) * (u - l)
                    })
                    .collect();
                let point_arr = Array::from_shape_vec(IxDyn(&fx.in_shape), x).unwrap();
                let point = BoundedTensor::new(point_arr.clone(), point_arr).expect("point box");
                let vals = graph.collect_node_bounds(&point).expect("point forward");
                let sat = premises.iter().all(|(rname, j, is_active)| {
                    let pre = graph.nodes.get(rname).unwrap().inputs.first().unwrap();
                    let v = vals.get(pre).unwrap().flatten().lower()[[*j]];
                    if *is_active {
                        v >= 0.0
                    } else {
                        v <= 0.0
                    }
                });
                if !sat {
                    continue;
                }
                kept += 1;
                let y = vals.get(&output_name).unwrap().flatten();
                for row in 0..num_specs {
                    let v: f32 = (0..output_dim)
                        .map(|k| spec[[row, k]] * y.lower()[[k]])
                        .sum();
                    let tol = 1e-3 * (1.0 + v.abs());
                    assert!(
                        wf.lower()[[row]] - tol <= v && v <= wf.upper()[[row]] + tol,
                        "{} dom {di} row {row}: TRUE VALUE {v} escapes the WIDE bound \
                         [{}, {}] — the chain-wide lane is UNSOUND",
                        fx.name,
                        wf.lower()[[row]],
                        wf.upper()[[row]],
                    );
                    assert!(
                        df.lower()[[row]] - tol <= v && v <= df.upper()[[row]] + tol,
                        "{} dom {di} row {row}: true value {v} escapes the DENSE bound \
                         [{}, {}] (baseline broken)",
                        fx.name,
                        df.lower()[[row]],
                        df.upper()[[row]],
                    );
                }
            }
            assert!(
                kept >= 8,
                "{} dom {di}: only {kept} premise-satisfying samples — oracle vacuous",
                fx.name
            );

            // --- (b) NEVER-LOOSER (the activation criterion) + evidence line. ---
            for row in 0..num_specs {
                compared_rows += 1;
                let (dl, du) = (df.lower()[[row]], df.upper()[[row]]);
                let (wl, wu) = (wf.lower()[[row]], wf.upper()[[row]]);
                eprintln!(
                    "[chain-wide-oracle] {} dom {di} row {row}: dense=[{dl:.6}, {du:.6}] \
                     wide=[{wl:.6}, {wu:.6}]",
                    fx.name
                );
                let tol = 1e-3 * (1.0 + dl.abs().max(du.abs()));
                if wl < dl - tol {
                    looser.push(format!(
                        "{} dom {di} row {row} LOWER: wide {wl} < dense {dl} (gap {})",
                        fx.name,
                        dl - wl
                    ));
                }
                if wu > du + tol {
                    looser.push(format!(
                        "{} dom {di} row {row} UPPER: wide {wu} > dense {du} (gap {})",
                        fx.name,
                        wu - du
                    ));
                }
            }
        }
    }

    assert!(compared_rows > 0, "oracle compared no rows");
    if !looser.is_empty() {
        eprintln!(
            "[chain-wide-oracle] ACTIVATION GATE RED — the wide replacement bound is LOOSER \
             than the dense node-by-node bound on {}/{} rows (NY_BAB_CHAIN_WIDE must stay \
             dark):\n{}",
            looser.len(),
            compared_rows,
            looser.join("\n"),
        );
    }
}

/// The wide chain lane must be a sound enclosure on every pure-conv-chain
/// fixture/domain/row. Extraction and sampling coverage must be non-vacuous.
#[test]
fn chain_wide_replacement_oracle_pure_conv_chains_is_sound() {
    chain_wide_replacement_oracle_body();
}

/// #lsnc-skip-node-bounds (S3b) parity: skipping the DISCARDED per-domain
/// node-bounds clone in `concretize_batched_results_specs` must leave every
/// verdict-bearing output BIT-IDENTICAL — output bounds, `input_linear`
/// presence, coefficients, biases, AND certified coefficient-error matrices —
/// with the ONLY observable difference being that `node_bounds` comes back
/// empty on the skip leg (the input-split caller drops it unread;
/// `input_split/shared_specs.rs:307-312` keeps only `output_bounds` +
/// `input_linear`). Both legs run the production
/// `batched_forward_then_backward_specs` via
/// `propagate_crown_batched_with_context_specs`, A/B'd through the cached-gate
/// test override (`force_skip_node_bounds`), not process env.
#[test]
fn test_input_split_skip_node_bounds_bit_identical_lsnc_s3b() {
    use super::spec_adapters::force_skip_node_bounds;
    use crate::batched_domain::{BatchedDomainOptions, BatchedDomainsBuilder};
    use crate::beta_crown::branching::GraphSplitHistory;
    use std::collections::HashMap;

    // Serialize with the other force_* gate tests: the forced-gate windows are
    // process-global and would otherwise race concurrently running parity legs.
    let _gate_guard = super::spec_adapters::SPEC_GATE_TEST_LOCK
        .lock()
        .unwrap_or_else(|p| p.into_inner());

    // 2 -> 3 (ReLU) -> 2 MLP with unstable hidden neurons over the sub-boxes,
    // mixed-sign weights so lower/upper planes genuinely differ.
    let linear1 = LinearLayer::new(
        arr2(&[[1.0, -1.0], [0.5, 0.5], [-1.0, 1.0]]),
        Some(arr1(&[0.1, -0.2, 0.0])),
    )
    .expect("valid linear1");
    let linear2 = LinearLayer::new(
        arr2(&[[1.0, 1.0, -1.0], [0.5, -1.0, 1.0]]),
        Some(arr1(&[0.0, 0.3])),
    )
    .expect("valid linear2");
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

    // Four disjoint input sub-boxes (the input-split children shape: no shared
    // warmup map => the per-domain `collect_intermediate_bounds` arm, exactly
    // the lsnc regime where root_node_bounds is None).
    let sub_boxes = [
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[0.0, 0.0]).into_dyn()).unwrap(),
        BoundedTensor::new(arr1(&[0.0, -1.0]).into_dyn(), arr1(&[1.0, 0.0]).into_dyn()).unwrap(),
        BoundedTensor::new(arr1(&[-1.0, 0.0]).into_dyn(), arr1(&[0.0, 1.0]).into_dyn()).unwrap(),
        BoundedTensor::new(arr1(&[0.0, 0.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap(),
    ];
    let n = sub_boxes.len();

    let mut builder =
        BatchedDomainsBuilder::new_with_options(Vec::new(), BatchedDomainOptions::default());
    let empty_layer_bounds: HashMap<String, (ndarray::ArrayD<f32>, ndarray::ArrayD<f32>)> =
        HashMap::new();
    for b in &sub_boxes {
        builder.add_domain(
            &empty_layer_bounds,
            b.lower().clone(),
            b.upper().clone(),
            0.0,
            0.0,
            0,
            Vec::new(),
        );
    }
    let batched = builder.build().expect("batched domains");
    let empty_history = GraphSplitHistory::new();

    // Multi-row spec: 3 rows over 2 outputs, mixed signs and an exact zero.
    let spec_matrix = Array2::from_shape_vec((3, 2), vec![1.0, -1.0, 0.0, 1.0, -0.5, 0.25])
        .expect("valid spec matrix");
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());

    let run_leg = |force: bool| {
        force_skip_node_bounds(Some(force));
        let ctx = BatchedBackwardContext {
            batched: &batched,
            histories: vec![&empty_history; n],
            beta_states: vec![None; n],
            base_bounds: vec![None; n],
            delta_seeds: vec![None; n],
            alpha_states: vec![None; n],
            cached_la: vec![None; n],
            mul_binary_alphas: None,
        };
        let out = verifier.propagate_crown_batched_with_context_specs(
            &graph,
            &ctx,
            &spec_matrix,
            &NaiveCpuGemmEngine,
        );
        force_skip_node_bounds(None);
        out.expect("dense-spec batched backward should succeed")
    };

    let reference = run_leg(false);
    let skipped = run_leg(true);

    assert_eq!(reference.len(), n);
    assert_eq!(skipped.len(), n);
    for i in 0..n {
        // Raw f32 BIT identity on the verdict-bearing output bounds.
        for (a, b) in reference[i]
            .output_bounds
            .lower()
            .iter()
            .zip(skipped[i].output_bounds.lower().iter())
        {
            assert_eq!(a.to_bits(), b.to_bits(), "domain {i}: lower bound bits");
        }
        for (a, b) in reference[i]
            .output_bounds
            .upper()
            .iter()
            .zip(skipped[i].output_bounds.upper().iter())
        {
            assert_eq!(a.to_bits(), b.to_bits(), "domain {i}: upper bound bits");
        }

        // input_linear: presence + exact coefficient/bias/certified-error bits.
        match (&reference[i].input_linear, &skipped[i].input_linear) {
            (None, None) => {}
            (Some(r), Some(s)) => {
                assert!(
                    r.lower_a()
                        .iter()
                        .zip(s.lower_a().iter())
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "domain {i}: lower_a bits"
                );
                assert!(
                    r.upper_a()
                        .iter()
                        .zip(s.upper_a().iter())
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "domain {i}: upper_a bits"
                );
                assert!(
                    r.lower_b()
                        .iter()
                        .zip(s.lower_b().iter())
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "domain {i}: lower_b bits"
                );
                assert!(
                    r.upper_b()
                        .iter()
                        .zip(s.upper_b().iter())
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "domain {i}: upper_b bits"
                );
                // Certified-error matrices: PRESENCE (None = exact marker,
                // I-D5) and per-entry bits must match.
                match (r.lower_a_err(), s.lower_a_err()) {
                    (None, None) => {}
                    (Some(re), Some(se)) => assert!(
                        re.iter()
                            .zip(se.iter())
                            .all(|(a, b)| a.to_bits() == b.to_bits()),
                        "domain {i}: lower_a_err bits"
                    ),
                    _ => panic!("domain {i}: lower_a_err presence diverged"),
                }
                match (r.upper_a_err(), s.upper_a_err()) {
                    (None, None) => {}
                    (Some(re), Some(se)) => assert!(
                        re.iter()
                            .zip(se.iter())
                            .all(|(a, b)| a.to_bits() == b.to_bits()),
                        "domain {i}: upper_a_err bits"
                    ),
                    _ => panic!("domain {i}: upper_a_err presence diverged"),
                }
            }
            _ => panic!("domain {i}: input_linear presence diverged between legs"),
        }

        // The ONLY allowed difference: reference populates node_bounds, the
        // skip leg leaves it empty (the input-split caller never reads it).
        assert!(
            !reference[i].node_bounds.is_empty(),
            "domain {i}: reference leg must populate node_bounds"
        );
        assert!(
            skipped[i].node_bounds.is_empty(),
            "domain {i}: skip leg must leave node_bounds empty"
        );
    }

    // Sampled concrete-evaluation containment (parity to a broken reference is
    // worthless): network outputs on a grid inside each box stay within the
    // reported spec-row bounds.
    for (i, b) in sub_boxes.iter().enumerate() {
        let flat_res = skipped[i].output_bounds.flatten();
        for gx in 0..=4 {
            for gy in 0..=4 {
                let x0 = b.lower()[[0]] + (b.upper()[[0]] - b.lower()[[0]]) * gx as f32 / 4.0;
                let x1 = b.lower()[[1]] + (b.upper()[[1]] - b.lower()[[1]]) * gy as f32 / 4.0;
                // Forward: linear1 -> relu -> linear2 (weights above).
                let h = [
                    (x0 - x1 + 0.1).max(0.0),
                    (0.5 * x0 + 0.5 * x1 - 0.2).max(0.0),
                    (-x0 + x1).max(0.0),
                ];
                let y = [h[0] + h[1] - h[2], 0.5 * h[0] - h[1] + h[2] + 0.3];
                for (row, spec) in spec_matrix.rows().into_iter().enumerate() {
                    let v: f32 = spec.iter().zip(y.iter()).map(|(c, yv)| c * yv).sum();
                    let lo = flat_res.lower()[[row]];
                    let hi = flat_res.upper()[[row]];
                    assert!(
                        v >= lo - SAMPLE_TOLERANCE_NY && v <= hi + SAMPLE_TOLERANCE_NY,
                        "domain {i} row {row}: sample {v} outside [{lo}, {hi}]"
                    );
                }
            }
        }
    }
}

/// #lsnc-batched-interm SEAM PARITY (design-doc slice S2): driving the
/// production dense-spec pipeline (`propagate_crown_batched_with_context_specs`)
/// in the no-warmup input-split regime (`base_bounds = None`, empty histories —
/// the lsnc plain-CROWN configuration) with the batched intermediate-bounds
/// forward FORCED ON vs FORCED OFF must be BIT-IDENTICAL on every output the
/// result carries: certified output bounds, the per-domain node-bounds maps,
/// AND the per-domain input `LinearBounds` (coefficients, biases, and
/// certified-error matrices — presence and bits). The fixture contains a
/// MulBinary so the graph is in the batched collector's supported class
/// (plain-IBP forward, like the real lsnc net), and the sub-boxes flip the
/// ReLU stability class across domains.
///
/// Also asserts, via the `BATCHED_INTERM_ENGAGED` probe, that the batched
/// collector actually RAN in the ON leg and did NOT run in the OFF leg
/// (checklist Part 3 A: the compared legs are genuinely different paths).
#[test]
fn test_input_split_batched_interm_seam_bit_identical() {
    use std::collections::HashMap;
    use std::sync::atomic::Ordering;

    use super::spec_adapters::{force_batched_interm, BATCHED_INTERM_ENGAGED};
    use crate::batched_domain::{BatchedDomainOptions, BatchedDomainsBuilder};
    use crate::beta_crown::branching::GraphSplitHistory;
    use crate::layers::MulBinaryLayer;

    // Serialize with the other force_* gate tests: a concurrent
    // `force_skip_node_bounds(Some(true))` window would empty this test's
    // node_bounds maps, and concurrent pipeline runs would corrupt the
    // engagement-probe deltas below.
    let _gate_guard = super::spec_adapters::SPEC_GATE_TEST_LOCK
        .lock()
        .unwrap_or_else(|p| p.into_inner());

    // 2 -> 3 (ReLU) -> 3 MLP + MulBinary(relu1, linear2) output. The MulBinary
    // keeps the graph OUT of the per-node CROWN-IBP class, so the reference
    // forward is the plain per-domain IBP collect — the arm S2 batches.
    let linear1 = LinearLayer::new(
        arr2(&[[1.0, -1.0], [0.5, 0.9], [-1.0, 0.4]]),
        Some(arr1(&[0.1, -0.2, 0.0])),
    )
    .expect("valid linear1");
    let linear2 = LinearLayer::new(
        arr2(&[[1.0, -0.5, 0.8], [-0.6, 0.7, -0.3], [0.4, 0.4, -0.9]]),
        Some(arr1(&[0.0, 0.1, -0.05])),
    )
    .expect("valid linear2");
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
        "mulb",
        Layer::MulBinary(MulBinaryLayer),
        vec!["relu1".to_string(), "linear2".to_string()],
    ));
    graph.set_output("mulb");
    assert!(
        graph.batched_interm_forward_supported(),
        "fixture must be in the batched-interm supported class"
    );

    // Sub-boxes that flip relu1 stability classes across domains (plus a
    // degenerate and an asymmetric box).
    let sub_boxes = [
        ([-1.0, -1.0], [0.0, 0.0]),
        ([0.0, -1.0], [1.0, 0.0]),
        ([-1.0, 0.0], [0.0, 1.0]),
        ([0.25, 0.25], [1.0, 1.0]),
        ([-0.5, -0.5], [0.5, 0.5]),
        ([0.125, -0.75], [0.125, -0.75]),
    ];
    let n = sub_boxes.len();

    let mut builder =
        BatchedDomainsBuilder::new_with_options(Vec::new(), BatchedDomainOptions::default());
    let empty_layer_bounds: HashMap<String, (ndarray::ArrayD<f32>, ndarray::ArrayD<f32>)> =
        HashMap::new();
    for (lo, hi) in &sub_boxes {
        let b = BoundedTensor::new(arr1(lo).into_dyn(), arr1(hi).into_dyn()).unwrap();
        builder.add_domain(
            &empty_layer_bounds,
            b.lower().clone(),
            b.upper().clone(),
            0.0,
            0.0,
            0,
            Vec::new(),
        );
    }
    let batched = builder.build().expect("batched domains");
    let empty_history = GraphSplitHistory::new();
    let spec_matrix = Array2::from_shape_vec((2, 3), vec![1.0, 0.0, -1.0, 0.0, 1.0, 0.5])
        .expect("valid spec matrix");
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());

    let run = || {
        // The lsnc no-warmup regime: every domain has base_bounds = None and
        // an empty split history.
        let ctx = BatchedBackwardContext {
            batched: &batched,
            histories: vec![&empty_history; n],
            beta_states: vec![None; n],
            base_bounds: vec![None; n],
            delta_seeds: vec![None; n],
            alpha_states: vec![None; n],
            cached_la: vec![None; n],
            mul_binary_alphas: None,
        };
        verifier
            .propagate_crown_batched_with_context_specs(
                &graph,
                &ctx,
                &spec_matrix,
                &NaiveCpuGemmEngine,
            )
            .expect("dense-spec propagation should succeed")
    };

    let engaged_before_on = BATCHED_INTERM_ENGAGED.load(Ordering::Relaxed);
    force_batched_interm(Some(true));
    let on = run();
    let engaged_after_on = BATCHED_INTERM_ENGAGED.load(Ordering::Relaxed);
    force_batched_interm(Some(false));
    let engaged_before_off = BATCHED_INTERM_ENGAGED.load(Ordering::Relaxed);
    let off = run();
    let engaged_after_off = BATCHED_INTERM_ENGAGED.load(Ordering::Relaxed);
    force_batched_interm(None); // restore env default

    assert!(
        engaged_after_on > engaged_before_on,
        "ON leg must actually engage the batched-interm collector"
    );
    assert_eq!(
        engaged_after_off, engaged_before_off,
        "OFF leg must run the per-domain reference, not the batched collector"
    );

    assert_eq!(on.len(), n);
    assert_eq!(off.len(), n);
    for i in 0..n {
        // Certified output bounds: raw f32 bits.
        for (a, b) in on[i]
            .output_bounds
            .lower()
            .iter()
            .zip(off[i].output_bounds.lower().iter())
        {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "domain {i}: output LOWER bits differ"
            );
        }
        for (a, b) in on[i]
            .output_bounds
            .upper()
            .iter()
            .zip(off[i].output_bounds.upper().iter())
        {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "domain {i}: output UPPER bits differ"
            );
        }

        // Per-domain node-bounds maps: identical key sets and bits.
        assert_eq!(
            on[i].node_bounds.len(),
            off[i].node_bounds.len(),
            "domain {i}: node_bounds map sizes differ"
        );
        for (name, on_b) in &on[i].node_bounds {
            let off_b = off[i]
                .node_bounds
                .get(name)
                .unwrap_or_else(|| panic!("domain {i}: node '{name}' missing in OFF leg"));
            for (a, b) in on_b.lower().iter().zip(off_b.lower().iter()) {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "domain {i} node '{name}': lower bits"
                );
            }
            for (a, b) in on_b.upper().iter().zip(off_b.upper().iter()) {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "domain {i} node '{name}': upper bits"
                );
            }
        }

        // Input LinearBounds: presence, coefficients, biases, AND certified
        // error matrices (presence + bits — checklist I-D5).
        match (&on[i].input_linear, &off[i].input_linear) {
            (None, None) => {}
            (Some(on_lb), Some(off_lb)) => {
                assert_eq!(
                    on_lb.lower_a(),
                    off_lb.lower_a(),
                    "domain {i}: lower_a differs"
                );
                assert_eq!(
                    on_lb.upper_a(),
                    off_lb.upper_a(),
                    "domain {i}: upper_a differs"
                );
                assert_eq!(
                    on_lb.lower_b(),
                    off_lb.lower_b(),
                    "domain {i}: lower_b differs"
                );
                assert_eq!(
                    on_lb.upper_b(),
                    off_lb.upper_b(),
                    "domain {i}: upper_b differs"
                );
                assert_eq!(
                    on_lb.lower_a_err(),
                    off_lb.lower_a_err(),
                    "domain {i}: lower_a_err differs (certified error must match bit-for-bit)"
                );
                assert_eq!(
                    on_lb.upper_a_err(),
                    off_lb.upper_a_err(),
                    "domain {i}: upper_a_err differs"
                );
            }
            (a, b) => panic!(
                "domain {i}: input_linear presence differs (on={}, off={})",
                a.is_some(),
                b.is_some()
            ),
        }
    }
}

/// Shared fixture for the #lsnc-batched-bwd (S3) parity tests: an
/// lsnc-shaped DAG — two Linear branches off the network input (a genuine
/// pending-merge at NETWORK_INPUT), ReLUs whose stability classes flip across
/// sub-boxes, and the generic-dispatch ops of the lsnc lane (MulBinary, Sub,
/// Concat, ReduceSum) whose Binary/Nary bias carriers hammer the
/// accumulate/merge path the SoA lane replaces.
#[cfg(test)]
fn build_lsnc_shaped_batched_bwd_graph() -> GraphNetwork {
    use crate::layers::{ConcatLayer, MulBinaryLayer, ReduceSumLayer, SubLayer};

    let linear1 = LinearLayer::new(
        arr2(&[[1.0, -1.0], [0.5, 0.9], [-1.0, 0.4], [0.3, -0.7]]),
        Some(arr1(&[0.1, -0.2, 0.0, 0.05])),
    )
    .expect("valid linear1");
    let linear2 = LinearLayer::new(
        arr2(&[[0.8, 0.3], [-0.4, 0.6], [0.2, -0.9], [1.1, 0.05]]),
        Some(arr1(&[0.02, 0.0, -0.1, 0.07])),
    )
    .expect("valid linear2");
    let linear3 = LinearLayer::new(
        arr2(&[
            [1.0, -0.5, 0.8, 0.2, -0.6, 0.7, -0.3, 1.0],
            [0.4, 0.4, -0.9, -0.1, 0.3, -0.8, 0.6, 0.2],
            [-0.2, 0.9, 0.1, -0.7, 0.5, 0.0, -0.4, 0.35],
            [0.15, -0.25, 0.45, 0.65, -0.85, 0.05, 0.75, -0.55],
        ]),
        Some(arr1(&[0.0, 0.1, -0.05, 0.02])),
    )
    .expect("valid linear3");
    let linear_out = LinearLayer::new(
        arr2(&[
            [1.0, -1.0, 0.5, 0.25, -0.75],
            [0.3, 0.8, -0.6, 0.0, 0.45],
            [-0.5, 0.2, 0.9, -0.35, 0.1],
        ]),
        Some(arr1(&[0.0, 0.0, 0.01])),
    )
    .expect("valid linear_out");

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));
    graph.add_node(GraphNode::from_input("linear2", Layer::Linear(linear2)));
    graph.add_node(GraphNode::new(
        "mulb",
        Layer::MulBinary(MulBinaryLayer),
        vec!["relu1".to_string(), "linear2".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "sub",
        Layer::Sub(SubLayer),
        vec!["mulb".to_string(), "relu1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "concat",
        Layer::Concat(ConcatLayer::new(0)),
        vec!["sub".to_string(), "relu1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear3",
        Layer::Linear(linear3),
        vec!["concat".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "relu2",
        Layer::ReLU(ReLULayer),
        vec!["linear3".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "reduce",
        Layer::ReduceSum(ReduceSumLayer::new(vec![-1], true)),
        vec!["relu2".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "concat2",
        Layer::Concat(ConcatLayer::new(0)),
        vec!["reduce".to_string(), "sub".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear_out",
        Layer::Linear(linear_out),
        vec!["concat2".to_string()],
    ));
    graph.set_output("linear_out");
    graph
}

/// Sub-boxes for the batched-bwd fixture: an LCG tiling of [-1, 1]^2 with
/// widths that flip relu1/relu2 stability classes across domains, plus a
/// degenerate point box. `n` is large enough that the fast lane's coarse
/// rayon chunking (min 16 domains/task) actually splits.
#[cfg(test)]
fn batched_bwd_sub_boxes(n: usize) -> Vec<([f32; 2], [f32; 2])> {
    let mut state: u64 = 0xB5AD4ECEDA1CE2A9;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) as f32) / (u32::MAX as f32)
    };
    let mut boxes = Vec::with_capacity(n);
    for i in 0..n {
        if i == 0 {
            boxes.push(([0.125, -0.75], [0.125, -0.75])); // degenerate point box
            continue;
        }
        let cx = next() * 2.0 - 1.0;
        let cy = next() * 2.0 - 1.0;
        let wx = next() * 0.9;
        let wy = next() * 0.9;
        let lo = [(cx - wx).max(-1.0), (cy - wy).max(-1.0)];
        let hi = [(cx + wx).min(1.0), (cy + wy).min(1.0)];
        boxes.push((lo, hi));
    }
    boxes
}

/// #lsnc-batched-bwd (S3) full-pipeline parity: the SoA batched-tensor
/// backward must be BIT-IDENTICAL to the per-domain reference loop — raw f32
/// bits of the certified output bounds, the input `LinearBounds` coefficient
/// matrices and biases, AND the certified coefficient-error matrices
/// (presence + bits, I-D5) — across a 44-domain lsnc-shaped batch whose
/// contributions exercise the Linear SoA kernel, the shared ReLU body, the
/// generic-dispatch Binary/Nary bias carriers, and repeated NETWORK_INPUT
/// merges. Engagement probes assert which leg actually ran.
#[ntest::timeout(120000)]
#[test]
fn test_input_split_batched_bwd_full_pipeline_bit_identical_lsnc_s3() {
    use std::collections::HashMap;

    use super::batched_bwd::{engaged_on_thread, force_batched_bwd};
    use crate::batched_domain::{BatchedDomainOptions, BatchedDomainsBuilder};
    use crate::beta_crown::branching::GraphSplitHistory;

    // Serialize with the other force_* gate tests (the forced-gate windows
    // are process-global).
    let _gate_guard = super::spec_adapters::SPEC_GATE_TEST_LOCK
        .lock()
        .unwrap_or_else(|p| p.into_inner());

    let graph = build_lsnc_shaped_batched_bwd_graph();
    let sub_boxes = batched_bwd_sub_boxes(44);
    let n = sub_boxes.len();

    let mut builder =
        BatchedDomainsBuilder::new_with_options(Vec::new(), BatchedDomainOptions::default());
    let empty_layer_bounds: HashMap<String, (ndarray::ArrayD<f32>, ndarray::ArrayD<f32>)> =
        HashMap::new();
    for (lo, hi) in &sub_boxes {
        let b = BoundedTensor::new(arr1(lo).into_dyn(), arr1(hi).into_dyn()).unwrap();
        builder.add_domain(
            &empty_layer_bounds,
            b.lower().clone(),
            b.upper().clone(),
            0.0,
            0.0,
            0,
            Vec::new(),
        );
    }
    let batched = builder.build().expect("batched domains");
    let empty_history = GraphSplitHistory::new();
    // Multi-clause-shaped spec: 3 rows, mixed signs, exact zeros.
    let spec_matrix =
        Array2::from_shape_vec((3, 3), vec![1.0, 0.0, -1.0, 0.0, 1.0, 0.5, -0.25, 0.0, 1.0])
            .expect("valid spec matrix");
    let mut verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    // This parity test exercises the historical unscored SoA lane. A finite
    // deadline intentionally declines that lane because its whole-batch kernel
    // has no cooperative cancellation seam.
    verifier.config.alpha_config.deadline = None;

    let run = || {
        let ctx = BatchedBackwardContext {
            batched: &batched,
            histories: vec![&empty_history; n],
            beta_states: vec![None; n],
            base_bounds: vec![None; n],
            delta_seeds: vec![None; n],
            alpha_states: vec![None; n],
            cached_la: vec![None; n],
            mul_binary_alphas: None,
        };
        verifier
            .propagate_crown_batched_with_context_specs(
                &graph,
                &ctx,
                &spec_matrix,
                &NaiveCpuGemmEngine,
            )
            .expect("dense-spec propagation should succeed")
    };

    let engaged_before_on = engaged_on_thread();
    force_batched_bwd(Some(true));
    let on = run();
    let engaged_after_on = engaged_on_thread();
    force_batched_bwd(Some(false));
    let engaged_before_off = engaged_on_thread();
    let off = run();
    let engaged_after_off = engaged_on_thread();
    force_batched_bwd(None); // restore env default

    assert!(
        engaged_after_on > engaged_before_on,
        "ON leg must actually engage the SoA batched backward"
    );
    assert_eq!(
        engaged_after_off, engaged_before_off,
        "OFF leg must run the per-domain reference, not the SoA lane"
    );

    assert_eq!(on.len(), n);
    assert_eq!(off.len(), n);
    let mut input_linear_present = 0usize;
    for i in 0..n {
        for (a, b) in on[i]
            .output_bounds
            .lower()
            .iter()
            .zip(off[i].output_bounds.lower().iter())
        {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "domain {i}: output LOWER bits differ"
            );
        }
        for (a, b) in on[i]
            .output_bounds
            .upper()
            .iter()
            .zip(off[i].output_bounds.upper().iter())
        {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "domain {i}: output UPPER bits differ"
            );
        }
        match (&on[i].input_linear, &off[i].input_linear) {
            (None, None) => {}
            (Some(on_lb), Some(off_lb)) => {
                input_linear_present += 1;
                let bits2 =
                    |x: &Array2<f32>| -> Vec<u32> { x.iter().map(|v| v.to_bits()).collect() };
                assert_eq!(
                    bits2(on_lb.lower_a()),
                    bits2(off_lb.lower_a()),
                    "domain {i}: lower_a bits differ"
                );
                assert_eq!(
                    bits2(on_lb.upper_a()),
                    bits2(off_lb.upper_a()),
                    "domain {i}: upper_a bits differ"
                );
                let bits1: Vec<u32> = on_lb.lower_b().iter().map(|v| v.to_bits()).collect();
                let bits1_off: Vec<u32> = off_lb.lower_b().iter().map(|v| v.to_bits()).collect();
                assert_eq!(bits1, bits1_off, "domain {i}: lower_b bits differ");
                let ubits: Vec<u32> = on_lb.upper_b().iter().map(|v| v.to_bits()).collect();
                let ubits_off: Vec<u32> = off_lb.upper_b().iter().map(|v| v.to_bits()).collect();
                assert_eq!(ubits, ubits_off, "domain {i}: upper_b bits differ");
                // Certified error: presence AND bits (a bounds-only compare
                // passes while the certificate silently loosens — I-D5).
                assert_eq!(
                    on_lb.lower_a_err().is_some(),
                    off_lb.lower_a_err().is_some(),
                    "domain {i}: lower_a_err presence differs"
                );
                assert_eq!(
                    on_lb.upper_a_err().is_some(),
                    off_lb.upper_a_err().is_some(),
                    "domain {i}: upper_a_err presence differs"
                );
                if let (Some(a), Some(b)) = (on_lb.lower_a_err(), off_lb.lower_a_err()) {
                    assert_eq!(bits2(a), bits2(b), "domain {i}: lower_a_err bits differ");
                }
                if let (Some(a), Some(b)) = (on_lb.upper_a_err(), off_lb.upper_a_err()) {
                    assert_eq!(bits2(a), bits2(b), "domain {i}: upper_a_err bits differ");
                }
            }
            (a, b) => panic!(
                "domain {i}: input_linear presence differs (on={}, off={})",
                a.is_some(),
                b.is_some()
            ),
        }
    }
    assert!(
        input_linear_present > 0,
        "fixture must produce captured input_linear rows (otherwise the merge \
         path under test never ran)"
    );
}

/// #lsnc-batched-bwd decline leg: a domain carrying a β state is OUTSIDE the
/// fast lane's clean class (input split carries no β) — with the gate forced
/// ON the lane must DECLINE (engagement probe unchanged) and the reference
/// loop must produce bit-identical results to the gate-OFF run. Fail-closed:
/// strictly a performance transform.
#[ntest::timeout(120000)]
#[test]
fn test_input_split_batched_bwd_declines_on_beta_state_lsnc_s3() {
    use std::collections::HashMap;

    use super::batched_bwd::{engaged_on_thread, force_batched_bwd};
    use crate::batched_domain::{BatchedDomainOptions, BatchedDomainsBuilder};
    use crate::beta_crown::branching::GraphSplitHistory;
    use crate::beta_crown::state::{GraphBetaEntry, GraphBetaState};

    let _gate_guard = super::spec_adapters::SPEC_GATE_TEST_LOCK
        .lock()
        .unwrap_or_else(|p| p.into_inner());

    let graph = build_lsnc_shaped_batched_bwd_graph();
    let sub_boxes = batched_bwd_sub_boxes(6);
    let n = sub_boxes.len();

    let mut builder =
        BatchedDomainsBuilder::new_with_options(Vec::new(), BatchedDomainOptions::default());
    let empty_layer_bounds: HashMap<String, (ndarray::ArrayD<f32>, ndarray::ArrayD<f32>)> =
        HashMap::new();
    for (lo, hi) in &sub_boxes {
        let b = BoundedTensor::new(arr1(lo).into_dyn(), arr1(hi).into_dyn()).unwrap();
        builder.add_domain(
            &empty_layer_bounds,
            b.lower().clone(),
            b.upper().clone(),
            0.0,
            0.0,
            0,
            Vec::new(),
        );
    }
    let batched = builder.build().expect("batched domains");
    let empty_history = GraphSplitHistory::new();
    let spec_matrix =
        Array2::from_shape_vec((3, 3), vec![1.0, 0.0, -1.0, 0.0, 1.0, 0.5, -0.25, 0.0, 1.0])
            .expect("valid spec matrix");
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    // A NUMERICALLY ACTIVE β entry: if the lane failed to decline, it would
    // drop this contribution (the SoA lane carries no β arm) and the ON-leg
    // bits below would diverge from the reference — so the bit comparison is a
    // second, counter-free witness of the fail-closed contract.
    let beta = GraphBetaState {
        entries: vec![GraphBetaEntry {
            node_name: "relu1".to_string(),
            neuron_idx: 0,
            split_point: 0.0,
            value: 0.5,
            sign: 1.0,
            grad: 0.0,
            m: 0.0,
            v: 0.0,
            v_max: 0.0,
        }],
        ..GraphBetaState::empty()
    };

    let run = || {
        let mut beta_states: Vec<Option<&GraphBetaState>> = vec![None; n];
        beta_states[0] = Some(&beta); // outside the clean class → decline
        let ctx = BatchedBackwardContext {
            batched: &batched,
            histories: vec![&empty_history; n],
            beta_states,
            base_bounds: vec![None; n],
            delta_seeds: vec![None; n],
            alpha_states: vec![None; n],
            cached_la: vec![None; n],
            mul_binary_alphas: None,
        };
        verifier
            .propagate_crown_batched_with_context_specs(
                &graph,
                &ctx,
                &spec_matrix,
                &NaiveCpuGemmEngine,
            )
            .expect("dense-spec propagation should succeed")
    };

    let engaged_before = engaged_on_thread();
    force_batched_bwd(Some(true));
    let on = run();
    let engaged_after = engaged_on_thread();
    force_batched_bwd(Some(false));
    let off = run();
    force_batched_bwd(None);

    assert_eq!(
        engaged_after, engaged_before,
        "β-carrying batch must DECLINE the SoA lane (fail-closed), not engage it"
    );
    for i in 0..n {
        for (a, b) in on[i]
            .output_bounds
            .lower()
            .iter()
            .zip(off[i].output_bounds.lower().iter())
        {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "domain {i}: decline-leg LOWER bits"
            );
        }
        for (a, b) in on[i]
            .output_bounds
            .upper()
            .iter()
            .zip(off[i].output_bounds.upper().iter())
        {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "domain {i}: decline-leg UPPER bits"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// #rebound-scratch parity moat: the thread-local scratch pool
// (`crate::rebound_scratch`, gate `NY_REBOUND_SCRATCH`) that recycles the CPU
// CROWN backward's faer f64 temporaries must be BIT-IDENTICAL to the historical
// re-alloc-every-layer path. These tests drive the SAME scalar spec backward
// the scored iso disjunctive lane runs (`propagate_crown_with_specs_and_node_
// bounds_and_linear_and_deadline`, engine `None` → CPU faer → `aw_f64_with_
// abssum`) with the pool forced OFF then ON and assert raw-bit equality of the
// output bounds AND every coefficient / certified-error matrix. Holds the
// A thread-local test override leaves the process-global production gate
// untouched, so unrelated parallel tests cannot observe either A/B leg.
#[cfg(test)]
mod rebound_scratch_parity {
    use super::*;
    use crate::LinearBounds;
    use std::collections::HashMap;

    /// Deterministic mixed-sign Linear/ReLU tower `dims[0] -> .. -> dims[n-1]`.
    /// Weights carry both signs and a wide-ish contraction so the certified
    /// `A·W` accumulation error (the thing the pool's f64 GEMM feeds) is
    /// non-trivial across layers.
    fn parity_tower(dims: &[usize]) -> GraphNetwork {
        let mut graph = GraphNetwork::new();
        let mut prev: Option<String> = None;
        let n_layers = dims.len() - 1;
        for l in 0..n_layers {
            let (in_d, out_d) = (dims[l], dims[l + 1]);
            let w = Array2::from_shape_fn((out_d, in_d), |(o, i)| {
                (((o * 31 + i * 17 + l * 7) % 13) as f32) / 6.0 - 1.0
            });
            let b = ndarray::Array1::from_shape_fn(out_d, |o| ((o % 5) as f32) * 0.1 - 0.2);
            let lin = LinearLayer::new(w, Some(b)).expect("valid linear");
            let lname = format!("lin{l}");
            match prev.take() {
                None => graph.add_node(GraphNode::from_input(&lname, Layer::Linear(lin))),
                Some(p) => graph.add_node(GraphNode::new(&lname, Layer::Linear(lin), vec![p])),
            }
            // ReLU after every layer except the last (keep a linear head).
            if l + 1 < n_layers {
                let rname = format!("relu{l}");
                graph.add_node(GraphNode::new(&rname, Layer::ReLU(ReLULayer), vec![lname]));
                prev = Some(rname);
            } else {
                prev = Some(lname);
            }
        }
        graph.set_output(&prev.expect("non-empty tower"));
        graph
    }

    fn assert_a2_bits_eq(on: &Array2<f32>, off: &Array2<f32>, what: &str) {
        assert_eq!(on.shape(), off.shape(), "{what}: shape");
        for (a, b) in on.iter().zip(off.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "{what}: coeff bits");
        }
    }

    fn assert_opt_a2_bits_eq(on: Option<&Array2<f32>>, off: Option<&Array2<f32>>, what: &str) {
        match (on, off) {
            (Some(a), Some(b)) => assert_a2_bits_eq(a, b, what),
            (None, None) => {}
            _ => panic!(
                "{what}: certified-error presence differs (ON={} OFF={})",
                on.is_some(),
                off.is_some()
            ),
        }
    }

    fn run_spec_backward(
        graph: &GraphNetwork,
        input: &BoundedTensor,
        spec: &Array2<f32>,
        node_bounds: &HashMap<String, BoundedTensor>,
    ) -> (BoundedTensor, Option<LinearBounds>) {
        graph
            .propagate_crown_with_specs_and_node_bounds_and_linear_and_deadline(
                input,
                spec,
                None,
                node_bounds,
                None,
            )
            .expect("scalar spec backward")
    }

    fn parity_case(dims: &[usize], label: &str) {
        let in_d = dims[0];
        let out_d = *dims.last().unwrap();
        let graph = parity_tower(dims);
        let lower = ndarray::Array1::from_shape_fn(in_d, |i| -0.3 - (i as f32) * 0.02).into_dyn();
        let upper = ndarray::Array1::from_shape_fn(in_d, |i| 0.3 + (i as f32) * 0.02).into_dyn();
        let input = BoundedTensor::new(lower, upper).expect("valid input");
        let node_bounds = graph.collect_node_bounds(&input).expect("node bounds");
        // Spec rows mirror the iso ±e_i band plus a mixed row.
        let mut spec = Array2::<f32>::zeros((out_d + 1, out_d));
        for r in 0..out_d {
            spec[[r, r]] = 1.0;
        }
        for c in 0..out_d {
            spec[[out_d, c]] = if c % 2 == 0 { 1.0 } else { -1.0 };
        }

        let pooled_calls_before = crate::rebound_scratch::pooled_call_count_for_test();
        let (off_bounds, off_lin) = {
            let _gate = crate::rebound_scratch::TestGateOverride::new(false);
            run_spec_backward(&graph, &input, &spec, &node_bounds)
        };
        let (on_bounds, on_lin) = {
            let _gate = crate::rebound_scratch::TestGateOverride::new(true);
            run_spec_backward(&graph, &input, &spec, &node_bounds)
        };
        assert!(
            crate::rebound_scratch::pooled_call_count_for_test() > pooled_calls_before,
            "{label}: pooled path was bypassed (check NY_NAIVE_F64_AW)"
        );

        // Concretized output bounds: raw f32-bit equality.
        for (a, b) in on_bounds.lower().iter().zip(off_bounds.lower().iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "{label}: output LOWER bits");
        }
        for (a, b) in on_bounds.upper().iter().zip(off_bounds.upper().iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "{label}: output UPPER bits");
        }

        // Returned LinearBounds: coefficients + biases + certified-error matrices.
        match (on_lin, off_lin) {
            (Some(on_l), Some(off_l)) => {
                assert_a2_bits_eq(
                    on_l.lower_a(),
                    off_l.lower_a(),
                    &format!("{label}: lower_a"),
                );
                assert_a2_bits_eq(
                    on_l.upper_a(),
                    off_l.upper_a(),
                    &format!("{label}: upper_a"),
                );
                for (a, b) in on_l.lower_b().iter().zip(off_l.lower_b().iter()) {
                    assert_eq!(a.to_bits(), b.to_bits(), "{label}: lower_b bits");
                }
                for (a, b) in on_l.upper_b().iter().zip(off_l.upper_b().iter()) {
                    assert_eq!(a.to_bits(), b.to_bits(), "{label}: upper_b bits");
                }
                assert_opt_a2_bits_eq(
                    on_l.lower_a_err(),
                    off_l.lower_a_err(),
                    &format!("{label}: lower_a_err"),
                );
                assert_opt_a2_bits_eq(
                    on_l.upper_a_err(),
                    off_l.upper_a_err(),
                    &format!("{label}: upper_a_err"),
                );
            }
            (None, None) => {}
            _ => panic!("{label}: LinearBounds presence differs across gate"),
        }
    }

    #[test]
    fn rebound_scratch_parity_iso_shaped() {
        // Iso ACAS diff-net shape: 5-D input, ~200 hidden neurons, linear head.
        parity_case(&[5, 64, 64, 64, 5], "iso");
    }

    #[test]
    fn rebound_scratch_parity_lsnc_shaped() {
        // lsnc AllInOne shape: wider contraction (the shared backward must stay
        // bit-identical on the lane the pool ALSO serves).
        parity_case(&[8, 120, 120, 8], "lsnc");
    }
}
