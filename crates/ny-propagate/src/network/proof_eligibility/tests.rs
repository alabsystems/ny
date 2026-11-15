// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::time::{Duration, Instant};

use ndarray::{array, Array1, Array2, ArrayD, IxDyn};
use ny_core::{NyError, VerificationSoundnessMode};
use ny_tensor::BoundedTensor;

use crate::beta_crown::domain::GraphPrecomputedBounds;
use crate::certify::work::WorkBudget;
use crate::layers::{Layer, LinearLayer, PointLinearBuildPolicy, PointLinearLayer};
use crate::network::graph_crown::spec_propagation::SpecCrownRequest;
use crate::{BetaCrownConfig, BetaCrownVerifier, GraphNode, PropagationMethod};

use super::*;

fn point_only(weights: Array2<f32>, bias: Option<Array1<f32>>) -> Layer {
    let budget = WorkBudget::new(None, 1 << 24, 1 << 28, 1 << 30, 1_024, Duration::ZERO).unwrap();
    Layer::PointLinear(
        PointLinearLayer::from_original_view(
            weights.view(),
            false,
            bias.as_ref().map(|values| values.view().into_dyn()),
            PointLinearBuildPolicy {
                budget: &budget,
                deadline: None,
                max_resident_bytes: 1 << 28,
                max_scratch_bytes: 1 << 24,
                max_work: 1 << 30,
            },
        )
        .unwrap(),
    )
}

fn box_input() -> BoundedTensor {
    BoundedTensor::new(array![-1.0].into_dyn(), array![1.0].into_dyn()).unwrap()
}

fn output_box(lo: f32, hi: f32) -> BoundedTensor {
    BoundedTensor::new(array![lo].into_dyn(), array![hi].into_dyn()).unwrap()
}

fn expired() -> Instant {
    Instant::now().checked_sub(Duration::from_secs(1)).unwrap()
}

fn diagnostic_error<T>(result: Result<T>) {
    match result {
        Err(NyError::UnsupportedOp(message)) => {
            assert!(
                message.contains("PointLinear") && message.contains("point-only"),
                "{message}"
            );
        }
        Err(error) => panic!("point-only proof authority requires UnsupportedOp, got {error}"),
        Ok(_) => panic!("point-only proof authority must refuse before returning a result"),
    }
}

fn sequential(compact: bool) -> Network {
    let mut network = Network::new();
    let weights = array![[2.0_f32]];
    let bias = array![1.0_f32];
    network.add_layer(if compact {
        point_only(weights, Some(bias))
    } else {
        Layer::Linear(LinearLayer::new(weights, Some(bias)).unwrap())
    });
    network
}

fn graph(compact: bool) -> GraphNetwork {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "affine",
        sequential(compact).into_layers().remove(0),
    ));
    graph.set_output("affine");
    graph
}

fn supplied() -> HashMap<String, BoundedTensor> {
    HashMap::from([("affine".into(), output_box(7.0, 7.0))])
}

#[test]
fn proof_point_only_precomputed_owned_expired_shortcut_refuses_before_publication() {
    let input = box_input();
    diagnostic_error(sequential(true).propagate_crown_with_precomputed_ibp(
        &input,
        vec![output_box(7.0, 7.0)],
        None,
        Some(expired()),
    ));
    // The ordinary owned-artifact expiry route remains exactly the old move.
    let expected = output_box(-1.0, 3.0);
    let actual = sequential(false)
        .propagate_crown_with_precomputed_ibp(&input, vec![expected.clone()], None, Some(expired()))
        .unwrap();
    assert_eq!(actual.lower(), expected.lower());
    assert_eq!(actual.upper(), expected.upper());
}

#[test]
fn proof_point_only_identity_seed_refusal_cannot_publish_supplied_output() {
    // This width's identity exceeds the ordinary dense admission cap while
    // the supplied vector and original one-column coefficients remain small.
    let width = 65_536;
    assert!(
        width * width * 2 * size_of::<f32>()
            > crate::network::crown_memory::cpu_crown_dense_budget_bytes(),
        "fixture must exercise identity seed admission, never allocate it"
    );
    let input = box_input();
    let fake_output = BoundedTensor::concrete(ArrayD::from_elem(IxDyn(&[width]), 7.0)).unwrap();
    let mut compact = Network::new();
    compact.add_layer(point_only(
        Array2::from_elem((width, 1), 2.0),
        Some(Array1::ones(width)),
    ));
    diagnostic_error(
        compact.propagate_crown_with_layer_bounds_and_engine_and_deadline(
            &input,
            &[fake_output],
            None,
            None,
        ),
    );
    // A real constant ordinary Linear remains eligible for the same fallback.
    let mut ordinary = Network::new();
    ordinary.add_layer(Layer::Linear(
        LinearLayer::new(
            Array2::zeros((width, 1)),
            Some(Array1::from_elem(width, 7.0)),
        )
        .unwrap(),
    ));
    let actual_output = BoundedTensor::concrete(ArrayD::from_elem(IxDyn(&[width]), 7.0)).unwrap();
    let actual = ordinary
        .propagate_crown_with_layer_bounds_and_engine_and_deadline(
            &input,
            &[actual_output.clone()],
            None,
            None,
        )
        .unwrap();
    assert_eq!(actual.lower(), actual_output.lower());
    assert_eq!(actual.upper(), actual_output.upper());
}

#[test]
fn proof_point_only_graph_supplied_zero_empty_and_truncated_specs_refuse() {
    let compact = graph(true);
    let input = box_input();
    let node_bounds = supplied();
    for rows in [0, 1] {
        let zero_specs = Array2::<f32>::zeros((rows, 1));
        diagnostic_error(
            compact.propagate_crown_with_specs_and_engine_with_node_bounds_and_deadline(
                &input,
                &zero_specs,
                None,
                &node_bounds,
                None,
            ),
        );
        diagnostic_error(
            SpecCrownRequest::new(&compact, &input, &zero_specs, None)
                .node_bounds(&node_bounds)
                .truncate_after(0)
                .run(),
        );
    }
    diagnostic_error(
        compact.propagate_crown_with_engine_and_deadline_and_node_bounds(
            &input,
            None,
            Some(expired()),
            Some(&node_bounds),
        ),
    );
    let ordinary = graph(false);
    let zero_specs = array![[0.0]];
    let valid = HashMap::from([("affine".into(), output_box(-1.0, 3.0))]);
    let bounds = ordinary
        .propagate_crown_with_specs_and_engine_with_node_bounds(&input, &zero_specs, None, &valid)
        .unwrap();
    assert!(bounds.lower()[[0]] <= 0.0 && bounds.upper()[[0]] >= 0.0);
}

#[test]
fn proof_point_only_direct_supplied_map_fallback_refuses() {
    let input = box_input();
    let spec = array![[1.0]];
    let compact = graph(true);
    diagnostic_error(compact.propagate_crown_with_specs_fallback_ibp(
        &input,
        &spec,
        &supplied(),
        "affine",
    ));
    diagnostic_error(
        compact.propagate_crown_with_specs_fallback_ibp_with_deadline(
            &input,
            &spec,
            &supplied(),
            "affine",
            Some(expired()),
        ),
    );
    let ordinary = graph(false);
    let valid = HashMap::from([("affine".into(), output_box(-1.0, 3.0))]);
    let bounds = ordinary
        .propagate_crown_with_specs_fallback_ibp(&input, &spec, &valid, "affine")
        .unwrap();
    assert!(bounds.lower()[[0]] <= -1.0 && bounds.upper()[[0]] >= 3.0);
}

#[test]
fn proof_point_only_precomputed_collector_and_empty_or_adopted_cache_refuse() {
    let compact = graph(true);
    let input = box_input();
    diagnostic_error(
        compact.collect_crown_ibp_bounds_dag_with_precomputed_ibp_and_width_threshold(
            &input,
            supplied(),
            None,
            f32::MAX,
        ),
    );
    diagnostic_error(compact.collect_forward_linear_bounds_dag_cached(&input, None, None));
    let ordinary = graph(false);
    let normal = ordinary
        .collect_forward_linear_bounds_dag_cached(&input, None, None)
        .unwrap();
    assert!(normal.contains_key("affine"));
    let mut adopted = compact;
    adopted.adopt_bound_caches_from(&ordinary);
    diagnostic_error(adopted.collect_forward_linear_bounds_dag_cached(&input, None, None));
    diagnostic_error(adopted.collect_crown_ibp_bounds_dag_with_precomputed_ibp(
        &input,
        supplied(),
        Some(expired()),
    ));
}

#[test]
fn proof_point_only_precomputed_bab_root_cannot_close_even_a_zero_objective() {
    let input = box_input();
    let compact = graph(true);
    let node_bounds = supplied();
    let output = output_box(7.0, 7.0);
    let precomputed = GraphPrecomputedBounds::new(&node_bounds, &output);
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    for objective in [[1.0], [0.0]] {
        diagnostic_error(verifier.verify_graph_relu_split_with_bounds_with_engine(
            &compact,
            &input,
            &objective,
            -1.0,
            &precomputed,
            None,
            Some(expired()),
        ));
    }
    let ordinary = graph(false);
    let valid = HashMap::from([("affine".into(), output_box(-1.0, 3.0))]);
    let output = output_box(-1.0, 3.0);
    let precomputed = GraphPrecomputedBounds::new(&valid, &output);
    let result = verifier
        .verify_graph_relu_split_with_bounds(&ordinary, &input, &[1.0], -2.0, &precomputed)
        .unwrap();
    assert_eq!(
        result.result,
        crate::beta_crown::result::BabVerificationStatus::Verified
    );
}

#[test]
fn proof_point_only_guard_tracks_mutation_and_checks_unreachable_nodes() {
    let mut network = sequential(false);
    assert!(network.ensure_proof_eligible().is_ok());
    network.layers_mut()[0] = point_only(array![[2.0]], Some(array![1.0]));
    diagnostic_error(network.ensure_proof_eligible());
    diagnostic_error(network.propagate_crown_fast(&box_input()));
    let mut ordinary = graph(false);
    ordinary.add_node(GraphNode::from_input(
        "unused_diagnostic",
        point_only(array![[0.0]], None),
    ));
    // The output still names the ordinary affine node; target/cached proof
    // walks are not allowed to qualify a point-only graph by ignoring it.
    assert_eq!(ordinary.output_name(), "affine");
    diagnostic_error(
        ordinary.propagate_crown_with_specs_and_engine_with_node_bounds(
            &box_input(),
            &array![[0.0]],
            None,
            &HashMap::from([("affine".into(), output_box(-1.0, 3.0))]),
        ),
    );
    diagnostic_error(ordinary.propagate_crown_input_linear_to_node(
        &box_input(),
        "affine",
        &HashMap::new(),
        &HashMap::new(),
        None,
        None,
    ));
}

#[test]
fn proof_point_only_other_proof_families_refuse_and_metadata_does_not_qualify() {
    let input = box_input();
    let compact = graph(true);
    diagnostic_error(compact.propagate_ibp(&input));
    diagnostic_error(compact.propagate_ibp_sound(&input));
    diagnostic_error(compact.propagate_alpha_crown(&input));
    diagnostic_error(compact.propagate_crown_batched(&input));
    diagnostic_error(compact.propagate_zonotope(&input, 1.0));
    let lowdim = crate::lowdim::prove_clause_folded(
        &compact,
        &input,
        &[crate::lowdim::FoldConjunct { s: 1.0, t: -2.0 }],
        Instant::now() + Duration::from_secs(1),
        &crate::lowdim::LowDimConfig::default(),
        &crate::lowdim::FoldCache::new(),
    );
    assert!(
        matches!(lowdim, crate::lowdim::LowDimOutcome::Declined(reason) if reason.contains("point-only"))
    );
    let zono = crate::zono_core::dag::precheck_dag(&compact);
    assert!(zono.err().unwrap().to_string().contains("point-only"));
    let seq = sequential(true);
    diagnostic_error(seq.propagate_sdp_crown(&input, &array![0.0], 1.0));
    diagnostic_error(BetaCrownVerifier::new(BetaCrownConfig::default()).verify(&seq, &input, -2.0));
    assert_eq!(
        crate::soundness::soundness_provenance_for_network(&seq, &PropagationMethod::Crown).mode(),
        VerificationSoundnessMode::Heuristic
    );
    assert_eq!(
        crate::soundness::soundness_provenance_for_graph(&compact, &PropagationMethod::Crown)
            .mode(),
        VerificationSoundnessMode::Heuristic
    );
    assert_eq!(
        crate::soundness::soundness_provenance_for_network(
            &sequential(false),
            &PropagationMethod::Crown
        )
        .mode(),
        VerificationSoundnessMode::Sound
    );
    // A named diagnostic point route remains useful and carries actual values.
    let point = BoundedTensor::concrete(array![0.5].into_dyn()).unwrap();
    let diagnostic = compact
        .propagate_concrete_point(&point, None, None)
        .unwrap();
    assert_eq!(diagnostic.lower()[[0]], 2.0);
    assert_eq!(diagnostic.lower(), diagnostic.upper());
}

#[test]
fn proof_point_only_f64_empty_batches_and_ancestor_walks_refuse_before_deadline() {
    let mut compact = graph(false);
    compact.add_node(GraphNode::from_input(
        "unused_diagnostic",
        point_only(array![[0.0]], None),
    ));
    let point = crate::network::graph_ibp_f64_cell::Interval64::point(array![0.5_f64].into_dyn());
    assert!(!compact.supports_ibp_f64_cell());
    diagnostic_error(compact.propagate_ibp_f64_cell(&point));
    diagnostic_error(compact.propagate_f64_fence_nodes(&point));
    diagnostic_error(compact.propagate_ibp_f64_centered_with_value(&point));
    diagnostic_error(compact.propagate_ibp_f64_cells_cached_with_deadline(
        &[],
        None,
        Some(expired()),
    ));
    diagnostic_error(
        compact.propagate_ibp_f64_centered_mono_cells_cached_with_deadline(
            &[],
            false,
            None,
            Some(expired()),
        ),
    );
    let ordinary = graph(false);
    assert!(ordinary.supports_ibp_f64_cell());
    let actual = ordinary.propagate_ibp_f64_cell(&point).unwrap();
    assert!(actual.lower[[0]] <= 2.0 && actual.upper[[0]] >= 2.0);
    assert!(ordinary
        .propagate_ibp_f64_cells_cached_with_deadline(&[], None, None)
        .unwrap()
        .is_empty());
    assert!(ordinary
        .propagate_ibp_f64_centered_mono_cells_cached_with_deadline(&[], false, None, None,)
        .unwrap()
        .is_empty());
}
