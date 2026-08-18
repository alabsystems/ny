// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::{
    AyTailAffineReachabilityEnvelope, AyTailRegionReluBounds, AyTailRegionSelectorEnvelope,
    AyTailRootAnchor, AyTailSharedInputReachabilityEnvelope,
};
use super::{
    anchor_cache_key, authoritative_candidate_lower, ay_region_selector_envelope_from_frontiers,
    batched_replay_gate, bounded_tail_pq_samples, build_replay_phase_census,
    certified_regional_relu_bounds, certify_ay_region_affine_reachability_partition_with,
    certify_ay_region_global_composition, certify_ay_region_partition_with,
    certify_ay_region_reachability_partition_with, certify_ay_region_selector_root_with,
    certify_ay_shared_input_root_with, certify_ay_tail_prefix_composition,
    certify_candidates_with_batched_replay, checked_ay_prefix_frontier_bytes,
    checked_budget_deadline, checked_duration_from_secs, checked_imb_region_box_plan,
    checked_region_selector_proof_deadline, checked_shared_input_bank_deadline,
    checked_shared_input_proof_deadline, concretize_region_anchors, directed_f64_lower_to_f32,
    evaluate_batched_replay_if_admitted, evaluate_before_deadline,
    evaluate_original_objective_leaf, exact_replay_box_key, farthest_support_index,
    independently_recheck_original_objective, independently_recheck_original_objectives_batched,
    index_exact_replay_leaves, k2_support_directions, maybe_run_replay_only_diagnostic,
    minimum_q_for_strict_composition, prefix_anchor_memo_allowed,
    preflight_replay_phase_census_collection, preflight_selector_k4_grid,
    rank_replay_phase_candidates, region_boxes, registered_replay_or_cpu_fallback,
    replay_only_leaf_request, replay_only_objective, replay_phase_deadline_truncated_marker,
    restore_tail_anchor_source_shape, selector_input_lift_kind,
    selector_k2_lift_enabled_for_objective, selector_k4_lift_enabled_for_objective,
    shared_input_envelope_crown_error, shared_root_envelope_from_session_cache,
    shared_support_bases, shared_support_basis_diagnosed, signed_replay_objective_plan,
    signed_replay_project_lower, split_box, standard_no_imb_objective_lower, tail_pq_self_check,
    validate_batched_replay_structure_if_admitted, validate_binary_partition_cover,
    validate_cached_shared_root_envelope_context, visit_replay_phase_entries_until_deadline,
    BatchedReplayPrevalidationShape, BatchedReplayResourceShape, ExactPrefixSession,
    FullObjectiveCertificate, ImbCandidate, RegionProposal, RegisteredFastF32ReplayEngine,
    RegisteredReplayAttempt, ReplayAuthorityEngineRoute, ReplayF64Attempt, ReplayLeafRoute,
    ReplayPhaseCandidate, ReplayPhaseCensusCollectionLimits, ReplayPhaseEmissionOutcome,
    ReplayStageTimings, SelectorInputLiftKind, SharedInputEnvelopeDeadlineStage,
    SharedInputEnvelopeDecline, SharedSupportBasisDecline, SignedReplayProjection, TailAnchorCoeff,
    MAX_AY_REGION_TOTAL_LEAVES, MAX_BATCHED_FULL_RECHECK_MEMBERSHIPS,
    MAX_BATCHED_REPLAY_ESTIMATED_BYTES, MAX_FULL_RECHECK_LEAVES,
    MAX_REPLAY_ONLY_PHASE_CENSUS_ENTRIES, REGION_SELECTOR_AY_PROOF_CAP, REPLAY_ONLY_EVALUATIONS,
    SHARED_INPUT_AY_PROOF_CAP, SHARED_INPUT_BANK_BUILD_CAP, SHARED_INPUT_EVIDENCE_REGION_COUNT,
};
use crate::beta_crown::engine::graph::input_split::grouped_semantics::disjunctive_domain_verified;
use crate::layers::{ConvTranspose2dLayer, Layer, LinearLayer, ReLULayer};
use crate::{GraphNetwork, GraphNode, LinearBounds};
use ndarray::{arr1, arr2, array, Array1, Array2, ArrayD, IxDyn};
use num_rational::BigRational;
use ny_core::GemmEngine;
use ny_tensor::{BoundedTensor, L2Constraint};
use ny_test_utils::CountingGemmEngine;
use std::cell::Cell;
use std::collections::HashMap;
use std::mem::size_of;
use std::sync::atomic::Ordering as AtomicOrdering;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

fn box1(lower: f32, upper: f32) -> BoundedTensor {
    BoundedTensor::new(array![lower].into_dyn(), array![upper].into_dyn()).expect("valid 1D box")
}

fn future_deadline() -> Instant {
    Instant::now() + Duration::from_secs(10)
}

#[test]
fn tail_pq_diagnostic_caps_samples_and_honors_expired_deadline() {
    assert_eq!(bounded_tail_pq_samples(0), 0);
    assert_eq!(bounded_tail_pq_samples(256), 256);
    assert_eq!(bounded_tail_pq_samples(usize::MAX), 4_096);

    let graph = linear_graph(1.0, 0.0);
    let root = box1(-1.0, 1.0);
    let expired = Instant::now()
        .checked_sub(Duration::from_millis(1))
        .expect("system uptime exceeds one millisecond");
    assert!(
        tail_pq_self_check(&graph, &root, &[1.0], 0.0, &[1.0], None, expired).is_nan(),
        "expired proposal-only diagnostics must fail closed without forwarding"
    );
}

fn certificate(lower: f32) -> FullObjectiveCertificate {
    FullObjectiveCertificate {
        lower,
        valid_until: future_deadline(),
    }
}

fn validate_cover(
    root: &BoundedTensor,
    terminal_boxes: &[BoundedTensor],
) -> Result<(), &'static str> {
    validate_binary_partition_cover(root, terminal_boxes, future_deadline())
}

fn linear_graph(weight: f32, bias: f32) -> GraphNetwork {
    let mut graph = GraphNetwork::new();
    let linear = LinearLayer::new(arr2(&[[weight]]), Some(arr1(&[bias])))
        .expect("valid scalar linear layer");
    graph.add_node(GraphNode::from_input("out", Layer::Linear(linear)));
    graph.set_output("out");
    graph
}

fn two_output_affine_graph() -> GraphNetwork {
    let mut graph = GraphNetwork::new();
    let linear = LinearLayer::new(
        arr2(&[[1.0_f32], [-1.0_f32]]),
        Some(arr1(&[2.0_f32, 2.0_f32])),
    )
    .expect("valid two-output affine layer");
    graph.add_node(GraphNode::from_input("out", Layer::Linear(linear)));
    graph.set_output("out");
    graph
}

/// `abs(x - 0.5) - 0.25`: both corners are +0.25, but the unsampled
/// midpoint is -0.25.
fn relu_valley_graph() -> GraphNetwork {
    let mut graph = GraphNetwork::new();
    let split = LinearLayer::new(arr2(&[[1.0], [-1.0]]), Some(arr1(&[-0.5, 0.5])))
        .expect("valid split layer");
    graph.add_node(GraphNode::from_input("split", Layer::Linear(split)));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer::new()),
        vec!["split".to_string()],
    ));
    let combine =
        LinearLayer::new(arr2(&[[1.0, 1.0]]), Some(arr1(&[-0.25]))).expect("valid combine layer");
    graph.add_node(GraphNode::new(
        "out",
        Layer::Linear(combine),
        vec!["relu".to_string()],
    ));
    graph.set_output("out");
    graph
}

fn constant_convtranspose_graph() -> GraphNetwork {
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 2]), vec![0.0_f32; 4])
        .expect("valid ConvTranspose kernel");
    let conv =
        ConvTranspose2dLayer::with_input_shape(kernel, Some(arr1(&[2.0])), (1, 1), (0, 0), 1, 1)
            .expect("valid ConvTranspose layer");
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("out", Layer::ConvTranspose2d(conv)));
    graph.set_output("out");
    graph
}

fn candidate(full_certificate: Option<FullObjectiveCertificate>) -> ImbCandidate {
    ImbCandidate {
        obj_idx: 0,
        imb_floor: 42.0,
        threshold: 0.0,
        measurement_only: false,
        full_certificate,
        terminal_boxes: Vec::new(),
        recheck_deadline: Instant::now() + Duration::from_secs(1),
    }
}

fn replay_candidate(
    obj_idx: usize,
    threshold: f32,
    terminal_boxes: Vec<BoundedTensor>,
    deadline: Instant,
    full_certificate: Option<FullObjectiveCertificate>,
) -> ImbCandidate {
    ImbCandidate {
        obj_idx,
        imb_floor: threshold + 1.0,
        threshold,
        measurement_only: false,
        full_certificate,
        terminal_boxes,
        recheck_deadline: deadline,
    }
}

fn region_proposal(
    p: f32,
    prefix_floor: f32,
    terminal_boxes: Vec<BoundedTensor>,
) -> RegionProposal {
    RegionProposal {
        floor: p + prefix_floor,
        sampled_slack: 0.0,
        p: vec![p],
        prefix_floor,
        terminal_boxes,
    }
}

fn shared_input_evidence_regions() -> Vec<BoundedTensor> {
    let count = SHARED_INPUT_EVIDENCE_REGION_COUNT as f32;
    (0..SHARED_INPUT_EVIDENCE_REGION_COUNT)
        .map(|idx| box1(idx as f32 / count, (idx + 1) as f32 / count))
        .collect()
}

fn exact_one_leaf_region_proposals<F>(
    regions: &[BoundedTensor],
    mut fields: F,
) -> Vec<RegionProposal>
where
    F: FnMut(usize) -> (f32, f32),
{
    regions
        .iter()
        .enumerate()
        .map(|(idx, region)| {
            let (p, prefix_floor) = fields(idx);
            region_proposal(p, prefix_floor, vec![region.clone()])
        })
        .collect()
}

fn region_proposal_vec(p: Vec<f32>) -> RegionProposal {
    RegionProposal {
        floor: 0.0,
        sampled_slack: 0.0,
        p,
        prefix_floor: 0.0,
        terminal_boxes: Vec::new(),
    }
}

fn box2(lower: [f32; 2], upper: [f32; 2]) -> BoundedTensor {
    BoundedTensor::new(
        array![lower[0], lower[1]].into_dyn(),
        array![upper[0], upper[1]].into_dyn(),
    )
    .expect("valid two-dimensional box")
}

fn affine_envelope(region: &BoundedTensor) -> AyTailAffineReachabilityEnvelope {
    AyTailAffineReachabilityEnvelope::from_prefix_crown(
        "seam".to_string(),
        region.clone(),
        array![[1.0, 0.0], [0.0, 1.0]],
        array![[1.0, 0.0], [0.0, 1.0]],
        Array1::zeros(2),
        array![[1.0, 0.0], [0.0, 1.0]],
        Array1::zeros(2),
    )
    .expect("valid exact identity envelope")
}

fn shared_root_envelope(
    root: &BoundedTensor,
    support_indices: Vec<usize>,
) -> AyTailSharedInputReachabilityEnvelope {
    let input_dim = root.flatten().len();
    AyTailSharedInputReachabilityEnvelope::from_prefix_crown(
        "seam".to_string(),
        root.clone(),
        root.clone(),
        support_indices,
        Array2::eye(4),
        Array2::zeros((4, input_dim)),
        Array1::zeros(4),
        Array2::zeros((4, input_dim)),
        Array1::zeros(4),
    )
    .expect("valid root-wide K4 bank")
}

fn selector_region_fixture() -> (
    BoundedTensor,
    Vec<BoundedTensor>,
    Vec<RegionProposal>,
    BoundedTensor,
) {
    let root = box1(0.0, 1.0);
    let regions: Vec<_> = (0..16)
        .map(|region_idx| box1(region_idx as f32 / 16.0, (region_idx + 1) as f32 / 16.0))
        .collect();
    let proposals = regions
        .iter()
        .enumerate()
        .map(|(region_idx, region)| RegionProposal {
            floor: 1.0,
            sampled_slack: 0.0,
            p: vec![1.0 + region_idx as f32 / 16.0, -0.5],
            prefix_floor: 1.0 + region_idx as f32 / 32.0,
            terminal_boxes: vec![region.clone()],
        })
        .collect();
    (root, regions, proposals, box2([-2.0, -3.0], [4.0, 5.0]))
}

fn selector_test_envelope(
    root: &BoundedTensor,
    regions: &[BoundedTensor],
    proposals: &[RegionProposal],
    seam: &BoundedTensor,
) -> AyTailRegionSelectorEnvelope {
    let anchors = selector_root_tail_anchors();
    let regional_relu_bounds = selector_regional_relu_bounds();
    ay_region_selector_envelope_from_frontiers(
        "seam",
        root,
        regions,
        seam,
        &anchors,
        &regional_relu_bounds,
        proposals,
        future_deadline(),
    )
    .expect("valid exact 16-region selector envelope")
}

fn selector_root_tail_anchors() -> Vec<AyTailRootAnchor> {
    vec![
        AyTailRootAnchor::from_certified_root_box("pre_1".to_string(), box1(-0.75, 0.5))
            .expect("valid first selector root anchor"),
        AyTailRootAnchor::from_certified_root_box("pre_2".to_string(), box1(-0.5, 0.75))
            .expect("valid second selector root anchor"),
    ]
}

fn selector_regional_relu_bounds() -> Vec<AyTailRegionReluBounds> {
    vec![
        AyTailRegionReluBounds::from_certified_region_box(0, "pre_1".to_string(), box1(-0.5, 0.25))
            .expect("valid region-zero first ReLU box"),
        AyTailRegionReluBounds::from_certified_region_box(0, "pre_2".to_string(), box1(-0.25, 0.5))
            .expect("valid region-zero second ReLU box"),
        AyTailRegionReluBounds::from_certified_region_box(
            1,
            "pre_1".to_string(),
            box1(-0.375, 0.375),
        )
        .expect("valid region-one first ReLU box"),
        AyTailRegionReluBounds::from_certified_region_box(
            1,
            "pre_2".to_string(),
            box1(-0.375, 0.625),
        )
        .expect("valid region-one second ReLU box"),
    ]
}

#[test]
fn identity_spec_tail_anchor_restores_spatial_source_shape() {
    let flat = BoundedTensor::new(
        array![-4.0, -3.0, -2.0, -1.0].into_dyn(),
        array![1.0, 2.0, 3.0, 4.0].into_dyn(),
    )
    .expect("valid flat identity-spec bounds");
    let spatial = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), vec![-5.0; 4]).expect("valid spatial lower"),
        ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), vec![5.0; 4]).expect("valid spatial upper"),
    )
    .expect("valid spatial source bounds");

    let restored = restore_tail_anchor_source_shape("pre", flat.clone(), 4, &spatial)
        .expect("equal element counts restore the source shape");
    assert_eq!(restored.shape(), &[1, 2, 2]);
    assert_eq!(
        restored.flatten().lower().as_slice(),
        flat.lower().as_slice(),
        "reshape must preserve the identity-spec row order exactly"
    );
    assert_eq!(
        restored.flatten().upper().as_slice(),
        flat.upper().as_slice(),
        "reshape must preserve every certified endpoint exactly"
    );

    let wrong_size = BoundedTensor::new(ArrayD::zeros(IxDyn(&[5])), ArrayD::ones(IxDyn(&[5])))
        .expect("valid mismatched source bounds");
    assert!(
        restore_tail_anchor_source_shape("pre", flat.clone(), 4, &wrong_size).is_none(),
        "an element-count mismatch must fail closed"
    );
    assert!(
        restore_tail_anchor_source_shape("pre", flat, 3, &spatial).is_none(),
        "a linear-row mismatch must fail closed"
    );
}

#[test]
fn selector_regional_relu_bounds_are_fresh_root_valid_concretizations() {
    let mut tail = GraphNetwork::new();
    let affine =
        LinearLayer::new(arr2(&[[1.0]]), Some(arr1(&[0.0]))).expect("valid scalar identity affine");
    tail.add_node(GraphNode::from_input("pre", Layer::Linear(affine)));
    tail.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer::new()),
        vec!["pre".to_string()],
    ));
    tail.set_output("relu");

    let linear = LinearBounds::new(arr2(&[[1.0]]), arr1(&[0.0]), arr2(&[[1.0]]), arr1(&[0.0]))
        .expect("exact scalar input-linear map");
    let root_box = box1(-1.0, 1.0);
    let coeffs = HashMap::from([(
        "pre".to_string(),
        TailAnchorCoeff {
            lin: linear,
            root_box: root_box.clone(),
            shape: vec![1],
        },
    )]);
    let regions: Vec<_> = (0..16)
        .map(|index| {
            let lower = -1.0 + index as f32 / 8.0;
            box1(lower, lower + 0.125)
        })
        .collect();
    let records = certified_regional_relu_bounds(&tail, &coeffs, &regions)
        .expect("regions zero and one each concretize the exact root-valid map");
    assert_eq!(records.len(), 2);
    for (record, expected_region) in records.iter().zip([0, 1]) {
        assert_eq!(record.region_index(), expected_region);
        assert_eq!(record.node_name(), "pre");
        assert!(
            record.bounds().lower()[[0]] <= regions[expected_region].lower()[[0]]
                && record.bounds().upper()[[0]] >= regions[expected_region].upper()[[0]],
            "directed concretization encloses the exact identity image"
        );
        assert!(
            record.bounds().lower()[[0]] >= root_box.lower()[[0]]
                && record.bounds().upper()[[0]] <= root_box.upper()[[0]],
            "regional authority remains root-contained"
        );
    }
}

fn exact_f32(value: f32) -> BigRational {
    BigRational::from_float(value).expect("finite binary32 has an exact rational image")
}

fn exact_affine_interval(
    coefficients: &[f32],
    bias: f32,
    lower: &[f32],
    upper: &[f32],
) -> (BigRational, BigRational) {
    assert_eq!(coefficients.len(), lower.len());
    assert_eq!(lower.len(), upper.len());
    let mut exact_lower = exact_f32(bias);
    let mut exact_upper = exact_lower.clone();
    for ((&coefficient, &lo), &hi) in coefficients.iter().zip(lower).zip(upper) {
        let (lower_input, upper_input) = if coefficient >= 0.0 {
            (lo, hi)
        } else {
            (hi, lo)
        };
        let exact_coefficient = exact_f32(coefficient);
        exact_lower += exact_coefficient.clone() * exact_f32(lower_input);
        exact_upper += exact_coefficient * exact_f32(upper_input);
    }
    (exact_lower, exact_upper)
}

fn concretize_exact_affine_for_test(
    coefficients: Vec<f32>,
    bias: f32,
    lower: Vec<f32>,
    upper: Vec<f32>,
) -> BoundedTensor {
    let input_dim = coefficients.len();
    let matrix =
        Array2::from_shape_vec((1, input_dim), coefficients).expect("one exact affine row");
    let linear = LinearBounds::new(matrix.clone(), arr1(&[bias]), matrix, arr1(&[bias]))
        .expect("finite exact affine map");
    let root_limit = 2.0_f32.powi(100);
    let coeffs = HashMap::from([(
        "pre".to_string(),
        TailAnchorCoeff {
            lin: linear,
            root_box: box1(-root_limit, root_limit),
            shape: vec![1],
        },
    )]);
    let region = BoundedTensor::new(
        Array1::from_vec(lower).into_dyn(),
        Array1::from_vec(upper).into_dyn(),
    )
    .expect("valid finite region box");
    concretize_region_anchors(&coeffs, &region)
        .expect("exact affine map concretizes")
        .remove("pre")
        .expect("source is preserved")
}

#[test]
fn region_anchor_concretization_encloses_adversarial_cancellation_exactly() {
    // A nearest-rounded accumulator loses the final unit after this cancellation.
    // The production path must instead enclose the exact real value, not merely a
    // sampled or higher-precision floating-point reference.
    let scale = 2.0_f32.powi(60);
    let coefficients = vec![scale, -scale, 1.0];
    let fixed_input = vec![1.0, 1.0, 1.0];
    let concretized = concretize_exact_affine_for_test(
        coefficients.clone(),
        0.0,
        fixed_input.clone(),
        fixed_input.clone(),
    );
    let (exact_lower, exact_upper) =
        exact_affine_interval(&coefficients, 0.0, &fixed_input, &fixed_input);
    assert_eq!(exact_lower, exact_f32(1.0));
    assert_eq!(exact_upper, exact_f32(1.0));

    let returned_lower = exact_f32(concretized.lower()[[0]]);
    let returned_upper = exact_f32(concretized.upper()[[0]]);
    assert!(
        returned_lower <= exact_lower,
        "directed lower endpoint must enclose the exact cancellation result"
    );
    assert!(
        returned_upper >= exact_upper,
        "directed upper endpoint must enclose the exact cancellation result"
    );
}

#[test]
fn region_anchor_concretization_contains_exact_rational_interval_oracle() {
    // Fixed-seed, mixed-scale dyadic cases give a bit-exact oracle while
    // exercising both sign-directed corners and severe cancellation.
    let mut state = 0x6a09_e667_f3bc_c909_u64;
    let mut next = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        state
    };

    for case_index in 0..96 {
        let cancellation_scale = 2.0_f32.powi(20 + case_index % 31);
        let cancellation_input = ((next() % 257) as i32 - 128) as f32 / 16.0;
        let mut coefficients = vec![
            cancellation_scale,
            -cancellation_scale,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ];
        let mut lower = vec![cancellation_input, cancellation_input];
        let mut upper = lower.clone();
        for coefficient in coefficients.iter_mut().skip(2) {
            let exponent = ((next() % 61) as i32) - 30;
            let mantissa = (8 + next() % 8) as f32 / 8.0;
            let sign = if next() & 1 == 0 { 1.0 } else { -1.0 };
            *coefficient = sign * mantissa * 2.0_f32.powi(exponent);

            let lo = ((next() % 513) as i32 - 256) as f32 / 16.0;
            let width = (next() % 33) as f32 / 16.0;
            lower.push(lo);
            upper.push(lo + width);
        }
        let bias = ((next() % 1025) as i32 - 512) as f32 / 32.0;
        let (exact_lower, exact_upper) = exact_affine_interval(&coefficients, bias, &lower, &upper);
        let concretized = concretize_exact_affine_for_test(coefficients, bias, lower, upper);
        let returned_lower = exact_f32(concretized.lower()[[0]]);
        let returned_upper = exact_f32(concretized.upper()[[0]]);
        assert!(
            returned_lower <= exact_lower,
            "case {case_index}: returned lower endpoint excludes the exact optimum"
        );
        assert!(
            returned_upper >= exact_upper,
            "case {case_index}: returned upper endpoint excludes the exact optimum"
        );
    }
}

#[test]
fn affine_k2_support_picker_is_rank_aware_and_deterministic() {
    let proposals = vec![
        region_proposal_vec(vec![1.0, 0.0]),
        region_proposal_vec(vec![-2.0, 0.0]),
        region_proposal_vec(vec![0.0, 3.0]),
        region_proposal_vec(vec![0.0, -4.0]),
    ];
    assert_eq!(farthest_support_index(&proposals, 0), Some(2));
    let (directions, second_idx) =
        k2_support_directions(&proposals, 0).expect("orthogonal support exists");
    assert_eq!(second_idx, 2, "equal-score ties use the earliest index");
    assert_eq!(directions, array![[1.0, 0.0], [0.0, 3.0]]);

    let rank_one = vec![
        region_proposal_vec(vec![1.0, 2.0]),
        region_proposal_vec(vec![-2.0, -4.0]),
        region_proposal_vec(vec![3.0, 6.0]),
    ];
    assert!(farthest_support_index(&rank_one, 0).is_none());
    assert!(k2_support_directions(&rank_one, 0).is_none());
}

#[test]
fn shared_support_picker_is_deterministic_and_uses_closed_descending_widths() {
    let identity_proposals: Vec<_> = (0..16)
        .map(|row| {
            let mut direction = vec![0.0_f32; 16];
            direction[row] = (row + 1) as f32;
            region_proposal_vec(direction)
        })
        .collect();
    let bases = shared_support_bases(&identity_proposals);
    assert_eq!(
        bases
            .iter()
            .map(|(directions, _)| directions.nrows())
            .collect::<Vec<_>>(),
        vec![16, 8, 4, 2]
    );
    for (_, indices) in &bases {
        assert_eq!(
            indices,
            &(0..indices.len()).collect::<Vec<_>>(),
            "equal residual-score ties use proposal order"
        );
    }

    let rank_nine: Vec<_> = (0..16)
        .map(|row| {
            let mut direction = vec![0.0_f32; 16];
            direction[row.min(8)] = 1.0;
            region_proposal_vec(direction)
        })
        .collect();
    assert_eq!(
        shared_support_bases(&rank_nine)
            .iter()
            .map(|(directions, _)| directions.nrows())
            .collect::<Vec<_>>(),
        vec![8, 4, 2],
        "rank exhaustion falls from K16 to the next closed width"
    );
}

#[test]
fn shared_support_rank_shortfall_reports_selected_indices_and_residual() {
    let proposals: Vec<_> = (0..16)
        .map(|row| {
            if row % 2 == 0 {
                region_proposal_vec(vec![1.0, 0.0, 0.0, 0.0])
            } else {
                region_proposal_vec(vec![0.0, 1.0, 0.0, 0.0])
            }
        })
        .collect();
    let decline = shared_support_basis_diagnosed(&proposals, 4)
        .expect_err("two-dimensional proposal span cannot produce K4");
    let SharedSupportBasisDecline::RankShortfall {
        requested_rows,
        selected_indices,
        max_remaining_relative_residual2,
        min_relative_residual2,
    } = &decline
    else {
        panic!("unexpected decline: {decline}");
    };
    assert_eq!(*requested_rows, 4);
    assert_eq!(selected_indices, &[0, 1]);
    assert_eq!(*max_remaining_relative_residual2, 0.0);
    assert_eq!(
        *min_relative_residual2,
        super::SHARED_SUPPORT_MIN_RELATIVE_RESIDUAL2
    );
    assert_eq!(
        decline.to_string(),
        "support-basis-rank-shortfall requested_rows=4 selected_indices=[0, 1] \
         max_remaining_relative_residual2=0.00000000000000000e0 \
         min_relative_residual2=1.00000000000000004e-10"
    );

    let (directions, selected) =
        shared_support_basis_diagnosed(&proposals, 2).expect("rank-two K2 bank");
    assert_eq!(selected, vec![0, 1]);
    assert_eq!(
        directions,
        array![[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    );
}

#[test]
fn selector_k2_accepts_tail_rank_two_without_lowering_k4_rank_cutoff() {
    let mut proposals = vec![
        region_proposal_vec(vec![1.0, 0.0, 0.0, 0.0]),
        region_proposal_vec(vec![0.0, 1.0, 0.0, 0.0]),
        region_proposal_vec(vec![1.0, 0.0, 2.0e-6, 0.0]),
        region_proposal_vec(vec![0.0, 1.0, 0.0, 2.0e-6]),
    ];
    while proposals.len() < SHARED_INPUT_EVIDENCE_REGION_COUNT {
        proposals.push(region_proposal_vec(vec![1.0, 1.0, 0.0, 0.0]));
    }

    let decline =
        shared_support_basis_diagnosed(&proposals, 4).expect_err("near-rank-two K4 must decline");
    let SharedSupportBasisDecline::RankShortfall {
        selected_indices,
        max_remaining_relative_residual2,
        min_relative_residual2,
        ..
    } = decline
    else {
        panic!("unexpected K4 decline: {decline}");
    };
    assert_eq!(selected_indices, vec![0, 1]);
    assert!(max_remaining_relative_residual2 > 0.0);
    assert!(max_remaining_relative_residual2 < min_relative_residual2);

    let (directions, selected) =
        shared_support_basis_diagnosed(&proposals, 2).expect("K2 retains the independent span");
    assert_eq!(selected, vec![0, 1]);
    assert_eq!(
        directions,
        array![[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    );
}

#[test]
fn shared_support_rank_cutoff_rejects_equal_or_smaller_residuals() {
    let proposals = vec![
        region_proposal_vec(vec![1.0, 0.0]),
        region_proposal_vec(vec![1.0, 1.0e-5]),
        region_proposal_vec(vec![1.0, -1.0e-5]),
        region_proposal_vec(vec![2.0, 0.0]),
    ];
    let decline =
        shared_support_basis_diagnosed(&proposals, 4).expect_err("near-collinear K4 must decline");
    let SharedSupportBasisDecline::RankShortfall {
        selected_indices,
        max_remaining_relative_residual2,
        min_relative_residual2,
        ..
    } = decline
    else {
        panic!("unexpected decline: {decline}");
    };
    assert_eq!(selected_indices, vec![0]);
    assert!(max_remaining_relative_residual2 > 0.0);
    assert!(max_remaining_relative_residual2 <= min_relative_residual2);
    assert_eq!(
        min_relative_residual2,
        super::SHARED_SUPPORT_MIN_RELATIVE_RESIDUAL2
    );
}

#[test]
fn shared_support_diagnostics_distinguish_malformed_inputs() {
    assert_eq!(
        shared_support_basis_diagnosed(&[], 4).expect_err("empty bank"),
        SharedSupportBasisDecline::EmptyProposals
    );

    let mut proposals: Vec<_> = (0..4)
        .map(|row| {
            let mut direction = vec![0.0_f32; 4];
            direction[row] = 1.0;
            region_proposal_vec(direction)
        })
        .collect();
    assert!(shared_support_basis_diagnosed(&proposals, 2).is_ok());
    assert_eq!(
        shared_support_basis_diagnosed(&proposals, 3).expect_err("K3 is closed"),
        SharedSupportBasisDecline::UnsupportedRequestedRows { requested_rows: 3 }
    );

    proposals[2].p.pop();
    assert_eq!(
        shared_support_basis_diagnosed(&proposals, 4).expect_err("ragged proposal bank"),
        SharedSupportBasisDecline::MalformedProposalWidth {
            proposal_idx: 2,
            expected_width: 4,
            actual_width: 3,
        }
    );
    proposals[2].p.push(0.0);
    proposals[3].p[1] = f32::NAN;
    assert_eq!(
        shared_support_basis_diagnosed(&proposals, 4).expect_err("non-finite proposal"),
        SharedSupportBasisDecline::NonFiniteProposal {
            proposal_idx: 3,
            value_idx: 1,
        }
    );
}

#[test]
fn shared_envelope_diagnostics_distinguish_operational_declines() {
    assert_eq!(
        SharedInputEnvelopeDecline::CrownError.to_string(),
        "crown-err"
    );
    assert_eq!(
        SharedInputEnvelopeDecline::MissingLinear.to_string(),
        "missing-linear"
    );
    assert_eq!(
        SharedInputEnvelopeDecline::Deadline(SharedInputEnvelopeDeadlineStage::AfterCrown)
            .to_string(),
        "deadline stage=after-crown"
    );
    assert_eq!(
        SharedInputEnvelopeDecline::MalformedBoundsShape {
            expected_values: 4,
            actual_values: 2,
        }
        .to_string(),
        "malformed-bounds-shape expected_values=4 actual_values=2"
    );
    assert_eq!(
        SharedInputEnvelopeDecline::EnvelopeRejected.to_string(),
        "envelope-rejection"
    );
    assert_eq!(
        shared_input_envelope_crown_error(&ny_core::NyError::DeadlineExceeded(
            "test deadline".to_owned()
        )),
        SharedInputEnvelopeDecline::Deadline(SharedInputEnvelopeDeadlineStage::DuringCrown)
    );
    assert_eq!(
        shared_input_envelope_crown_error(&ny_core::NyError::UnsupportedOp(
            "test operation".to_owned()
        )),
        SharedInputEnvelopeDecline::CrownError
    );
}

#[test]
fn shared_root_session_cache_builds_once_and_bypasses_later_rank_shortfall() {
    let root = box2([-1.0, -2.0], [3.0, 4.0]);
    let cache = OnceLock::new();
    let builds = Cell::new(0usize);
    let first = shared_root_envelope_from_session_cache(&cache, "seam", &root, 4, 16, || {
        builds.set(builds.get() + 1);
        Some(shared_root_envelope(&root, vec![0, 1, 2, 3]))
    })
    .expect("first valid K4 bank populates the session cache");

    let rank_two_proposals: Vec<_> = (0..16)
        .map(|row| {
            if row % 2 == 0 {
                region_proposal_vec(vec![1.0, 0.0, 0.0, 0.0])
            } else {
                region_proposal_vec(vec![0.0, 1.0, 0.0, 0.0])
            }
        })
        .collect();
    assert!(matches!(
        shared_support_basis_diagnosed(&rank_two_proposals, 4),
        Err(SharedSupportBasisDecline::RankShortfall { .. })
    ));

    let second = shared_root_envelope_from_session_cache(&cache, "seam", &root, 4, 16, || {
        panic!("a valid cached K4 bank must bypass the later rank-two builder")
    })
    .expect("later objective consumes the root-valid K4 premise");
    assert_eq!(builds.get(), 1);
    assert_eq!(first.support_indices(), second.support_indices());
    assert_eq!(first.directions(), second.directions());
}

#[test]
fn failed_shared_root_session_build_does_not_poison_retry() {
    let root = box2([-1.0, -2.0], [3.0, 4.0]);
    let cache = OnceLock::new();
    let builds = Cell::new(0usize);
    assert!(
        shared_root_envelope_from_session_cache(&cache, "seam", &root, 4, 16, || {
            builds.set(builds.get() + 1);
            None
        },)
        .is_none()
    );
    assert!(
        cache.get().is_none(),
        "a failed bank build must leave the session cell uninitialized"
    );

    assert!(
        shared_root_envelope_from_session_cache(&cache, "seam", &root, 4, 16, || {
            builds.set(builds.get() + 1);
            Some(shared_root_envelope(&root, vec![0, 1, 2, 3]))
        },)
        .is_some()
    );
    assert_eq!(builds.get(), 2);
    assert!(cache.get().is_some());
}

#[test]
fn shared_root_session_cache_context_is_strict_and_mismatch_fails_closed() {
    let root = box2([-1.0, -2.0], [3.0, 4.0]);
    let valid = shared_root_envelope(&root, vec![0, 1, 2, 3]);
    assert_eq!(
        validate_cached_shared_root_envelope_context(&valid, "seam", &root, 4, 16),
        Ok(())
    );
    assert_eq!(
        validate_cached_shared_root_envelope_context(&valid, "other", &root, 4, 16),
        Err("seam-identity-mismatch")
    );
    assert_eq!(
        validate_cached_shared_root_envelope_context(&valid, "seam", &root, 3, 16),
        Err("seam-width-mismatch")
    );
    assert_eq!(
        validate_cached_shared_root_envelope_context(&valid, "seam", &root, 4, 8),
        Err("region-count-mismatch")
    );
    let other_root = box2([-1.0, -2.0], [3.0, 5.0]);
    assert_eq!(
        validate_cached_shared_root_envelope_context(&valid, "seam", &other_root, 4, 16),
        Err("certified-root-mismatch")
    );

    let input_dim = root.flatten().len();
    let regional = AyTailSharedInputReachabilityEnvelope::from_prefix_crown(
        "seam".to_owned(),
        root.clone(),
        box2([0.0, -1.0], [2.0, 3.0]),
        vec![0, 1, 2, 3],
        Array2::eye(4),
        Array2::zeros((4, input_dim)),
        Array1::zeros(4),
        Array2::zeros((4, input_dim)),
        Array1::zeros(4),
    )
    .expect("valid non-global view for mismatch test");
    assert_eq!(
        validate_cached_shared_root_envelope_context(&regional, "seam", &root, 4, 16),
        Err("bank-is-not-global-root")
    );

    let k8 = AyTailSharedInputReachabilityEnvelope::from_prefix_crown(
        "seam".to_owned(),
        root.clone(),
        root.clone(),
        (0..8).collect(),
        Array2::eye(8),
        Array2::zeros((8, input_dim)),
        Array1::zeros(8),
        Array2::zeros((8, input_dim)),
        Array1::zeros(8),
    )
    .expect("K8 is a dark but well-formed type capability");
    assert_eq!(
        validate_cached_shared_root_envelope_context(&k8, "seam", &root, 8, 16),
        Err("bank-is-not-production-k4")
    );

    let out_of_range_support =
        shared_root_envelope(&root, vec![0, 1, 2, SHARED_INPUT_EVIDENCE_REGION_COUNT]);
    assert_eq!(
        validate_cached_shared_root_envelope_context(&out_of_range_support, "seam", &root, 4, 16,),
        Err("support-indices-invalid")
    );

    let cache = OnceLock::new();
    cache.set(valid).expect("seed valid immutable bank");
    let builder_calls = Cell::new(0usize);
    assert!(
        shared_root_envelope_from_session_cache(&cache, "other", &root, 4, 16, || {
            builder_calls.set(builder_calls.get() + 1);
            Some(shared_root_envelope(&root, vec![0, 1, 2, 3]))
        },)
        .is_none()
    );
    assert_eq!(
        builder_calls.get(),
        0,
        "a populated context mismatch is an invariant failure, not a rebuild request"
    );
}

#[test]
fn shared_bank_deadline_is_strictly_capped_and_expiry_fails_closed() {
    let now = Instant::now();
    let long_overall = now + Duration::from_mins(2);
    assert_eq!(
        checked_shared_input_bank_deadline(long_overall, now),
        Some(now + SHARED_INPUT_BANK_BUILD_CAP)
    );
    let short_overall = now + Duration::from_secs(5);
    assert_eq!(
        checked_shared_input_bank_deadline(short_overall, now),
        Some(short_overall)
    );
    assert_eq!(checked_shared_input_bank_deadline(now, now), None);
    assert_eq!(
        checked_shared_input_proof_deadline(long_overall, now),
        Some(now + SHARED_INPUT_AY_PROOF_CAP)
    );
    assert_eq!(
        checked_shared_input_proof_deadline(short_overall, now),
        Some(short_overall)
    );
    assert_eq!(checked_shared_input_proof_deadline(now, now), None);
    assert_eq!(
        checked_region_selector_proof_deadline(long_overall, now),
        Some(now + REGION_SELECTOR_AY_PROOF_CAP)
    );
    assert_eq!(
        checked_region_selector_proof_deadline(short_overall, now),
        Some(short_overall)
    );
    assert_eq!(checked_region_selector_proof_deadline(now, now), None);
}

#[test]
fn affine_envelope_discharges_coefficient_error_before_transport() {
    let lower_a = array![[1.0, 2.0], [3.0, 4.0]];
    let upper_a = lower_a.clone();
    let mut linear = LinearBounds::new(
        lower_a,
        Array1::from_vec(vec![10.0, 20.0]),
        upper_a,
        Array1::from_vec(vec![30.0, 40.0]),
    )
    .unwrap();
    linear.set_coeff_err(
        array![[0.25, 0.5], [0.0, 1.0]],
        array![[0.75, 0.25], [0.5, 0.0]],
    );
    assert!(linear.has_coeff_err());
    // Input magnitudes are [4, 5], so the lower penalties are [3.5, 5]
    // and the upper penalties are [4.25, 2].
    linear.fold_coeff_err_into_bias(&[-2.0, -3.0], &[4.0, 5.0]);
    assert!(!linear.has_coeff_err());
    let (lower_a, lower_b, upper_a, upper_b) = linear.into_parts();
    assert!(lower_b[0] <= 6.5 && lower_b[1] <= 15.0);
    assert!(upper_b[0] >= 34.25 && upper_b[1] >= 42.0);

    let transported = AyTailAffineReachabilityEnvelope::from_prefix_crown(
        "seam".to_string(),
        box2([-2.0, -3.0], [4.0, 5.0]),
        array![[1.0, 0.0], [0.0, 1.0]],
        lower_a,
        lower_b,
        upper_a,
        upper_b,
    );
    assert!(
        transported.is_some(),
        "only the outward-widened, error-free A/b parts cross the authority boundary"
    );
}

#[test]
fn shared_bank_discharges_coefficient_error_over_root_not_region() {
    let mut linear = LinearBounds::new(
        Array2::zeros((4, 2)),
        Array1::from_vec(vec![10.0; 4]),
        Array2::zeros((4, 2)),
        Array1::from_vec(vec![20.0; 4]),
    )
    .expect("valid K4 linear relation");
    linear.set_coeff_err(
        array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.25]],
        array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.25]],
    );
    // Root magnitudes are [4, 5]. A narrower regional fold would be
    // insufficient for a bank reused by all disjuncts.
    linear.fold_coeff_err_into_bias(&[-2.0, -3.0], &[4.0, 5.0]);
    assert!(!linear.has_coeff_err());
    let (lower_a, lower_b, upper_a, upper_b) = linear.into_parts();
    assert!(lower_b[2] <= 1.0);
    assert!(upper_b[2] >= 29.0);
    let root = box2([-2.0, -3.0], [4.0, 5.0]);
    let region = box2([0.0, 0.0], [1.0, 1.0]);
    assert!(
        AyTailSharedInputReachabilityEnvelope::from_prefix_crown(
            "seam".to_string(),
            root,
            region,
            vec![0, 1, 2, 3],
            Array2::eye(4),
            lower_a,
            lower_b,
            upper_a,
            upper_b,
        )
        .is_some(),
        "only the root-widened error-free A/b payload crosses authority"
    );
}

#[test]
fn affine_region_partition_preflights_all_disjuncts_before_any_proof() {
    let root = box2([0.0, 0.0], [1.0, 1.0]);
    let regions = vec![box2([0.0, 0.0], [0.5, 1.0]), box2([0.5, 0.0], [1.0, 1.0])];
    let envelopes: Vec<_> = regions.iter().map(affine_envelope).collect();
    let mut calls = Vec::new();
    let lower = certify_ay_region_affine_reachability_partition_with(
        &root,
        &regions,
        &envelopes,
        0.0,
        future_deadline(),
        |region_idx, envelope, requested, _| {
            calls.push((region_idx, envelope.region_input().clone()));
            Some(requested)
        },
    )
    .expect("exact disjunctive cover and exact per-region tokens");
    assert!(lower > 0.0);
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].0, 0);
    assert_eq!(calls[1].0, 1);

    let mut mismatched = envelopes;
    mismatched.swap(0, 1);
    let mut rejected_calls = 0usize;
    assert!(certify_ay_region_affine_reachability_partition_with(
        &root,
        &regions,
        &mismatched,
        0.0,
        future_deadline(),
        |_, _, requested, _| {
            rejected_calls += 1;
            Some(requested)
        },
    )
    .is_none());
    assert_eq!(
        rejected_calls, 0,
        "a late region mismatch must reject before admitting any exact proof"
    );
}

#[test]
fn shared_root_proof_uses_exactly_one_call_for_sixteen_region_proposals() {
    let root = box1(0.0, 1.0);
    let regions: Vec<_> = (0..16)
        .map(|index| box1(index as f32 / 16.0, (index + 1) as f32 / 16.0))
        .collect();
    let envelope = shared_root_envelope(&root, vec![0, 5, 10, 15]);
    let started = Instant::now();
    let mut calls = 0usize;
    let lower = certify_ay_shared_input_root_with(
        &root,
        &regions,
        &envelope,
        0.0,
        started + Duration::from_mins(2),
        |received, requested, proof_deadline| {
            calls += 1;
            assert_eq!(received.support_indices(), &[0, 5, 10, 15]);
            let remaining = proof_deadline
                .checked_duration_since(Instant::now())
                .expect("proof callback begins before its hard deadline");
            assert!(remaining > Duration::ZERO);
            assert!(remaining <= SHARED_INPUT_AY_PROOF_CAP);
            Some(requested)
        },
    )
    .expect("one exact global root proof");
    assert!(lower > 0.0);
    assert_eq!(calls, 1, "R=16 must still admit exactly one AY call");

    let mut inconclusive_calls = 0usize;
    assert!(certify_ay_shared_input_root_with(
        &root,
        &regions,
        &envelope,
        0.0,
        future_deadline(),
        |_, _, _| {
            inconclusive_calls += 1;
            None
        },
    )
    .is_none());
    assert_eq!(inconclusive_calls, 1);

    let mut rejected_calls = 0usize;
    let expired = Instant::now()
        .checked_sub(Duration::from_millis(1))
        .expect("system uptime exceeds one millisecond");
    assert!(certify_ay_shared_input_root_with(
        &root,
        &regions,
        &envelope,
        0.0,
        expired,
        |_, requested, _| {
            rejected_calls += 1;
            Some(requested)
        },
    )
    .is_none());
    let mut gapped_regions = regions;
    gapped_regions[7] = box1(7.0 / 16.0, 15.0 / 32.0);
    assert!(certify_ay_shared_input_root_with(
        &root,
        &gapped_regions,
        &envelope,
        0.0,
        future_deadline(),
        |_, requested, _| {
            rejected_calls += 1;
            Some(requested)
        },
    )
    .is_none());

    let four_regions: Vec<_> = (0..4)
        .map(|index| box1(index as f32 / 4.0, (index + 1) as f32 / 4.0))
        .collect();
    let four_region_envelope = shared_root_envelope(&root, vec![0, 1, 2, 3]);
    assert!(certify_ay_shared_input_root_with(
        &root,
        &four_regions,
        &four_region_envelope,
        0.0,
        future_deadline(),
        |_, requested, _| {
            rejected_calls += 1;
            Some(requested)
        },
    )
    .is_none());
    assert_eq!(
        rejected_calls, 0,
        "gapped, wrong-count, or expired proposal covers cannot admit a partial proof"
    );
}

#[test]
fn selector_root_proof_preflights_all_frontiers_then_uses_one_capped_call() {
    let (root, regions, proposals, seam) = selector_region_fixture();
    let envelope = selector_test_envelope(&root, &regions, &proposals, &seam);
    let anchors = selector_root_tail_anchors();
    let regional_relu_bounds = selector_regional_relu_bounds();
    let mut calls = 0usize;
    let lower = certify_ay_region_selector_root_with(
        &root,
        &regions,
        &proposals,
        "seam",
        &anchors,
        &regional_relu_bounds,
        &envelope,
        0.0,
        Instant::now() + Duration::from_mins(2),
        |received, requested, proof_deadline| {
            calls += 1;
            assert_eq!(received.region_inputs().len(), 16);
            assert_eq!(received.selector_coefficients().shape(), &[16, 4]);
            let remaining = proof_deadline
                .checked_duration_since(Instant::now())
                .expect("selector callback begins before its hard deadline");
            assert!(remaining > Duration::ZERO);
            assert!(remaining <= REGION_SELECTOR_AY_PROOF_CAP);
            Some(requested)
        },
    )
    .expect("one exact selector-root proof");
    assert!(lower > 0.0);
    assert_eq!(calls, 1);
}

#[test]
fn selector_frontier_or_request_mutation_fails_before_solver_admission() {
    let (root, regions, proposals, seam) = selector_region_fixture();
    let envelope = selector_test_envelope(&root, &regions, &proposals, &seam);
    let anchors = selector_root_tail_anchors();
    let regional_relu_bounds = selector_regional_relu_bounds();

    let mut malformed = proposals;
    malformed[15].terminal_boxes = vec![box1(15.0 / 16.0, 31.0 / 32.0), box1(63.0 / 64.0, 1.0)];
    assert!(
        ay_region_selector_envelope_from_frontiers(
            "seam",
            &root,
            &regions,
            &seam,
            &anchors,
            &regional_relu_bounds,
            &malformed,
            future_deadline(),
        )
        .is_none(),
        "a gap in the last prefix frontier must reject before envelope construction"
    );

    let mut calls = 0usize;
    assert!(certify_ay_region_selector_root_with(
        &root,
        &regions,
        &malformed,
        "seam",
        &anchors,
        &regional_relu_bounds,
        &envelope,
        0.0,
        future_deadline(),
        |_, requested, _| {
            calls += 1;
            Some(requested)
        },
    )
    .is_none());
    assert_eq!(
        calls, 0,
        "a malformed late frontier cannot partially admit the exact worker"
    );

    let (_, _, valid_proposals, _) = selector_region_fixture();
    assert!(certify_ay_region_selector_root_with(
        &root,
        &regions,
        &valid_proposals,
        "other-seam",
        &anchors,
        &regional_relu_bounds,
        &envelope,
        0.0,
        future_deadline(),
        |_, requested, _| {
            calls += 1;
            Some(requested)
        },
    )
    .is_none());
    let mut changed_anchors = anchors;
    changed_anchors[0] =
        AyTailRootAnchor::from_certified_root_box("pre_1".to_string(), box1(-0.75, 0.625))
            .expect("valid but different root anchor");
    assert!(certify_ay_region_selector_root_with(
        &root,
        &regions,
        &valid_proposals,
        "seam",
        &changed_anchors,
        &regional_relu_bounds,
        &envelope,
        0.0,
        future_deadline(),
        |_, requested, _| {
            calls += 1;
            Some(requested)
        },
    )
    .is_none());
    assert_eq!(
        calls, 0,
        "root-anchor endpoint mismatch cannot admit the exact worker"
    );
    assert_eq!(calls, 0, "seam identity mutation must fail closed");
}

#[test]
fn sampled_pass_cannot_authorize_a_verdict() {
    let proposal = candidate(None);
    assert_eq!(authoritative_candidate_lower(&proposal), None);
}

#[test]
fn tolerated_negative_sample_slack_cannot_authorize_a_verdict() {
    // This is accepted by the historical sampled gate: -5e-7 >= -1e-6.
    let sampled_slack = -5e-7_f32;
    let historical_tolerance = -1e-6_f32;
    assert!(sampled_slack >= historical_tolerance);
    let proposal = candidate(None);
    assert_eq!(authoritative_candidate_lower(&proposal), None);
}

#[test]
fn only_certificate_lower_is_authoritative() {
    let proposal = candidate(Some(certificate(0.25)));
    // Sampling is diagnostics only, and the decomposed IMB floor (42) is never
    // returned as proof authority.
    assert_eq!(authoritative_candidate_lower(&proposal), Some(0.25));
}

#[test]
fn ay_tail_and_prefix_compose_only_over_an_exact_cover() {
    let root = box1(0.0, 1.0);
    let leaves = vec![box1(0.0, 0.5), box1(0.5, 1.0)];
    let certificate =
        certify_ay_tail_prefix_composition(&root, &leaves, 0.75, 0.5, 1.0, future_deadline())
            .expect("exact cover and clearing outward sum");
    assert!(certificate.lower > 1.0);
    assert!(f64::from(certificate.lower) <= 1.25);

    let gap = vec![box1(0.0, 0.4), box1(0.5, 1.0)];
    assert!(
        certify_ay_tail_prefix_composition(&root, &gap, 0.75, 0.5, 1.0, future_deadline(),)
            .is_none()
    );
}

#[test]
fn ay_tail_prefix_composition_cannot_round_into_a_verdict() {
    let root = box1(0.0, 1.0);
    // The exact sum equals the threshold. Directed rounding must leave it at
    // or below that threshold, so strict verification cannot be manufactured.
    assert!(certify_ay_tail_prefix_composition(
        &root,
        std::slice::from_ref(&root),
        0.75,
        0.5,
        1.25,
        future_deadline(),
    )
    .is_none());
}

#[test]
fn ay_region_required_q_matches_the_real_directed_composition_boundary() {
    for (prefix_bits, threshold_bits, expected_q_bits) in [
        (0xb7ca_56ca, 0xb7d0_0abf, 0xb536_7e6f),
        (0x3f80_0000, 0x3f80_0000, 0x3440_0000),
        (0x0000_0000, 0x0000_0000, 0x0000_0002),
    ] {
        let q = minimum_q_for_strict_composition(
            f32::from_bits(prefix_bits),
            f32::from_bits(threshold_bits),
        )
        .expect("finite exact boundary");
        assert_eq!(
            q.to_bits(),
            expected_q_bits,
            "bit-ordered search must cross wide cancellation/ULP gaps"
        );
    }

    for (prefix, threshold) in [
        (-0.002_233_f32, -0.679_319_f32),
        (0.5, 1.0),
        (-f32::from_bits(1), 0.0),
        (f32::MAX, 1.0),
    ] {
        let q = minimum_q_for_strict_composition(prefix, threshold)
            .expect("a finite strict-composition threshold exists");
        let composed = directed_f64_lower_to_f32(f64::from(q) + f64::from(prefix))
            .expect("finite directed composition");
        assert!(composed > threshold);

        let previous = ny_tensor::next_down_f32(q);
        if previous.is_finite() {
            let previous_composed =
                directed_f64_lower_to_f32(f64::from(previous) + f64::from(prefix));
            assert!(
                previous_composed.is_none_or(|lower| lower <= threshold),
                "returned q must be the smallest binary32 threshold that composes strictly"
            );
        }
    }
    assert!(minimum_q_for_strict_composition(-f32::MAX, f32::MAX).is_none());
    assert!(minimum_q_for_strict_composition(f32::NAN, 0.0).is_none());
    assert!(minimum_q_for_strict_composition(0.0, f32::INFINITY).is_none());
}

#[test]
fn ay_region_partition_pairs_each_residual_with_its_own_prefix() {
    let root = box1(0.0, 1.0);
    let regions = vec![box1(0.0, 0.5), box1(0.5, 1.0)];
    let proposals = vec![
        region_proposal(11.0, 0.5, vec![box1(0.0, 0.5)]),
        region_proposal(22.0, 0.25, vec![box1(0.5, 1.0)]),
    ];
    let mut seen = Vec::new();
    let lower = certify_ay_region_partition_with(
        &root,
        &regions,
        &proposals,
        1.0,
        future_deadline(),
        |idx, region, p, required_q, _| {
            let flat = region.flatten();
            seen.push((idx, flat.lower()[[0]], flat.upper()[[0]], p[0]));
            assert!(required_q < if idx == 0 { 0.75 } else { 0.9 });
            Some(if idx == 0 { 0.75 } else { 0.9 })
        },
    )
    .expect("both exact region compositions clear the threshold");

    assert_eq!(seen, vec![(0, 0.0, 0.5, 11.0), (1, 0.5, 1.0, 22.0)]);
    assert!(lower > 1.0);
    assert!(f64::from(lower) <= 1.15);

    let leaves = vec![box1(0.0, 0.5), box1(0.5, 1.0)];
    let certificate =
        certify_ay_region_global_composition(&root, &leaves, lower, 1.0, future_deadline())
            .expect("all local proofs plus the exact global cover mint one token");
    assert_eq!(certificate.lower, lower);
}

#[test]
fn ay_region_reachability_pairs_each_prefix_fact_with_the_original_proof() {
    let root = box1(0.0, 1.0);
    let regions = shared_input_evidence_regions();
    let proposals = exact_one_leaf_region_proposals(&regions, |idx| {
        (11.0 + idx as f32, 0.25 + idx as f32 / 64.0)
    });
    let threshold = -0.679_319_44_f32;
    let expected = ny_tensor::next_up_f32(threshold);
    let mut seen = Vec::new();
    let lower = certify_ay_region_reachability_partition_with(
        &root,
        &regions,
        &proposals,
        threshold,
        future_deadline(),
        |idx, region, p, prefix_lower, requested_lower, _| {
            let flat = region.flatten();
            seen.push((
                idx,
                flat.lower()[[0]],
                flat.upper()[[0]],
                p[0],
                prefix_lower,
                requested_lower,
            ));
            Some(requested_lower)
        },
    )
    .expect("both conditional original-objective proofs clear the threshold");

    assert_eq!(lower.to_bits(), expected.to_bits());
    assert_eq!(seen.len(), SHARED_INPUT_EVIDENCE_REGION_COUNT);
    for (idx, &(seen_idx, lower, upper, p, prefix_lower, requested_lower)) in
        seen.iter().enumerate()
    {
        let count = SHARED_INPUT_EVIDENCE_REGION_COUNT as f32;
        assert_eq!(seen_idx, idx);
        assert_eq!(lower, idx as f32 / count);
        assert_eq!(upper, (idx + 1) as f32 / count);
        assert_eq!(p, 11.0 + idx as f32);
        assert_eq!(prefix_lower, 0.25 + idx as f32 / 64.0);
        assert_eq!(requested_lower.to_bits(), expected.to_bits());
    }
}

#[test]
fn ay_region_reachability_is_atomic_and_bit_binds_the_requested_lower() {
    let root = box1(0.0, 1.0);
    let regions = shared_input_evidence_regions();
    let proposals = exact_one_leaf_region_proposals(&regions, |idx| (idx as f32 + 1.0, 0.5));
    let mut calls = 0;
    assert!(certify_ay_region_reachability_partition_with(
        &root,
        &regions,
        &proposals,
        1.0,
        future_deadline(),
        |idx, _, _, _, requested_lower, _| {
            calls += 1;
            if idx == 0 {
                Some(requested_lower)
            } else {
                Some(ny_tensor::next_up_f32(requested_lower))
            }
        },
    )
    .is_none());
    assert_eq!(calls, 2);
}

#[test]
fn ay_region_reachability_preflights_all_covers_before_solver_admission() {
    let root = box1(0.0, 1.0);
    let regions = shared_input_evidence_regions();
    let mut proposals = exact_one_leaf_region_proposals(&regions, |idx| (idx as f32 + 1.0, 0.5));
    let late_idx = SHARED_INPUT_EVIDENCE_REGION_COUNT - 1;
    let count = SHARED_INPUT_EVIDENCE_REGION_COUNT as f32;
    let lower = late_idx as f32 / count;
    let midpoint = (late_idx as f32 + 0.5) / count;
    proposals[late_idx].terminal_boxes = vec![box1(lower, midpoint)];
    let mut calls = 0;
    assert!(certify_ay_region_reachability_partition_with(
        &root,
        &regions,
        &proposals,
        1.0,
        future_deadline(),
        |_, _, _, _, requested_lower, _| {
            calls += 1;
            Some(requested_lower)
        },
    )
    .is_none());
    assert_eq!(calls, 0);
}

#[test]
fn ay_region_reachability_threshold_edges_fail_closed_and_preserve_signed_zero() {
    let root = box1(0.0, 1.0);
    let regions = shared_input_evidence_regions();
    let proposals = exact_one_leaf_region_proposals(&regions, |idx| (idx as f32 + 1.0, 0.0));
    for threshold in [f32::MAX, f32::NAN, f32::INFINITY] {
        let mut calls = 0;
        assert!(certify_ay_region_reachability_partition_with(
            &root,
            &regions,
            &proposals,
            threshold,
            future_deadline(),
            |_, _, _, _, requested_lower, _| {
                calls += 1;
                Some(requested_lower)
            },
        )
        .is_none());
        assert_eq!(calls, 0);
    }

    let mut requested_bits = None;
    let lower = certify_ay_region_reachability_partition_with(
        &root,
        &regions,
        &proposals,
        -0.0,
        future_deadline(),
        |_, _, _, _, requested_lower, _| {
            requested_bits = Some(requested_lower.to_bits());
            Some(requested_lower)
        },
    )
    .expect("the next binary32 value above signed negative zero is finite");
    assert_eq!(requested_bits, Some(f32::from_bits(1).to_bits()));
    assert_eq!(lower.to_bits(), f32::from_bits(1).to_bits());
}

#[test]
fn ay_region_reachability_rejects_l2_semantics_before_callback() {
    let constraint = L2Constraint::new(
        ArrayD::zeros(IxDyn(&[1])),
        ArrayD::from_elem(IxDyn(&[]), 0.5),
        0,
        &[1],
    )
    .expect("valid scalar L2 constraint");
    let root = box1(0.0, 1.0).with_l2_constraint(constraint);
    let regions = shared_input_evidence_regions();
    let proposals = exact_one_leaf_region_proposals(&regions, |idx| (idx as f32 + 1.0, 0.5));
    let mut calls = 0;
    assert!(certify_ay_region_reachability_partition_with(
        &root,
        &regions,
        &proposals,
        1.0,
        future_deadline(),
        |_, _, _, _, requested_lower, _| {
            calls += 1;
            Some(requested_lower)
        },
    )
    .is_none());
    assert_eq!(calls, 0);
}

#[test]
fn ay_region_partition_is_atomic_when_any_residual_is_missing() {
    let root = box1(0.0, 1.0);
    let regions = vec![box1(0.0, 0.5), box1(0.5, 1.0)];
    let proposals = vec![
        region_proposal(1.0, 0.5, vec![box1(0.0, 0.5)]),
        region_proposal(2.0, 0.5, vec![box1(0.5, 1.0)]),
    ];
    let mut calls = 0;
    assert!(certify_ay_region_partition_with(
        &root,
        &regions,
        &proposals,
        1.0,
        future_deadline(),
        |idx, _, _, _, _| {
            calls += 1;
            (idx == 0).then_some(0.75)
        },
    )
    .is_none());
    assert_eq!(
        calls, 2,
        "the second missing proof invalidates the whole token"
    );
}

#[test]
fn ay_region_partition_preflights_every_cover_before_solver_admission() {
    let root = box1(0.0, 1.0);
    let regions = vec![box1(0.0, 0.5), box1(0.5, 1.0)];
    let proposals = vec![
        region_proposal(1.0, 0.5, vec![box1(0.0, 0.5)]),
        region_proposal(2.0, 0.5, vec![box1(0.5, 0.7), box1(0.8, 1.0)]),
    ];
    let mut calls = 0;
    assert!(certify_ay_region_partition_with(
        &root,
        &regions,
        &proposals,
        1.0,
        future_deadline(),
        |_, _, _, _, _| {
            calls += 1;
            Some(0.75)
        },
    )
    .is_none());
    assert_eq!(
        calls, 0,
        "a malformed late region must not admit any solver"
    );
}

#[test]
fn ay_region_partition_rejects_bad_grid_and_global_frontier() {
    let root = box1(0.0, 1.0);
    let bad_regions = vec![box1(0.0, 0.4), box1(0.5, 1.0)];
    let proposals = vec![
        region_proposal(1.0, 0.5, vec![box1(0.0, 0.4)]),
        region_proposal(2.0, 0.5, vec![box1(0.5, 1.0)]),
    ];
    let mut calls = 0;
    assert!(certify_ay_region_partition_with(
        &root,
        &bad_regions,
        &proposals,
        1.0,
        future_deadline(),
        |_, _, _, _, _| {
            calls += 1;
            Some(0.75)
        },
    )
    .is_none());
    assert_eq!(calls, 0);

    let bad_frontier = vec![box1(0.0, 0.4), box1(0.5, 1.0)];
    assert!(certify_ay_region_global_composition(
        &root,
        &bad_frontier,
        1.25,
        1.0,
        future_deadline(),
    )
    .is_none());
}

#[test]
fn exact_ay_never_uses_hash_only_prefix_anchor_memo_identity() {
    let input = box1(0.0, 1.0);
    let prefix_a = linear_graph(1.0, 0.0);
    let prefix_b = linear_graph(2.0, 0.0);
    assert_eq!(
        anchor_cache_key(&prefix_a, &input),
        anchor_cache_key(&prefix_b, &input),
        "the historical hash does not bind weights"
    );
    assert!(prefix_anchor_memo_allowed(false));
    assert!(
        !prefix_anchor_memo_allowed(true),
        "exact AY authority must rebuild rather than trust a hash-only hit"
    );
}

#[test]
fn exact_prefix_session_reuses_paired_prefix_and_anchor_allocations() {
    let graph = linear_graph(1.0, 0.0);
    let input = box1(0.0, 1.0);
    let prefix_builds = Cell::new(0usize);
    let anchor_builds = Cell::new(0usize);
    let mut session = ExactPrefixSession::new(&graph, &input);

    let first = session
        .prepare_with(
            &graph,
            &input,
            "out",
            future_deadline(),
            |_, _| {
                prefix_builds.set(prefix_builds.get() + 1);
                Some(linear_graph(1.0, 0.0))
            },
            |_, _, _| {
                anchor_builds.set(anchor_builds.get() + 1);
                Some(HashMap::from([("out".to_owned(), box1(0.0, 1.0))]))
            },
        )
        .expect("first exact preparation");
    let second = session
        .prepare_with(
            &graph,
            &input,
            "out",
            future_deadline(),
            |_, _| panic!("run-local hit must not rebuild the prefix"),
            |_, _, _| panic!("run-local hit must not rebuild the anchor"),
        )
        .expect("run-local reuse");

    assert_eq!(prefix_builds.get(), 1);
    assert_eq!(anchor_builds.get(), 1);
    assert!(Arc::ptr_eq(&first.prefix, &second.prefix));
    assert!(Arc::ptr_eq(
        first.anchor.as_ref().expect("first anchor"),
        second.anchor.as_ref().expect("second anchor")
    ));
    assert!(Arc::ptr_eq(
        &first.shared_root_envelope,
        &second.shared_root_envelope
    ));
}

#[test]
fn exact_prefix_session_rejects_foreign_graph_input_and_seam() {
    let graph = linear_graph(1.0, 0.0);
    let colliding_graph = linear_graph(2.0, 0.0);
    let input = box1(0.0, 1.0);
    let equal_bits_foreign_input = input.clone();
    assert_eq!(
        anchor_cache_key(&graph, &input),
        anchor_cache_key(&colliding_graph, &input),
        "regression setup requires the legacy identity collision"
    );

    let mut session = ExactPrefixSession::new(&graph, &input);
    session
        .prepare_with(
            &graph,
            &input,
            "out",
            future_deadline(),
            |_, _| Some(linear_graph(1.0, 0.0)),
            |_, _, _| Some(HashMap::from([("out".to_owned(), box1(0.0, 1.0))])),
        )
        .expect("seed exact session");

    for rejected in [
        session.prepare_with(
            &colliding_graph,
            &input,
            "out",
            future_deadline(),
            |_, _| panic!("foreign graph must fail before builders"),
            |_, _, _| panic!("foreign graph must fail before builders"),
        ),
        session.prepare_with(
            &graph,
            &equal_bits_foreign_input,
            "out",
            future_deadline(),
            |_, _| panic!("foreign input must fail before builders"),
            |_, _, _| panic!("foreign input must fail before builders"),
        ),
        session.prepare_with(
            &graph,
            &input,
            "other",
            future_deadline(),
            |_, _| panic!("seam drift must fail before builders"),
            |_, _, _| panic!("seam drift must fail before builders"),
        ),
    ] {
        assert!(rejected.is_none());
    }
}

#[test]
fn exact_prefix_session_does_not_cache_failed_or_incomplete_anchors() {
    let graph = linear_graph(1.0, 0.0);
    let input = box1(0.0, 1.0);
    let prefix_builds = Cell::new(0usize);
    let anchor_builds = Cell::new(0usize);
    let mut session = ExactPrefixSession::new(&graph, &input);

    let missing = session
        .prepare_with(
            &graph,
            &input,
            "out",
            future_deadline(),
            |_, _| {
                prefix_builds.set(prefix_builds.get() + 1);
                Some(linear_graph(1.0, 0.0))
            },
            |_, _, _| {
                anchor_builds.set(anchor_builds.get() + 1);
                Some(HashMap::from([("not-out".to_owned(), box1(0.0, 1.0))]))
            },
        )
        .expect("an incomplete map remains usable only as a local proposal");
    assert!(missing.anchor.is_some());

    let failed = session
        .prepare_with(
            &graph,
            &input,
            "out",
            future_deadline(),
            |_, _| {
                prefix_builds.set(prefix_builds.get() + 1);
                Some(linear_graph(1.0, 0.0))
            },
            |_, _, _| {
                anchor_builds.set(anchor_builds.get() + 1);
                None
            },
        )
        .expect("anchor absence retains the non-authoritative fallback");
    assert!(failed.anchor.is_none());

    let complete = session
        .prepare_with(
            &graph,
            &input,
            "out",
            future_deadline(),
            |_, _| {
                prefix_builds.set(prefix_builds.get() + 1);
                Some(linear_graph(1.0, 0.0))
            },
            |_, _, _| {
                anchor_builds.set(anchor_builds.get() + 1);
                Some(HashMap::from([("out".to_owned(), box1(0.0, 1.0))]))
            },
        )
        .expect("later objective retries and completes");
    assert!(complete.anchor.is_some());
    assert_eq!(prefix_builds.get(), 3);
    assert_eq!(anchor_builds.get(), 3);
}

#[test]
fn exact_prefix_session_rejects_expired_cache_consumption() {
    let graph = linear_graph(1.0, 0.0);
    let input = box1(0.0, 1.0);
    let mut session = ExactPrefixSession::new(&graph, &input);
    session
        .prepare_with(
            &graph,
            &input,
            "out",
            future_deadline(),
            |_, _| Some(linear_graph(1.0, 0.0)),
            |_, _, _| Some(HashMap::from([("out".to_owned(), box1(0.0, 1.0))])),
        )
        .expect("seed exact cache");
    let expired = Instant::now()
        .checked_sub(Duration::from_millis(1))
        .expect("system uptime exceeds one millisecond");
    assert!(session
        .prepare_with(
            &graph,
            &input,
            "out",
            expired,
            |_, _| panic!("expired hit must fail before builders"),
            |_, _, _| panic!("expired hit must fail before builders"),
        )
        .is_none());
}

#[test]
fn uniform_region_materialization_has_checked_count_and_byte_caps() {
    assert!(checked_imb_region_box_plan(5, 4, 2).is_some());
    assert!(checked_imb_region_box_plan(5, 1, 257).is_none());
    assert!(checked_imb_region_box_plan(usize::MAX, 4, 2).is_none());
    assert!(checked_imb_region_box_plan(5, usize::MAX, 2).is_none());
    assert_eq!(
        MAX_AY_REGION_TOTAL_LEAVES, 1_024,
        "the immutable aggregate must retain 64 prefix leaves in each R=16 cGAN region"
    );
    assert_eq!(MAX_AY_REGION_TOTAL_LEAVES / 16, 64);
    let frontier_bytes = checked_ay_prefix_frontier_bytes(5, MAX_AY_REGION_TOTAL_LEAVES)
        .expect("target region frontier fits");
    assert!(
        frontier_bytes >= 5 * MAX_AY_REGION_TOTAL_LEAVES * 8 * size_of::<f32>(),
        "byte admission must include eight simultaneous endpoint carriers"
    );
    assert!(
        checked_ay_prefix_frontier_bytes(5, MAX_AY_REGION_TOTAL_LEAVES + 1).is_none(),
        "one leaf past the immutable aggregate cap must fail before allocation"
    );
    assert!(checked_ay_prefix_frontier_bytes(usize::MAX, MAX_AY_REGION_TOTAL_LEAVES).is_none());
}

#[test]
fn ay_region_authority_accepts_exact_16_by_64_guillotine_frontier() {
    let root = box1(0.0, 1.0);
    let mut regions = Vec::with_capacity(16);
    let mut proposals = Vec::with_capacity(16);
    let mut global_frontier = Vec::with_capacity(MAX_AY_REGION_TOTAL_LEAVES);

    for region_idx in 0..16 {
        // Every endpoint is dyadic, so the f32 boxes meet bit-exactly.
        let region_lower = region_idx as f32 / 16.0;
        let region_upper = (region_idx + 1) as f32 / 16.0;
        regions.push(box1(region_lower, region_upper));

        let mut leaves = Vec::with_capacity(64);
        for local_idx in 0..64 {
            let global_idx = region_idx * 64 + local_idx;
            let lower = global_idx as f32 / 1_024.0;
            let upper = (global_idx + 1) as f32 / 1_024.0;
            let leaf = box1(lower, upper);
            global_frontier.push(leaf.clone());
            leaves.push(leaf);
        }
        proposals.push(region_proposal(1.0, 0.5, leaves));
    }

    let mut calls = 0;
    let lower = certify_ay_region_partition_with(
        &root,
        &regions,
        &proposals,
        1.0,
        future_deadline(),
        |region_idx, region, p, required_q, _| {
            calls += 1;
            assert_eq!(region.lower(), regions[region_idx].lower());
            assert_eq!(region.upper(), regions[region_idx].upper());
            assert_eq!(p, [1.0]);
            assert!(required_q < 0.75);
            Some(0.75)
        },
    )
    .expect("the checked 1,024-leaf frontier must reach all 16 exact AY obligations");

    assert_eq!(calls, 16);
    assert!(lower > 1.0);
    assert!(certify_ay_region_global_composition(
        &root,
        &global_frontier,
        lower,
        1.0,
        future_deadline(),
    )
    .is_some());

    let two_cover_shape = BatchedReplayResourceShape {
        unique_leaves: 2 * MAX_AY_REGION_TOTAL_LEAVES,
        replay_objectives: 2,
        total_memberships: 2 * MAX_AY_REGION_TOTAL_LEAVES,
        input_elements_per_leaf: 5,
        output_dim: 1,
    };
    let mut evaluator_called = false;
    assert_eq!(
        evaluate_batched_replay_if_admitted(two_cover_shape, || {
            evaluator_called = true;
            Some(())
        }),
        Some(()),
        "two worst-case cGAN authority covers must pass every replay resource guard"
    );
    assert!(evaluator_called);
}

#[test]
fn ay_region_authority_rejects_l2_semantics_before_callback() {
    let constraint = L2Constraint::new(
        ArrayD::zeros(IxDyn(&[1])),
        ArrayD::from_elem(IxDyn(&[]), 0.5),
        0,
        &[1],
    )
    .expect("valid scalar L2 constraint");
    let root = box1(0.0, 1.0).with_l2_constraint(constraint);
    let regions = vec![box1(0.0, 0.5), box1(0.5, 1.0)];
    let proposals = vec![
        region_proposal(1.0, 0.5, vec![box1(0.0, 0.5)]),
        region_proposal(2.0, 0.5, vec![box1(0.5, 1.0)]),
    ];
    let mut calls = 0;
    assert!(certify_ay_region_partition_with(
        &root,
        &regions,
        &proposals,
        1.0,
        future_deadline(),
        |_, _, _, _, _| {
            calls += 1;
            Some(0.75)
        },
    )
    .is_none());
    assert_eq!(calls, 0);
}

#[test]
fn ay_region_total_leaf_cap_is_checked_before_callback() {
    let root = box1(0.0, 1.0);
    let regions = vec![box1(0.0, 0.5), box1(0.5, 1.0)];
    let proposals = vec![
        region_proposal(1.0, 0.5, vec![box1(0.0, 0.5)]),
        region_proposal(2.0, 0.5, vec![box1(0.5, 1.0); MAX_AY_REGION_TOTAL_LEAVES]),
    ];
    let mut calls = 0;
    assert!(certify_ay_region_partition_with(
        &root,
        &regions,
        &proposals,
        1.0,
        future_deadline(),
        |_, _, _, _, _| {
            calls += 1;
            Some(0.75)
        },
    )
    .is_none());
    assert_eq!(calls, 0);
}

#[test]
fn non_finite_full_recheck_fails_closed() {
    for lower in [f32::NAN, f32::NEG_INFINITY, f32::INFINITY] {
        let proposal = candidate(Some(certificate(lower)));
        assert_eq!(authoritative_candidate_lower(&proposal), None);
    }
}

#[test]
fn loaded_measurement_only_candidate_stays_non_authoritative() {
    let mut proposal = candidate(Some(certificate(0.25)));
    proposal.measurement_only = true;
    assert_eq!(authoritative_candidate_lower(&proposal), None);
}

#[test]
fn expired_full_certificate_cannot_be_consumed() {
    let expired = Instant::now()
        .checked_sub(Duration::from_millis(1))
        .expect("system uptime exceeds one millisecond");
    let proposal = candidate(Some(FullObjectiveCertificate {
        lower: 0.25,
        valid_until: expired,
    }));
    assert_eq!(authoritative_candidate_lower(&proposal), None);
}

#[test]
fn exact_binary_partition_cover_accepts_nested_splits() {
    let root = box1(0.0, 1.0);
    let leaves = vec![box1(0.0, 0.25), box1(0.25, 0.5), box1(0.5, 1.0)];
    assert_eq!(validate_cover(&root, &leaves), Ok(()));
}

#[test]
fn partition_coverage_gap_fails_closed() {
    let root = box1(0.0, 1.0);
    let leaves = vec![box1(0.0, 0.4), box1(0.5, 1.0)];
    assert!(validate_cover(&root, &leaves).is_err());
}

#[test]
fn partition_overlap_fails_closed() {
    let root = box1(0.0, 1.0);
    let leaves = vec![box1(0.0, 0.6), box1(0.4, 1.0)];
    assert!(validate_cover(&root, &leaves).is_err());
}

#[test]
fn non_finite_partition_bounds_fail_closed() {
    let root = box1(0.0, 1.0);
    let infinite = BoundedTensor::new_allow_infinite(
        array![f32::NEG_INFINITY].into_dyn(),
        array![f32::INFINITY].into_dyn(),
    )
    .expect("infinite conservative tensor");
    assert!(validate_cover(&root, &[infinite]).is_err());
    assert!(BoundedTensor::new(array![f32::NAN].into_dyn(), array![1.0_f32].into_dyn()).is_err());
}

#[test]
fn expired_partition_validation_deadline_fails_closed() {
    let root = box1(0.0, 1.0);
    let expired = Instant::now()
        .checked_sub(Duration::from_millis(1))
        .expect("system uptime exceeds one millisecond");
    assert!(validate_binary_partition_cover(&root, std::slice::from_ref(&root), expired,).is_err());
}

#[test]
fn huge_finite_budget_fails_closed_without_panicking() {
    let huge = 1e300_f64;
    assert!(huge.is_finite());
    assert!(checked_duration_from_secs(huge).is_none());
    assert!(checked_budget_deadline(Instant::now(), huge, None).is_none());
}

#[test]
fn evaluator_finishing_after_deadline_is_rejected() {
    let deadline = Instant::now() + Duration::from_millis(200);
    let result = evaluate_before_deadline(deadline, || {
        std::thread::sleep(Duration::from_millis(250));
        Some(42_u32)
    });
    assert_eq!(result, None);
}

#[test]
fn replay_only_request_requires_exact_gate_and_decimal_leaf() {
    assert_eq!(replay_only_leaf_request(None, None), Ok(None));
    assert_eq!(replay_only_leaf_request(Some("0"), Some("4")), Ok(None));
    assert!(replay_only_leaf_request(Some("true"), Some("4")).is_err());
    assert_eq!(replay_only_leaf_request(Some("1"), Some("4")), Ok(Some(4)));

    for malformed in [
        None,
        Some(""),
        Some("+4"),
        Some("-1"),
        Some(" 4"),
        Some("4 "),
    ] {
        assert!(replay_only_leaf_request(Some("1"), malformed).is_err());
    }
    assert_eq!(replay_only_objective(None, 2), Ok(0));
    assert_eq!(replay_only_objective(Some("1"), 2), Ok(1));
    assert!(replay_only_objective(Some("2"), 2).is_err());
    assert!(replay_only_objective(Some(" 0"), 2).is_err());
}

#[test]
fn batched_replay_gate_is_strict_and_default_off() {
    assert_eq!(batched_replay_gate(None), Ok(false));
    assert_eq!(batched_replay_gate(Some("0")), Ok(false));
    assert_eq!(batched_replay_gate(Some("1")), Ok(true));
    for malformed in ["", "00", "true", " 1", "1 ", "+1"] {
        assert!(batched_replay_gate(Some(malformed)).is_err());
    }
}

#[test]
fn replay_authority_engine_route_labels_are_stable() {
    assert_eq!(ReplayAuthorityEngineRoute::Caller.as_str(), "caller");
    assert_eq!(
        ReplayAuthorityEngineRoute::RegisteredFastF32.as_str(),
        "registered_fast_f32"
    );
    assert_eq!(
        ReplayAuthorityEngineRoute::CpuFallback.as_str(),
        "cpu_fallback"
    );
}

#[test]
fn registered_replay_failure_never_retries_cpu() {
    let cpu_calls = Cell::new(0_usize);
    let selected_failure =
        registered_replay_or_cpu_fallback(RegisteredReplayAttempt::Evaluated(None::<u32>), || {
            cpu_calls.set(cpu_calls.get() + 1);
            Some(7)
        });
    assert_eq!(selected_failure, None);
    assert_eq!(
        cpu_calls.get(),
        0,
        "an evaluated failure is not registry unavailability"
    );

    let unavailable =
        registered_replay_or_cpu_fallback(RegisteredReplayAttempt::Unavailable, || {
            cpu_calls.set(cpu_calls.get() + 1);
            Some(11)
        });
    assert_eq!(unavailable, Some(11));
    assert_eq!(
        cpu_calls.get(),
        1,
        "only an unavailable registry may enter the CPU fallback"
    );
}

#[test]
fn registered_fast_f32_replay_adapter_confines_optional_capabilities() {
    let registered = CountingGemmEngine::new();
    let adapter = RegisteredFastF32ReplayEngine(&registered);
    let product = adapter
        .gemm_f32(1, 2, 1, &[2.0, 3.0], &[4.0, 5.0])
        .expect("registered RN-f32 GEMM delegates");

    assert_eq!(product, vec![23.0]);
    assert_eq!(registered.gemm_calls(), 1);
    assert!(
        adapter.gemm_f64(1, 1, 1, &[1.0_f64], &[1.0_f64]).is_err(),
        "the fast-f32 registry must not acquire an f64 capability"
    );
    assert!(
        adapter.as_gpu_crown_backward().is_none(),
        "the fast-f32 registry must not acquire a GPU-CROWN capability"
    );
    assert!(
        adapter.as_gpu_ibp_forward().is_none(),
        "the fast-f32 registry must not acquire a GPU-IBP capability"
    );
}

#[test]
fn signed_replay_plan_quotients_exact_opposites_and_preserves_other_rows() {
    let positive = [1.0_f32, 0.0];
    // Ordinary +0 is mathematically the negation of either signed zero.
    let negative = [-1.0_f32, 0.0];
    let independent = [0.0_f32, 1.0];
    let duplicate = [1.0_f32, 0.0];
    let objectives: Vec<&[f32]> = vec![&positive, &negative, &independent, &duplicate];

    let plan = signed_replay_objective_plan(&objectives, 2).expect("finite equal-width rows");
    assert_eq!(
        plan.representatives,
        vec![positive.as_slice(), independent.as_slice()]
    );
    assert_eq!(
        plan.projections,
        vec![
            SignedReplayProjection {
                representative: 0,
                use_negated_upper: false,
            },
            SignedReplayProjection {
                representative: 0,
                use_negated_upper: true,
            },
            SignedReplayProjection {
                representative: 1,
                use_negated_upper: false,
            },
            SignedReplayProjection {
                representative: 0,
                use_negated_upper: false,
            },
        ]
    );
}

#[test]
fn signed_replay_plan_leaves_zero_orientation_unquotiented_and_refuses_bad_rows() {
    let positive_zero = [0.0_f32];
    let negative_zero = [-0.0_f32];
    let zeros: Vec<&[f32]> = vec![&positive_zero, &negative_zero];
    let plan = signed_replay_objective_plan(&zeros, 1).expect("finite zero rows");
    assert_eq!(
        plan.representatives.len(),
        2,
        "an all-zero row has no non-zero coefficient that can bind orientation"
    );
    assert!(plan
        .projections
        .iter()
        .all(|projection| !projection.use_negated_upper));

    let finite = [1.0_f32];
    let nan = [f32::NAN];
    let wide = [1.0_f32, 2.0];
    assert!(signed_replay_objective_plan(&[&finite, &nan], 1).is_none());
    assert!(signed_replay_objective_plan(&[&finite, &wide], 1).is_none());
    assert!(signed_replay_objective_plan(&[&finite], 0).is_none());
}

#[test]
fn signed_replay_negated_upper_projection_is_exact_and_fail_closed() {
    let opposite = SignedReplayProjection {
        representative: 0,
        use_negated_upper: true,
    };
    assert_eq!(
        signed_replay_project_lower(&[(1.0, 3.0)], opposite),
        Some(-3.0)
    );

    let projected_zero =
        signed_replay_project_lower(&[(-0.0, 0.0)], opposite).expect("finite zero interval");
    assert_eq!(
        projected_zero.to_bits(),
        (-0.0_f32).to_bits(),
        "negation is an exact sign-bit flip, including zero"
    );
    assert!(signed_replay_project_lower(&[(f32::NEG_INFINITY, 1.0)], opposite).is_none());
    assert!(signed_replay_project_lower(&[(0.0, f32::INFINITY)], opposite).is_none());
    assert!(signed_replay_project_lower(&[(2.0, 1.0)], opposite).is_none());
    assert!(signed_replay_project_lower(&[], opposite).is_none());
}

#[test]
fn exact_replay_box_identity_preserves_bound_bits() {
    let positive_zero = box1(0.0, 1.0);
    let negative_zero = box1(-0.0, 1.0);
    assert_ne!(
        exact_replay_box_key(&positive_zero),
        exact_replay_box_key(&negative_zero),
        "signed-zero leaf endpoints must not be silently canonicalized"
    );
    assert_eq!(
        exact_replay_box_key(&positive_zero),
        exact_replay_box_key(&positive_zero.clone())
    );
}

#[test]
fn exact_replay_box_identity_rejects_extra_l2_semantics() {
    let constraint = L2Constraint::new(
        array![0.0_f32].into_dyn(),
        ArrayD::from_elem(IxDyn(&[]), 1.0),
        0,
        &[1],
    )
    .expect("valid scalar-slice L2 constraint");
    let annotated = box1(-1.0, 1.0).with_l2_constraint(constraint);
    assert!(
        exact_replay_box_key(&annotated).is_none(),
        "box-equal leaves with different semantic constraints must never deduplicate"
    );
}

#[test]
fn batched_replay_deduplicates_bit_identical_cross_cover_leaves_only() {
    let cover_a = vec![box1(-1.0, 0.0), box1(0.0, 1.0)];
    let cover_b = vec![box1(-1.0, 0.0), box1(0.0, 1.0)];
    let partitions: Vec<&[BoundedTensor]> = vec![&cover_a, &cover_b];
    let (unique, memberships) =
        index_exact_replay_leaves(&partitions, future_deadline()).expect("valid exact leaves");
    assert_eq!(unique.len(), 2);
    assert_eq!(memberships, vec![vec![0, 1], vec![0, 1]]);

    let signed_zero_cover = vec![box1(-1.0, -0.0), box1(-0.0, 1.0)];
    let partitions: Vec<&[BoundedTensor]> = vec![&cover_a, &signed_zero_cover];
    let (unique, memberships) =
        index_exact_replay_leaves(&partitions, future_deadline()).expect("finite exact leaves");
    assert_eq!(
        unique.len(),
        4,
        "signed-zero endpoints must not deduplicate"
    );
    assert_eq!(memberships, vec![vec![0, 1], vec![2, 3]]);
}

#[test]
fn batched_replay_rejects_cartesian_bomb_before_evaluator() {
    // Every scalar admission passes: 8,192 unique leaves, 8,193 non-empty
    // covers, and exactly 16,384 total memberships can arise from one 8,192
    // leaf cover plus 8,192 one-leaf covers. Their dense Cartesian result is
    // ~67M row/domain cells and must be declined before either evaluator route.
    let shape = BatchedReplayResourceShape {
        unique_leaves: MAX_FULL_RECHECK_LEAVES / 2,
        replay_objectives: MAX_FULL_RECHECK_LEAVES / 2 + 1,
        total_memberships: MAX_BATCHED_FULL_RECHECK_MEMBERSHIPS,
        input_elements_per_leaf: 1,
        output_dim: 1,
    };
    let mut evaluator_called = false;
    let result = evaluate_batched_replay_if_admitted(shape, || {
        evaluator_called = true;
        Some(())
    });
    assert!(result.is_none());
    assert!(
        !evaluator_called,
        "oversized Cartesian product must be rejected before evaluator entry"
    );
}

#[test]
fn batched_replay_rejects_large_input_before_cover_copies() {
    let bytes_per_membership_element = 10 * size_of::<f32>();
    let input_elements_per_leaf =
        MAX_BATCHED_REPLAY_ESTIMATED_BYTES / bytes_per_membership_element + 1;
    let shape = BatchedReplayPrevalidationShape {
        total_memberships: 1,
        input_elements_per_leaf,
    };
    let mut cover_work_called = false;
    let result = validate_batched_replay_structure_if_admitted(shape, || {
        cover_work_called = true;
        Some(())
    });
    assert!(result.is_none());
    assert!(
        !cover_work_called,
        "oversized input must be rejected before flattening or indexing its cover"
    );
}

#[test]
fn batched_replay_counts_retained_ay_covers_before_copy_or_evaluator_work() {
    let retained_certified_memberships = MAX_BATCHED_FULL_RECHECK_MEMBERSHIPS;
    let replay_memberships = 1usize;
    let total_live_memberships = retained_certified_memberships
        .checked_add(replay_memberships)
        .expect("small checked sum");
    let mut cover_work_called = false;
    let result = validate_batched_replay_structure_if_admitted(
        BatchedReplayPrevalidationShape {
            total_memberships: total_live_memberships,
            input_elements_per_leaf: 1,
        },
        || {
            cover_work_called = true;
            Some(())
        },
    );
    assert!(result.is_none());
    assert!(
        !cover_work_called,
        "aggregate retained AY covers must reject before cover copies"
    );

    let graph = linear_graph(1.0, 2.0);
    let root = box1(-1.0, 1.0);
    let cover = vec![root.clone()];
    let objective = [1.0_f32];
    assert!(independently_recheck_original_objectives_batched(
        &graph,
        &root,
        &[&cover],
        &[&objective],
        &[0.5],
        retained_certified_memberships,
        None,
        future_deadline(),
    )
    .is_none());
}

#[test]
fn batched_replay_rejects_linear_workspace_bomb_before_evaluator() {
    // This shape passes the leaf, membership, dense-cell, and spec-element
    // caps. The retained+cloned input LinearBounds alone can exceed 1 GiB:
    // U=512 domains × O=2048 objectives × D=64 input coefficients ×
    // 8 lower/upper/error/clone carriers × sizeof(f32).
    let shape = BatchedReplayResourceShape {
        unique_leaves: 512,
        replay_objectives: 2_048,
        total_memberships: 2_558,
        input_elements_per_leaf: 64,
        output_dim: 1,
    };
    let mut evaluator_called = false;
    let result = evaluate_batched_replay_if_admitted(shape, || {
        evaluator_called = true;
        Some(())
    });
    assert!(result.is_none());
    assert!(
        !evaluator_called,
        "oversized retained LinearBounds must be rejected before evaluator entry"
    );
}

#[test]
fn batched_replay_counts_duplicate_caller_boxes_in_peak_memory() {
    // Exact-bit deduplication reduces evaluator work to U domains, but it does
    // not free the M caller-owned terminal boxes. This shape was just below
    // 256 MiB under a U-only input estimate; its still-live duplicate endpoint
    // storage pushes the known peak over the immutable cap.
    let shape = BatchedReplayResourceShape {
        unique_leaves: 512,
        replay_objectives: 245,
        total_memberships: MAX_BATCHED_FULL_RECHECK_MEMBERSHIPS,
        input_elements_per_leaf: 64,
        output_dim: 1,
    };
    let mut evaluator_called = false;
    let result = evaluate_batched_replay_if_admitted(shape, || {
        evaluator_called = true;
        Some(())
    });
    assert!(result.is_none());
    assert!(
        !evaluator_called,
        "deduplicated execution must still account for every live caller box"
    );
}

#[test]
fn uniform_region_leaf_four_matches_run_region_loop_order() {
    // `region_boxes` is the exact constructor consumed by `run_region_loop`.
    // Its mixed-radix order makes index 4 (binary 0100) the high half of
    // split dim 2 and the low half of dims 0, 1, and 4.
    let root = BoundedTensor::new(
        array![0.0_f32, 0.0, 0.0, 7.0, 0.0].into_dyn(),
        array![2.0_f32, 2.0, 2.0, 7.0, 2.0].into_dyn(),
    )
    .expect("valid root");
    let regions = region_boxes(&root, &[0, 1, 2, 4], 2);
    assert_eq!(regions.len(), 16);
    validate_binary_partition_cover(&root, &regions, future_deadline())
        .expect("uniform regions form an exact cover");
    let leaf = regions[4].flatten();
    assert_eq!(
        leaf.lower().as_slice().expect("contiguous"),
        &[0.0, 0.0, 1.0, 7.0, 0.0]
    );
    assert_eq!(
        leaf.upper().as_slice().expect("contiguous"),
        &[1.0, 1.0, 2.0, 7.0, 1.0]
    );
}

#[test]
fn selector_input_lift_canary_gates_are_exact_conflict_closed_and_grid_checked() {
    assert!(selector_k4_lift_enabled_for_objective(Some("1"), 1));
    assert!(!selector_k4_lift_enabled_for_objective(Some("1"), 0));
    assert!(!selector_k4_lift_enabled_for_objective(Some("1"), 2));
    assert!(selector_k2_lift_enabled_for_objective(Some("1"), 1));
    assert!(!selector_k2_lift_enabled_for_objective(Some("1"), 0));
    assert!(!selector_k2_lift_enabled_for_objective(Some("1"), 2));
    for value in [None, Some(""), Some("0"), Some("true"), Some(" 1")] {
        assert!(!selector_k4_lift_enabled_for_objective(value, 1));
        assert!(!selector_k2_lift_enabled_for_objective(value, 1));
    }
    assert_eq!(
        selector_input_lift_kind(Some("1"), None, 1),
        Ok(Some(SelectorInputLiftKind::K2))
    );
    assert_eq!(
        selector_input_lift_kind(None, Some("1"), 1),
        Ok(Some(SelectorInputLiftKind::K4))
    );
    assert_eq!(selector_input_lift_kind(Some("1"), None, 0), Ok(None));
    assert_eq!(selector_input_lift_kind(None, Some("1"), 0), Ok(None));
    assert_eq!(
        selector_input_lift_kind(Some("true"), Some("1"), 1),
        Ok(Some(SelectorInputLiftKind::K4)),
        "malformed values remain off and do not create a false conflict"
    );
    assert_eq!(
        selector_input_lift_kind(Some("1"), Some("1"), 1),
        Err("conflicting-selector-k2-k4-lift-gates")
    );
    assert_eq!(
        selector_input_lift_kind(Some("1"), Some("1"), 0),
        Err("conflicting-selector-k2-k4-lift-gates"),
        "conflicting authority requests fail closed before objective filtering"
    );

    let root = BoundedTensor::new(
        array![-4.0_f32, 2.0, -9.0, 11.0, 0.0].into_dyn(),
        array![8.0_f32, 10.0, -1.0, 11.0, 20.0].into_dyn(),
    )
    .expect("valid root");
    let regions = region_boxes(&root, &[0, 1, 2, 4], 2);
    assert_eq!(regions.len(), 16);
    preflight_selector_k4_grid(&root, &regions)
        .expect("the production mixed-radix little-endian grid is admitted");

    let mut misordered = regions;
    misordered.swap(1, 8);
    assert!(
        preflight_selector_k4_grid(&root, &misordered).is_err(),
        "region count alone cannot authorize a differently ordered 16-cell grid"
    );
}

#[test]
fn replay_phase_census_uses_distance_to_stability_order_and_deterministic_ties() {
    let closest = ReplayPhaseCandidate::new(3, 8, -0.03125, 0.96875).expect("unstable");
    let tied_wide = ReplayPhaseCandidate::new(4, 20, -0.25, 2.0).expect("unstable");
    let tied_early_flat = ReplayPhaseCandidate::new(2, 5, -0.25, 1.0).expect("unstable");
    let tied_late_flat = ReplayPhaseCandidate::new(2, 9, -0.25, 1.0).expect("unstable");

    let ranked = rank_replay_phase_candidates(
        [tied_late_flat, tied_early_flat, tied_wide, closest],
        MAX_REPLAY_ONLY_PHASE_CENSUS_ENTRIES,
    );
    let identities: Vec<(usize, usize)> = ranked
        .iter()
        .map(|candidate| (candidate.relu_exec_index, candidate.flat_index))
        .collect();
    assert_eq!(identities, vec![(3, 8), (4, 20), (2, 5), (2, 9)]);
    assert_eq!(ranked[0].score, 0.03125);
    assert_eq!(ranked[1].score, 0.25);
}

#[test]
fn replay_phase_census_preserves_endpoint_bits_and_digest() {
    let graph = relu_valley_graph();
    let lower = f32::from_bits(0xbe80_0001);
    let upper = f32::from_bits(0x3f00_0001);
    let producer_bounds = BoundedTensor::new(
        array![lower, -2.0_f32].into_dyn(),
        array![upper, -0.0_f32].into_dyn(),
    )
    .expect("valid producer bounds");
    let node_bounds = HashMap::from([("split".to_string(), producer_bounds)]);
    let input = box1(-1.0, 1.0);

    let first = build_replay_phase_census(
        &graph,
        &input,
        &node_bounds,
        future_deadline(),
        MAX_REPLAY_ONLY_PHASE_CENSUS_ENTRIES,
    )
    .expect("census");
    let second = build_replay_phase_census(
        &graph,
        &input,
        &node_bounds,
        future_deadline(),
        MAX_REPLAY_ONLY_PHASE_CENSUS_ENTRIES,
    )
    .expect("repeat census");

    assert_eq!(first.relu_nodes, 1);
    assert_eq!(first.coordinates, 2);
    assert_eq!(first.unstable, 1);
    assert_eq!(first.stable_inactive, 1);
    assert_eq!(first.stable_active, 0);
    assert_eq!(first.ranked.len(), 1);
    assert_eq!(first.ranked[0].lower.to_bits(), lower.to_bits());
    assert_eq!(first.ranked[0].upper.to_bits(), upper.to_bits());
    assert_eq!(
        first.node_box_endpoint_digest,
        second.node_box_endpoint_digest
    );
}

#[test]
fn replay_phase_census_output_is_hard_capped() {
    let candidates = (0..=MAX_REPLAY_ONLY_PHASE_CENSUS_ENTRIES)
        .map(|flat_index| ReplayPhaseCandidate::new(7, flat_index, -0.5, 0.5).expect("unstable"));
    let ranked = rank_replay_phase_candidates(candidates, usize::MAX);
    assert_eq!(ranked.len(), MAX_REPLAY_ONLY_PHASE_CENSUS_ENTRIES);
    assert_eq!(
        ranked.first().map(|candidate| candidate.flat_index),
        Some(0)
    );
    assert_eq!(
        ranked.last().map(|candidate| candidate.flat_index),
        Some(MAX_REPLAY_ONLY_PHASE_CENSUS_ENTRIES - 1)
    );
}

#[test]
fn replay_phase_preflight_counts_oversized_non_relu_cached_output() {
    let mut graph = relu_valley_graph();
    let oversized = LinearLayer::new(Array2::<f32>::zeros((8, 1)), Some(Array1::<f32>::zeros(8)))
        .expect("valid disconnected non-ReLU output");
    graph.add_node(GraphNode::from_input(
        "oversized_nonrelu",
        Layer::Linear(oversized),
    ));

    let cached_box = |len: usize| {
        BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[len]), -1.0),
            ArrayD::from_elem(IxDyn(&[len]), 1.0),
        )
        .expect("valid cached box")
    };
    let node_bounds = HashMap::from([
        ("split".to_string(), cached_box(2)),
        ("relu".to_string(), cached_box(2)),
        ("out".to_string(), cached_box(1)),
        ("oversized_nonrelu".to_string(), cached_box(8)),
    ]);
    let input_l2 = L2Constraint::new(
        array![0.0_f32].into_dyn(),
        ArrayD::from_elem(IxDyn(&[]), 1.0),
        0,
        &[1],
    )
    .expect("valid scalar input L2 annotation");
    let input = box1(-1.0, 1.0).with_l2_constraint(input_l2);
    let limits = ReplayPhaseCensusCollectionLimits {
        graph_nodes: 8,
        // input endpoints=2 + input L2 center/radius=2, split=4, relu=4,
        // out=2, non-ReLU=16 => 30 retained f32 values.
        cached_f32_values: 29,
        name_bytes: 1_024,
    };

    assert_eq!(
        preflight_replay_phase_census_collection(
            &graph,
            &input,
            &node_bounds,
            future_deadline(),
            limits,
        ),
        Err("cached-f32-value-cap"),
        "a tiny ReLU producer must not hide an oversized cached non-ReLU output"
    );
    preflight_replay_phase_census_collection(
        &graph,
        &input,
        &node_bounds,
        future_deadline(),
        ReplayPhaseCensusCollectionLimits {
            cached_f32_values: 30,
            ..limits
        },
    )
    .expect("the exact full-cache payload fits at its inclusive cap");
}

#[test]
fn replay_phase_entry_output_polls_and_marks_deadline_truncation() {
    let ranked = rank_replay_phase_candidates(
        (0..3).map(|flat_index| {
            ReplayPhaseCandidate::new(7, flat_index, -0.5, 0.5).expect("unstable")
        }),
        MAX_REPLAY_ONLY_PHASE_CENSUS_ENTRIES,
    );
    let polls = Cell::new(0usize);
    let mut emitted_ranks = Vec::new();
    let outcome = visit_replay_phase_entries_until_deadline(
        &ranked,
        || {
            let poll = polls.get();
            polls.set(poll + 1);
            poll < 2
        },
        |rank, _candidate| emitted_ranks.push(rank),
    );

    assert_eq!(emitted_ranks, vec![0, 1]);
    assert_eq!(polls.get(), 3, "every attempted entry must be pre-polled");
    assert_eq!(
        outcome,
        ReplayPhaseEmissionOutcome {
            emitted: 2,
            deadline_truncated: true,
        }
    );
    assert_eq!(
        replay_phase_deadline_truncated_marker(4, 16, 0xc37c_64a8_b0e1_bd0b, 3, outcome).as_deref(),
        Some(
            "[imb-replay-only] phase-census deadline-truncated leaf=4/16 \
             box_fingerprint=fnv1a64:c37c64a8b0e1bd0b emitted=2 retained=3 authority=false"
        )
    );
}

#[test]
fn replay_phase_census_expired_deadline_fails_neutral() {
    let graph = relu_valley_graph();
    let expired = Instant::now()
        .checked_sub(Duration::from_millis(1))
        .expect("system uptime exceeds one millisecond");
    let result = build_replay_phase_census(
        &graph,
        &box1(-1.0, 1.0),
        &HashMap::new(),
        expired,
        MAX_REPLAY_ONLY_PHASE_CENSUS_ENTRIES,
    );
    assert_eq!(result.unwrap_err(), "deadline-before-scan");
}

#[test]
fn replay_leaf_telemetry_reports_standard_route_and_bound() {
    let graph = linear_graph(1.0, 2.0);
    let leaf = box1(-1.0, 0.0);
    let mut timings = ReplayStageTimings::default();
    let evaluation = evaluate_original_objective_leaf(
        &graph,
        &leaf,
        &arr2(&[[1.0]]),
        &[0.5],
        &[1],
        None,
        future_deadline(),
        false,
        Some(&mut timings),
    );
    assert_eq!(evaluation.route, ReplayLeafRoute::StandardFallback);
    assert_eq!(evaluation.f64_attempt, ReplayF64Attempt::GraphUnsupported);
    assert!(evaluation.lower.is_some_and(|lower| lower > 0.5));
    assert_eq!(evaluation.lower, evaluation.standard_lower);
    assert_eq!(timings.f64, None);
    assert!(timings.standard.is_some());
}

#[test]
fn replay_only_cannot_combine_selected_leaf_into_a_verdict() {
    ny_test_utils::env::with_serialized_env_vars(
        &[
            ("NY_IMB", "1"),
            ("NY_IMB_WIRE", "1"),
            ("NY_IMB_REPLAY_ONLY", "1"),
            ("NY_IMB_REPLAY_ONLY_LEAF", "0"),
            ("NY_IMB_REGION_K", "2"),
            ("NY_IMB_OBJ", "0"),
            ("NY_IMB_BUDGET_S", "10"),
        ],
        || {
            crate::imb::reset_early_attempted();
            REPLAY_ONLY_EVALUATIONS.store(0, AtomicOrdering::Relaxed);
            let graph = constant_convtranspose_graph();
            let input = BoundedTensor::new(
                ArrayD::from_shape_vec(IxDyn(&[1, 1, 1]), vec![-1.0]).expect("lower"),
                ArrayD::from_shape_vec(IxDyn(&[1, 1, 1]), vec![1.0]).expect("upper"),
            )
            .expect("valid input");
            let objectives = vec![vec![1.0, 0.0, 0.0, 0.0]];
            let thresholds = vec![1.0];
            let baseline = vec![(-10.0, 10.0)];

            // The early grouped entry owns the admitted diagnostic.
            let grouped = super::imb_multi_objective_floors(
                &graph,
                &input,
                &objectives,
                &thresholds,
                &[1],
                None,
                &HashMap::new(),
                Some(future_deadline()),
            );
            assert_eq!(grouped, vec![(f32::NEG_INFINITY, f32::INFINITY)]);
            assert!(!disjunctive_domain_verified(&grouped, &thresholds, &[1]));
            assert_eq!(REPLAY_ONLY_EVALUATIONS.load(AtomicOrdering::Relaxed), 1);

            // The later root injection must neither evaluate again nor combine
            // the already-observed positive diagnostic lower into authority.
            let unchanged = super::tighten_root_objective_bounds_imb(
                &graph,
                &input,
                &objectives,
                &thresholds,
                None,
                &HashMap::new(),
                None,
                &baseline,
                Some(future_deadline()),
            );
            assert_eq!(unchanged, baseline);
            assert_eq!(REPLAY_ONLY_EVALUATIONS.load(AtomicOrdering::Relaxed), 1);
            crate::imb::reset_early_attempted();
        },
    );
}

#[test]
fn replay_only_attempt_guard_resets_per_verification() {
    crate::imb::reset_early_attempted();
    assert!(!crate::imb::replay_only_attempted());
    assert!(crate::imb::begin_replay_only_attempt());
    assert!(crate::imb::replay_only_attempted());
    assert!(!crate::imb::begin_replay_only_attempt());

    crate::imb::reset_early_attempted();
    assert!(!crate::imb::replay_only_attempted());
    assert!(crate::imb::begin_replay_only_attempt());
    crate::imb::reset_early_attempted();
}

#[test]
fn refused_replay_only_admission_does_not_suppress_later_entry() {
    ny_test_utils::env::with_env_edits(|env| {
        env.set("NY_IMB_REPLAY_ONLY", "1");
        env.set("NY_IMB_REPLAY_ONLY_LEAF", "999");
        env.set("NY_IMB_REGION_K", "2");
        env.set("NY_IMB_OBJ", "0");
        env.set("NY_IMB_BUDGET_S", "10");
        crate::imb::reset_early_attempted();
        REPLAY_ONLY_EVALUATIONS.store(0, AtomicOrdering::Relaxed);

        let graph = constant_convtranspose_graph();
        let input = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[1, 1, 1]), vec![-1.0]).expect("lower"),
            ArrayD::from_shape_vec(IxDyn(&[1, 1, 1]), vec![1.0]).expect("upper"),
        )
        .expect("valid input");
        let objectives = vec![vec![1.0, 0.0, 0.0, 0.0]];
        let thresholds = vec![1.0];

        assert!(maybe_run_replay_only_diagnostic(
            &graph,
            &input,
            &objectives,
            &thresholds,
            None,
            &HashMap::new(),
            Some(future_deadline()),
        ));
        assert!(!crate::imb::replay_only_attempted());
        assert_eq!(REPLAY_ONLY_EVALUATIONS.load(AtomicOrdering::Relaxed), 0);

        env.set("NY_IMB_REPLAY_ONLY_LEAF", "0");
        assert!(maybe_run_replay_only_diagnostic(
            &graph,
            &input,
            &objectives,
            &thresholds,
            None,
            &HashMap::new(),
            Some(future_deadline()),
        ));
        assert!(crate::imb::replay_only_attempted());
        assert_eq!(REPLAY_ONLY_EVALUATIONS.load(AtomicOrdering::Relaxed), 1);
        crate::imb::reset_early_attempted();
    });
}

#[test]
fn one_ulp_box_is_unsplittable() {
    let lower = 1.0_f32;
    let upper = f32::from_bits(lower.to_bits() + 1);
    let root = box1(lower, upper);
    assert!(split_box(&root, 0).is_none());
}

#[test]
fn directed_f64_downcast_never_overclaims() {
    for lower in [1.0_f64, 1.0 + 7e-6, -5e-7, f64::from(f32::MAX)] {
        let cast = directed_f64_lower_to_f32(lower).expect("finite directed cast");
        assert!(f64::from(cast) <= lower);
    }
    assert_eq!(directed_f64_lower_to_f32(f64::NAN), None);
    assert_eq!(directed_f64_lower_to_f32(f64::INFINITY), None);

    // Explicit one-ULP boundary: a certified f64 lower between adjacent f32
    // values rounds to the upper carrier under RN, then the directed step must
    // return the lower carrier. It therefore cannot manufacture a strict
    // `> 1.0` verdict at an f32 threshold of exactly 1.0.
    let one = 1.0_f32;
    let next = f32::from_bits(one.to_bits() + 1);
    let between = f64::from(one) + (f64::from(next) - f64::from(one)) * 0.75;
    let cast = directed_f64_lower_to_f32(between).expect("finite one-ULP cast");
    assert_eq!(cast, one);
    assert!(cast <= one);
}

#[test]
fn positive_affine_partition_gets_full_certificate() {
    // y=x+2 over [-1,1], so the true lower is 1 and clears 0.5.
    let graph = linear_graph(1.0, 2.0);
    let root = box1(-1.0, 1.0);
    let leaves = vec![box1(-1.0, 0.0), box1(0.0, 1.0)];
    let cert = independently_recheck_original_objective(
        &graph,
        &root,
        &leaves,
        &[1.0],
        0.5,
        None,
        future_deadline(),
    )
    .expect("full original-objective replay should certify");
    assert!(cert.lower > 0.5);
}

#[test]
fn batched_original_objective_replay_certifies_distinct_exact_covers_atomically() {
    // y=[x+2, -x+2] over [-1,1]. Both output coordinates have true lower 1,
    // and each objective is independently replayed over a different exact
    // binary cover. The batched transaction may evaluate extra row/leaf pairs,
    // but authority consumes only row i over cover i.
    let graph = two_output_affine_graph();
    let root = box1(-1.0, 1.0);
    let cover_a = vec![box1(-1.0, 0.0), box1(0.0, 1.0)];
    let cover_b = vec![box1(-1.0, -0.5), box1(-0.5, 1.0)];
    let partitions: Vec<&[BoundedTensor]> = vec![&cover_a, &cover_b];
    let objective_a = [1.0_f32, 0.0];
    let objective_b = [0.0_f32, 1.0];
    let objectives: Vec<&[f32]> = vec![&objective_a, &objective_b];
    let certificates = independently_recheck_original_objectives_batched(
        &graph,
        &root,
        &partitions,
        &objectives,
        &[0.5, 0.5],
        0,
        None,
        future_deadline(),
    )
    .expect("both exact covers should certify in one batch");
    assert_eq!(certificates.len(), 2);
    assert!(certificates
        .iter()
        .all(|certificate| certificate.lower > 0.5));
}

#[test]
fn batched_original_objective_replay_honors_explicit_caller_engine() {
    let graph = two_output_affine_graph();
    let root = box1(-1.0, 1.0);
    let cover = vec![box1(-1.0, 0.0), box1(0.0, 1.0)];
    let objective = [1.0_f32, 0.0];
    let engine = CountingGemmEngine::new();

    let certificates = independently_recheck_original_objectives_batched(
        &graph,
        &root,
        &[&cover],
        &[&objective],
        &[0.5],
        0,
        Some(&engine),
        future_deadline(),
    )
    .expect("explicit caller engine replay should certify");

    assert_eq!(certificates.len(), 1);
    assert!(certificates[0].lower > 0.5);
    assert!(
        engine.gemm_calls() > 0,
        "the explicit caller must retain precedence and reach batched propagation"
    );
}

#[test]
fn signed_quotient_replay_certifies_opposites_with_the_unchanged_deadline() {
    // y=x+2 over [-1,1]: lower(y)=1 and lower(-y)=-3. The two objectives
    // are one signed equivalence class, so one propagated row's lower/upper
    // channels must certify both independently covered clauses.
    let graph = linear_graph(1.0, 2.0);
    let root = box1(-1.0, 1.0);
    let cover_a = vec![box1(-1.0, 0.0), box1(0.0, 1.0)];
    let cover_b = vec![box1(-1.0, -0.5), box1(-0.5, 1.0)];
    let positive = [1.0_f32];
    let negative = [-1.0_f32];
    let deadline = future_deadline();
    let certificates = independently_recheck_original_objectives_batched(
        &graph,
        &root,
        &[&cover_a, &cover_b],
        &[&positive, &negative],
        &[0.5, -3.5],
        0,
        None,
        deadline,
    )
    .expect("one representative lower/upper pair should certify both objectives");

    assert_eq!(certificates.len(), 2);
    assert!(certificates[0].lower > 0.5);
    assert!(certificates[1].lower > -3.5);
    assert!(
        certificates
            .iter()
            .all(|certificate| certificate.valid_until == deadline),
        "spec compaction must not shorten, refresh, or replace the authority deadline"
    );
}

#[test]
fn signed_quotient_replay_remains_atomic_when_the_negated_upper_does_not_clear() {
    let graph = linear_graph(1.0, 2.0);
    let root = box1(-1.0, 1.0);
    let cover = vec![box1(-1.0, 0.0), box1(0.0, 1.0)];
    let positive = [1.0_f32];
    let negative = [-1.0_f32];

    assert!(
        independently_recheck_original_objectives_batched(
            &graph,
            &root,
            &[&cover, &cover],
            &[&positive, &negative],
            // lower(y)=1 clears, but lower(-y)=-3 does not clear -2.5.
            &[0.5, -2.5],
            0,
            None,
            future_deadline(),
        )
        .is_none(),
        "one failed signed projection must withhold the complete certificate vector"
    );
}

#[test]
fn batched_original_objective_replay_rejects_all_on_one_uncleared_clause() {
    let graph = two_output_affine_graph();
    let root = box1(-1.0, 1.0);
    let cover = vec![box1(-1.0, 0.0), box1(0.0, 1.0)];
    let partitions: Vec<&[BoundedTensor]> = vec![&cover, &cover];
    let objective_a = [1.0_f32, 0.0];
    let objective_b = [0.0_f32, 1.0];
    let objectives: Vec<&[f32]> = vec![&objective_a, &objective_b];
    assert!(
        independently_recheck_original_objectives_batched(
            &graph,
            &root,
            &partitions,
            &objectives,
            &[0.5, 1.5],
            0,
            None,
            future_deadline(),
        )
        .is_none(),
        "one uncleared cover/objective pair must withhold every certificate"
    );
}

#[test]
fn batched_original_objective_replay_rejects_invalid_cover_before_bounding() {
    let graph = two_output_affine_graph();
    let root = box1(-1.0, 1.0);
    let gapped = vec![box1(-1.0, -0.25), box1(0.0, 1.0)];
    let objective = [1.0_f32, 0.0];
    assert!(independently_recheck_original_objectives_batched(
        &graph,
        &root,
        &[&gapped],
        &[&objective],
        &[0.5],
        0,
        None,
        future_deadline(),
    )
    .is_none());
}

#[test]
fn batched_original_objective_replay_rejects_expired_deadline() {
    let graph = two_output_affine_graph();
    let root = box1(-1.0, 1.0);
    let cover = vec![box1(-1.0, 0.0), box1(0.0, 1.0)];
    let objective = [1.0_f32, 0.0];
    let expired = Instant::now()
        .checked_sub(Duration::from_secs(1))
        .expect("representable expired instant");
    assert!(independently_recheck_original_objectives_batched(
        &graph,
        &root,
        &[&cover],
        &[&objective],
        &[0.5],
        0,
        None,
        expired,
    )
    .is_none());
}

#[test]
fn batched_original_objective_replay_rejects_nonfinite_inputs() {
    let graph = two_output_affine_graph();
    let root = box1(-1.0, 1.0);
    let cover = vec![box1(-1.0, 0.0), box1(0.0, 1.0)];
    let finite_objective = [1.0_f32, 0.0];
    let nan_objective = [f32::NAN, 0.0];

    assert!(independently_recheck_original_objectives_batched(
        &graph,
        &root,
        &[&cover],
        &[&nan_objective],
        &[0.5],
        0,
        None,
        future_deadline(),
    )
    .is_none());
    assert!(independently_recheck_original_objectives_batched(
        &graph,
        &root,
        &[&cover],
        &[&finite_objective],
        &[f32::INFINITY],
        0,
        None,
        future_deadline(),
    )
    .is_none());
}

#[test]
fn weak_ay_proposal_and_batched_replay_authorities_install_as_one_transaction() {
    let graph = two_output_affine_graph();
    let root = box1(-1.0, 1.0);
    let deadline = future_deadline();
    let cover_a = vec![box1(-1.0, 0.0), box1(0.0, 1.0)];
    let cover_b = vec![box1(-1.0, -0.5), box1(-0.5, 1.0)];
    let mut candidates = vec![
        replay_candidate(
            0,
            0.5,
            cover_a,
            deadline,
            Some(FullObjectiveCertificate {
                lower: 0.75,
                valid_until: deadline,
            }),
        ),
        replay_candidate(1, 0.5, cover_b, deadline, None),
    ];
    // The first row's decomposed proposal is telemetry only. Its exact AY
    // certificate clears even though that proposal floor does not.
    candidates[0].imb_floor = 0.25;
    let objectives = vec![vec![1.0_f32, 0.0], vec![0.0_f32, 1.0]];
    let lowers = certify_candidates_with_batched_replay(
        &graph,
        &root,
        &mut candidates,
        &objectives,
        &[0.5, 0.5],
        None,
    )
    .expect("AY-certified row and replayed row should commit atomically");

    assert_eq!(lowers.len(), 2);
    assert_eq!(lowers[0], 0.75);
    assert!(lowers[1] > 0.5);
    assert!(candidates
        .iter()
        .all(|candidate| candidate.full_certificate.is_some()));
}

#[test]
fn uncertified_batched_row_still_requires_a_clearing_proposal() {
    let graph = two_output_affine_graph();
    let root = box1(-1.0, 1.0);
    let deadline = future_deadline();
    let cover = vec![box1(-1.0, 0.0), box1(0.0, 1.0)];
    let mut candidates = vec![replay_candidate(0, 0.5, cover, deadline, None)];
    candidates[0].imb_floor = 0.25;
    let objectives = vec![vec![1.0_f32, 0.0]];

    assert!(certify_candidates_with_batched_replay(
        &graph,
        &root,
        &mut candidates,
        &objectives,
        &[0.5],
        None,
    )
    .is_none());
    assert!(candidates[0].full_certificate.is_none());
}

#[test]
fn failed_batched_row_clears_quarantined_ay_authority() {
    let graph = two_output_affine_graph();
    let root = box1(-1.0, 1.0);
    let deadline = future_deadline();
    let cover = vec![box1(-1.0, 0.0), box1(0.0, 1.0)];
    let mut candidates = vec![
        replay_candidate(
            0,
            0.5,
            cover.clone(),
            deadline,
            Some(FullObjectiveCertificate {
                lower: 0.75,
                valid_until: deadline,
            }),
        ),
        // The decomposed proposal claims clearance, but the original second
        // objective has true lower 1 and cannot clear 1.5.
        replay_candidate(1, 1.5, cover, deadline, None),
    ];
    let objectives = vec![vec![1.0_f32, 0.0], vec![0.0_f32, 1.0]];
    assert!(certify_candidates_with_batched_replay(
        &graph,
        &root,
        &mut candidates,
        &objectives,
        &[0.5, 1.5],
        None,
    )
    .is_none());
    assert!(
        candidates
            .iter()
            .all(|candidate| candidate.full_certificate.is_none()),
        "one replay miss must leave no AY or replay certificate installed"
    );
}

#[test]
fn mismatched_preexisting_certificate_is_quarantined_not_reused() {
    let graph = two_output_affine_graph();
    let root = box1(-1.0, 1.0);
    let deadline = future_deadline();
    let mismatched_deadline = deadline + Duration::from_millis(1);
    let cover = vec![box1(-1.0, 0.0), box1(0.0, 1.0)];
    let mut candidates = vec![replay_candidate(
        0,
        0.5,
        cover,
        deadline,
        Some(FullObjectiveCertificate {
            lower: 0.75,
            valid_until: mismatched_deadline,
        }),
    )];
    let objectives = vec![vec![1.0_f32, 0.0]];
    assert!(certify_candidates_with_batched_replay(
        &graph,
        &root,
        &mut candidates,
        &objectives,
        &[0.5],
        None,
    )
    .is_none());
    assert!(candidates[0].full_certificate.is_none());
}

#[test]
fn standard_no_imb_recheck_api_certifies_affine_bound() {
    let graph = linear_graph(1.0, 2.0);
    let root = box1(-1.0, 1.0);
    let lower = standard_no_imb_objective_lower(&graph, &root, &arr2(&[[1.0]]), None, None)
        .expect("standard no-IMB CROWN/IBP lower");
    assert!(lower > 0.5);
}

#[test]
fn standard_no_imb_recheck_preserves_outer_recursion_guard() {
    let graph = linear_graph(1.0, 2.0);
    let root = box1(-1.0, 1.0);
    let _scope = crate::imb::scope();
    assert!(crate::imb::in_progress());
    let lower = standard_no_imb_objective_lower(&graph, &root, &arr2(&[[1.0]]), None, None)
        .expect("low-level no-IMB replay");
    assert!(lower > 0.5);
    assert!(
        crate::imb::in_progress(),
        "the low-level replay must not enter/drop a nested IMB scope"
    );
}

#[test]
fn expired_full_recheck_deadline_fails_closed() {
    let graph = linear_graph(1.0, 2.0);
    let root = box1(-1.0, 1.0);
    let expired = Instant::now()
        .checked_sub(Duration::from_millis(1))
        .expect("system uptime exceeds one millisecond");
    assert!(independently_recheck_original_objective(
        &graph,
        &root,
        std::slice::from_ref(&root),
        &[1.0],
        0.5,
        None,
        expired,
    )
    .is_none());
}

#[test]
fn grouped_verdict_rejects_any_unrechecked_clause() {
    let thresholds = [0.0, 0.0];
    let clause_sizes = [1, 1];
    let one_clause_only = [(0.25, f32::INFINITY), (f32::NEG_INFINITY, f32::INFINITY)];
    assert!(!disjunctive_domain_verified(
        &one_clause_only,
        &thresholds,
        &clause_sizes,
    ));
    let both_rechecked = [(0.25, f32::INFINITY), (0.5, f32::INFINITY)];
    assert!(disjunctive_domain_verified(
        &both_rechecked,
        &thresholds,
        &clause_sizes,
    ));
}

#[test]
fn tolerated_negative_residual_is_rejected_by_full_recheck() {
    // Historical sampled p/q logic admitted -5e-7 against tol=-1e-6 and could
    // propose floor 0.  The original function is the constant -5e-7, which
    // does not strictly clear -2.5e-7.
    let graph = linear_graph(0.0, -5e-7);
    let root = box1(-1.0, 1.0);
    let cert = independently_recheck_original_objective(
        &graph,
        &root,
        std::slice::from_ref(&root),
        &[1.0],
        -2.5e-7,
        None,
        future_deadline(),
    );
    assert!(cert.is_none());
}

#[test]
fn corner_sampling_miss_cannot_escape_to_authority() {
    let graph = relu_valley_graph();
    let root = box1(0.0, 1.0);

    // A corners-only diagnostic would see +0.25 and accept a claimed positive
    // floor; the true midpoint is -0.25.  Full-network CROWN must reject.
    let mut proposal = candidate(None);
    proposal.imb_floor = 0.1;
    proposal.threshold = 0.0;
    proposal.terminal_boxes = vec![root.clone()];
    proposal.full_certificate = independently_recheck_original_objective(
        &graph,
        &root,
        &proposal.terminal_boxes,
        &[1.0],
        proposal.threshold,
        None,
        future_deadline(),
    );
    assert_eq!(proposal.full_certificate.map(|c| c.lower), None);
    assert_eq!(authoritative_candidate_lower(&proposal), None);
}
