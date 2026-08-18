// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! CPU oracles for clause-aware DomainList state and sealed lease authority.

use super::*;

use crate::batched_domain::ConstraintTuple;
use ndarray::{ArrayD, IxDyn};

type Box2 = ([f32; 2], [f32; 2]);

const GROUPED_OBJECTIVE_A: &[u8] = b"canonical grouped objective A";

fn grouped_layout() -> GroupedDisjunctiveLayout {
    grouped_layout_for(GROUPED_OBJECTIVE_A)
}

fn grouped_layout_for(canonical_objective: &[u8]) -> GroupedDisjunctiveLayout {
    GroupedDisjunctiveLayout::new(vec![0.0; 4], vec![2, 2], canonical_objective).unwrap()
}

fn grouped_config(traversal: TreeTraversal, max_queue_size: usize) -> DomainListConfig {
    DomainListConfig {
        traversal,
        layer_names: Vec::new(),
        layer_shapes: HashMap::new(),
        input_shape: vec![2],
        initial_capacity: 8,
        max_queue_size,
    }
}

fn processed(
    depths: &[usize],
    boxes: &[Box2],
    histories: &[Vec<ConstraintTuple>],
    keep_mask: Vec<bool>,
) -> ProcessedDomains {
    assert_eq!(depths.len(), boxes.len());
    assert_eq!(depths.len(), histories.len());
    let batch = depths.len();
    ProcessedDomains {
        layer_lowers: HashMap::new(),
        layer_uppers: HashMap::new(),
        input_lowers: ArrayD::from_shape_vec(
            IxDyn(&[batch, 2]),
            boxes
                .iter()
                .flat_map(|(lower, _upper)| lower.iter().copied())
                .collect(),
        )
        .unwrap(),
        input_uppers: ArrayD::from_shape_vec(
            IxDyn(&[batch, 2]),
            boxes
                .iter()
                .flat_map(|(_lower, upper)| upper.iter().copied())
                .collect(),
        )
        .unwrap(),
        // Deliberately stale: sealed grouped append replaces scalar summaries
        // from clause-correct rows.
        global_lbs: vec![1234.0; batch],
        global_ubs: vec![5678.0; batch],
        metadata: depths
            .iter()
            .zip(histories)
            .map(|(&depth, history)| {
                DomainMetadata::new(1234.0, 5678.0, depth, history.clone(), None, None).unwrap()
            })
            .collect(),
        keep_mask,
    }
}

fn single_processed(depth: usize, bounds: Box2, history: Vec<ConstraintTuple>) -> ProcessedDomains {
    processed(&[depth], &[bounds], &[history], vec![true])
}

fn row_bounds(rows: &[[f32; 4]]) -> PackedGroupedBounds {
    let lower_flat: Vec<f32> = rows.iter().flatten().copied().collect();
    let upper_flat: Vec<f32> = lower_flat.iter().map(|lower| lower + 0.25).collect();
    PackedGroupedBounds::new(
        ArrayD::from_shape_vec(IxDyn(&[rows.len(), 4]), lower_flat).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[rows.len(), 4]), upper_flat).unwrap(),
    )
}

fn unresolved_rows() -> [[f32; 4]; 1] {
    [[-1.0, -0.5, -1.0, -0.5]]
}

fn verified_rows() -> [[f32; 4]; 1] {
    [[1.0, -1.0, 1.0, -1.0]]
}

enum TestChildDisposition {
    Queued(PackedGroupedBounds),
    Verified(PackedGroupedBounds),
    CertifiedEmpty,
}

struct TestChild {
    processed: ProcessedDomains,
    disposition: TestChildDisposition,
}

fn queued_child(
    depth: usize,
    bounds: Box2,
    history: Vec<ConstraintTuple>,
    rows: [[f32; 4]; 1],
) -> TestChild {
    TestChild {
        processed: single_processed(depth, bounds, history),
        disposition: TestChildDisposition::Queued(row_bounds(&rows)),
    }
}

fn verified_child(depth: usize, bounds: Box2, history: Vec<ConstraintTuple>) -> TestChild {
    TestChild {
        processed: single_processed(depth, bounds, history),
        disposition: TestChildDisposition::Verified(row_bounds(&verified_rows())),
    }
}

fn certified_empty_child(depth: usize, bounds: Box2, history: Vec<ConstraintTuple>) -> TestChild {
    TestChild {
        processed: single_processed(depth, bounds, history),
        disposition: TestChildDisposition::CertifiedEmpty,
    }
}

fn evaluate_child(
    token: GroupedChildEvaluationToken,
    evaluator_layout: &GroupedDisjunctiveLayout,
    child: TestChild,
) -> ny_core::Result<EvaluatedGroupedChild> {
    match child.disposition {
        TestChildDisposition::Queued(bounds) => {
            evaluate_grouped_queued_for_test(token, evaluator_layout, child.processed, bounds)
        }
        TestChildDisposition::Verified(bounds) => {
            evaluate_grouped_verified_for_test(token, evaluator_layout, child.processed, bounds)
        }
        TestChildDisposition::CertifiedEmpty => {
            evaluate_grouped_empty_for_test(token, evaluator_layout, child.processed)
        }
    }
}

fn input_outcome(
    picked: &PickedGroupedDomains,
    parent_index: usize,
    first: TestChild,
    second: TestChild,
) -> ny_core::Result<GroupedParentOutcome> {
    let [first_token, second_token] =
        picked.mint_input_bisection_tokens(parent_index, &first.processed, &second.processed)?;
    Ok(GroupedParentOutcome::input_bisection(
        evaluate_child(first_token, picked.layout(), first)?,
        evaluate_child(second_token, picked.layout(), second)?,
    ))
}

fn phase_outcome(
    picked: &PickedGroupedDomains,
    parent_index: usize,
    first: TestChild,
    second: TestChild,
) -> ny_core::Result<GroupedParentOutcome> {
    let [first_token, second_token] =
        picked.mint_phase_split_tokens(parent_index, &first.processed, &second.processed)?;
    Ok(GroupedParentOutcome::phase_split(
        evaluate_child(first_token, picked.layout(), first)?,
        evaluate_child(second_token, picked.layout(), second)?,
    ))
}

fn add_grouped_roots_for_test(
    list: &mut DomainList,
    processed: ProcessedDomains,
    row_bounds: PackedGroupedBounds,
    dispositions: Vec<TestGroupedRootDisposition>,
) -> ny_core::Result<()> {
    assert!(processed.keep_mask.iter().all(|keep| *keep));
    let token = list.mint_grouped_root_evaluation_token(&processed)?;
    let evaluator_layout = list
        .grouped
        .as_ref()
        .expect("test grouped queue")
        .layout
        .clone();
    let evaluated = evaluate_grouped_roots_for_test(
        list,
        token,
        &evaluator_layout,
        processed,
        row_bounds,
        dispositions,
    )?;
    list.accept_grouped_root_evaluation(evaluated)
}

fn picked_depths(picked: &PickedGroupedDomains) -> Vec<usize> {
    picked
        .domains()
        .metadata
        .iter()
        .map(DomainMetadata::depth)
        .collect()
}

fn one_parent(
    depth: usize,
    bounds: Box2,
    history: Vec<ConstraintTuple>,
    rows: [[f32; 4]; 1],
) -> (DomainList, PickedGroupedDomains) {
    let mut list = DomainList::new_grouped(
        grouped_config(TreeTraversal::BreadthFirst, 0),
        grouped_layout(),
    )
    .unwrap();
    add_grouped_roots_for_test(
        &mut list,
        single_processed(depth, bounds, history),
        row_bounds(&rows),
        vec![TestGroupedRootDisposition::Queued],
    )
    .unwrap();
    let picked = list.pick_out_grouped(1).unwrap();
    (list, picked)
}

fn resolution(
    picked: &PickedGroupedDomains,
    index: usize,
    outcome: GroupedParentOutcome,
) -> GroupedParentResolution {
    GroupedParentResolution::new(picked.domain_ids()[index], outcome)
}

#[test]
fn grouped_uninitialized_empty_queue_is_not_a_proof() {
    let list = DomainList::new_grouped(
        grouped_config(TreeTraversal::BreadthFirst, 0),
        grouped_layout(),
    )
    .unwrap();
    assert_eq!(
        list.grouped_queue_status().unwrap(),
        GroupedQueueStatus::ExhaustedUnknown
    );
}

#[test]
fn grouped_queue_rejects_uncensused_byte_cap_and_upper_priority() {
    let mut list = DomainList::new_grouped(
        grouped_config(TreeTraversal::BreadthFirst, 0),
        grouped_layout(),
    )
    .unwrap();

    let byte_error = list.configure_queue_eviction(1024, false).unwrap_err();
    assert!(
        byte_error.to_string().contains("row-sidecar bytes"),
        "grouped byte-cap refusal must identify the uncensused sidecar: {byte_error}"
    );

    let upper_error = list.configure_queue_eviction(0, true).unwrap_err();
    assert!(
        upper_error
            .to_string()
            .contains("only supports lower-margin verification"),
        "grouped upper-priority refusal must be explicit: {upper_error}"
    );
}

#[test]
fn grouped_root_token_rejects_non_neutral_keep_mask() {
    let list = DomainList::new_grouped(
        grouped_config(TreeTraversal::BreadthFirst, 0),
        grouped_layout(),
    )
    .unwrap();
    let mut root = single_processed(1, ([0.0, 0.0], [1.0, 1.0]), Vec::new());
    root.keep_mask[0] = false;
    assert!(list.mint_grouped_root_evaluation_token(&root).is_err());
    assert_eq!(
        list.grouped_queue_status().unwrap(),
        GroupedQueueStatus::ExhaustedUnknown
    );
}

#[test]
fn grouped_root_evaluator_rejects_swapped_geometry_and_rows() {
    let list = DomainList::new_grouped(
        grouped_config(TreeTraversal::BreadthFirst, 0),
        grouped_layout(),
    )
    .unwrap();
    let boxes = [([0.0, 0.0], [1.0, 1.0]), ([2.0, 0.0], [3.0, 1.0])];
    let roots = processed(
        &[1, 2],
        &boxes,
        &[Vec::new(), vec![("history".to_string(), 1, true, None)]],
        vec![true; 2],
    );
    let token = list.mint_grouped_root_evaluation_token(&roots).unwrap();
    let swapped = processed(
        &[2, 1],
        &[boxes[1], boxes[0]],
        &[vec![("history".to_string(), 1, true, None)], Vec::new()],
        vec![true; 2],
    );
    assert!(evaluate_grouped_roots_for_test(
        &list,
        token,
        &grouped_layout(),
        swapped,
        row_bounds(&[verified_rows()[0], verified_rows()[0]]),
        vec![TestGroupedRootDisposition::Verified; 2],
    )
    .is_err());
}

#[test]
fn grouped_root_evaluator_rejects_identical_layout_cross_spec_rows() {
    let list = DomainList::new_grouped(
        grouped_config(TreeTraversal::BreadthFirst, 0),
        grouped_layout(),
    )
    .unwrap();
    let root = single_processed(1, ([0.0, 0.0], [1.0, 1.0]), Vec::new());
    let token = list.mint_grouped_root_evaluation_token(&root).unwrap();
    let other_spec = grouped_layout_for(b"canonical grouped objective root B");
    assert_eq!(other_spec.thresholds(), grouped_layout().thresholds());
    assert_eq!(other_spec.clause_sizes(), grouped_layout().clause_sizes());
    assert!(evaluate_grouped_roots_for_test(
        &list,
        token,
        &other_spec,
        root,
        row_bounds(&verified_rows()),
        vec![TestGroupedRootDisposition::Verified],
    )
    .is_err());
}

#[test]
fn grouped_root_entry_rejects_cross_queue_swap() {
    let first = DomainList::new_grouped(
        grouped_config(TreeTraversal::BreadthFirst, 0),
        grouped_layout(),
    )
    .unwrap();
    let mut second = DomainList::new_grouped(
        grouped_config(TreeTraversal::BreadthFirst, 0),
        grouped_layout(),
    )
    .unwrap();
    let root = single_processed(1, ([0.0, 0.0], [1.0, 1.0]), Vec::new());
    let token = first.mint_grouped_root_evaluation_token(&root).unwrap();
    let evaluated = evaluate_grouped_roots_for_test(
        &first,
        token,
        &grouped_layout(),
        root,
        row_bounds(&verified_rows()),
        vec![TestGroupedRootDisposition::Verified],
    )
    .unwrap();
    assert!(second.accept_grouped_root_evaluation(evaluated).is_err());
    assert!(first.is_empty());
    assert!(second.is_empty());
}

#[test]
fn grouped_sealed_root_dispositions_control_exhaustion() {
    let mut dropped = DomainList::new_grouped(
        grouped_config(TreeTraversal::BreadthFirst, 0),
        grouped_layout(),
    )
    .unwrap();
    add_grouped_roots_for_test(
        &mut dropped,
        single_processed(1, ([0.0, 0.0], [1.0, 1.0]), Vec::new()),
        row_bounds(&unresolved_rows()),
        vec![TestGroupedRootDisposition::UnresolvedDropped],
    )
    .unwrap();
    assert_eq!(
        dropped.grouped_queue_status().unwrap(),
        GroupedQueueStatus::ExhaustedUnknown
    );

    let mut empty = DomainList::new_grouped(
        grouped_config(TreeTraversal::BreadthFirst, 0),
        grouped_layout(),
    )
    .unwrap();
    add_grouped_roots_for_test(
        &mut empty,
        single_processed(1, ([0.0, 0.0], [1.0, 1.0]), Vec::new()),
        row_bounds(&unresolved_rows()),
        vec![TestGroupedRootDisposition::CertifiedEmpty],
    )
    .unwrap();
    assert_eq!(
        empty.grouped_queue_status().unwrap(),
        GroupedQueueStatus::ExhaustedVerified
    );
}

#[test]
fn grouped_compaction_of_certified_domain_can_finish_verified() {
    let mut list = DomainList::new_grouped(
        grouped_config(TreeTraversal::BreadthFirst, 0),
        grouped_layout(),
    )
    .unwrap();
    add_grouped_roots_for_test(
        &mut list,
        processed(&[1], &[([0.0, 0.0], [1.0, 1.0])], &[Vec::new()], vec![true]),
        row_bounds(&verified_rows()),
        vec![TestGroupedRootDisposition::Verified],
    )
    .unwrap();
    assert!(list.is_empty());
    assert_eq!(
        list.grouped_queue_status().unwrap(),
        GroupedQueueStatus::ExhaustedVerified
    );
}

#[test]
fn grouped_compaction_preserves_rows_metadata_and_layout() {
    let mut list = DomainList::new_grouped(
        grouped_config(TreeTraversal::BreadthFirst, 0),
        grouped_layout(),
    )
    .unwrap();
    let rows = [
        [-1.0, -0.9, -0.8, -0.7],
        [-2.0, -1.9, -1.8, -1.7],
        [-3.0, -2.9, -2.8, -2.7],
        [-4.0, -3.9, -3.8, -3.7],
    ];
    let boxes = [([0.0, 0.0], [1.0, 1.0]); 4];
    add_grouped_roots_for_test(
        &mut list,
        processed(
            &[10, 20, 30, 40],
            &boxes,
            &[Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            vec![true; 4],
        ),
        row_bounds(&rows),
        vec![
            TestGroupedRootDisposition::Queued,
            TestGroupedRootDisposition::UnresolvedDropped,
            TestGroupedRootDisposition::Queued,
            TestGroupedRootDisposition::UnresolvedDropped,
        ],
    )
    .unwrap();

    let picked = list.pick_out_grouped(8).unwrap();
    assert_eq!(picked_depths(&picked), vec![10, 30]);
    assert_eq!(picked.layout().clause_sizes(), &[2, 2]);
    assert_eq!(picked.layout().thresholds(), &[0.0; 4]);
    assert_eq!(
        picked.row_bounds().row_lowers(),
        &ArrayD::from_shape_vec(
            IxDyn(&[2, 4]),
            rows[0].iter().chain(rows[2].iter()).copied().collect()
        )
        .unwrap()
    );
    assert_eq!(picked.domains().global_lbs, vec![-0.9, -2.9]);
}

#[test]
fn grouped_sort_preserves_row_alignment() {
    let mut list = DomainList::new_grouped(
        grouped_config(TreeTraversal::BreadthFirst, 0),
        grouped_layout(),
    )
    .unwrap();
    let rows = [
        [-0.9, -0.8, -0.7, -0.6],
        [-0.4, -0.3, -0.2, -0.1],
        [-0.6, -0.5, -0.9, -0.4],
    ];
    add_grouped_roots_for_test(
        &mut list,
        processed(
            &[10, 20, 30],
            &[([0.0, 0.0], [1.0, 1.0]); 3],
            &[Vec::new(), Vec::new(), Vec::new()],
            vec![true; 3],
        ),
        row_bounds(&rows),
        vec![TestGroupedRootDisposition::Queued; 3],
    )
    .unwrap();
    list.sort_by_domain_priority(false).unwrap();

    let picked = list.pick_out_grouped(3).unwrap();
    assert_eq!(picked_depths(&picked), vec![10, 30, 20]);
    assert_eq!(picked.domains().global_lbs, vec![-0.8, -0.5, -0.3]);
    let first_rows: Vec<f32> = picked
        .row_bounds()
        .row_lowers()
        .axis_iter(ndarray::Axis(0))
        .map(|row| row[0])
        .collect();
    assert_eq!(first_rows, vec![-0.9, -0.6, -0.4]);
}

#[test]
fn grouped_eviction_preserves_rows_and_forces_unknown() {
    let mut list = DomainList::new_grouped(
        grouped_config(TreeTraversal::BreadthFirst, 2),
        grouped_layout(),
    )
    .unwrap();
    let rows = [
        [-0.9, -0.8, -0.9, -0.8],
        [-0.4, -0.3, -0.4, -0.3],
        [-0.6, -0.5, -0.6, -0.5],
    ];
    add_grouped_roots_for_test(
        &mut list,
        processed(
            &[10, 20, 30],
            &[([0.0, 0.0], [1.0, 1.0]); 3],
            &[Vec::new(), Vec::new(), Vec::new()],
            vec![true; 3],
        ),
        row_bounds(&rows),
        vec![TestGroupedRootDisposition::Queued; 3],
    )
    .unwrap();
    assert_eq!(list.evicted_count(), 1);

    let picked = list.pick_out_grouped(2).unwrap();
    assert_eq!(picked_depths(&picked), vec![10, 30]);
    let resolutions = (0..2)
        .map(|index| resolution(&picked, index, GroupedParentOutcome::UnresolvedDropped))
        .collect();
    list.resolve_grouped_batch(picked, resolutions).unwrap();
    assert_eq!(
        list.grouped_queue_status().unwrap(),
        GroupedQueueStatus::ExhaustedUnknown
    );
}

#[test]
fn grouped_in_flight_and_unresolved_parent_force_nonverified_status() {
    let (mut list, picked) = one_parent(7, ([0.0, 0.0], [1.0, 1.0]), Vec::new(), unresolved_rows());
    assert!(!picked.summaries().unwrap()[0].is_verified());
    assert_eq!(
        list.grouped_queue_status().unwrap(),
        GroupedQueueStatus::Pending
    );
    let resolutions = vec![resolution(
        &picked,
        0,
        GroupedParentOutcome::UnresolvedDropped,
    )];
    list.resolve_grouped_batch(picked, resolutions).unwrap();
    assert_eq!(
        list.grouped_queue_status().unwrap(),
        GroupedQueueStatus::ExhaustedUnknown
    );
}

#[test]
fn grouped_verified_uses_immutable_leased_summary() {
    let (mut list, picked) = one_parent(9, ([0.0, 0.0], [1.0, 1.0]), Vec::new(), verified_rows());
    let resolutions = vec![resolution(&picked, 0, GroupedParentOutcome::Verified)];
    let completion = list.resolve_grouped_batch(picked, resolutions).unwrap();
    assert_eq!(completion.verified, 1);
    assert_eq!(
        list.grouped_queue_status().unwrap(),
        GroupedQueueStatus::ExhaustedVerified
    );
}

#[test]
fn grouped_false_verified_is_rejected_and_lease_stays_pending() {
    let (mut list, picked) = one_parent(1, ([0.0, 0.0], [1.0, 1.0]), Vec::new(), unresolved_rows());
    let resolutions = vec![resolution(&picked, 0, GroupedParentOutcome::Verified)];
    assert!(list.resolve_grouped_batch(picked, resolutions).is_err());
    assert_eq!(
        list.grouped_queue_status().unwrap(),
        GroupedQueueStatus::Pending
    );
}

#[test]
fn grouped_exact_input_cover_keeps_descendants_pending() {
    let parent_box = ([0.0, 0.0], [2.0, 2.0]);
    let (mut list, parent) = one_parent(1, parent_box, Vec::new(), unresolved_rows());
    let outcome = input_outcome(
        &parent,
        0,
        queued_child(2, ([0.0, 0.0], [1.0, 2.0]), Vec::new(), verified_rows()),
        queued_child(2, ([1.0, 0.0], [2.0, 2.0]), Vec::new(), verified_rows()),
    )
    .unwrap();
    let resolutions = vec![resolution(&parent, 0, outcome)];
    list.resolve_grouped_batch(parent, resolutions).unwrap();
    assert_eq!(
        list.grouped_queue_status().unwrap(),
        GroupedQueueStatus::Pending
    );

    let children = list.pick_out_grouped(2).unwrap();
    let resolutions = (0..2)
        .map(|index| resolution(&children, index, GroupedParentOutcome::Verified))
        .collect();
    list.resolve_grouped_batch(children, resolutions).unwrap();
    assert_eq!(
        list.grouped_queue_status().unwrap(),
        GroupedQueueStatus::ExhaustedVerified
    );
}

#[test]
fn grouped_input_cover_rejects_duplicate_children() {
    let (list, picked) = one_parent(1, ([0.0, 0.0], [2.0, 2.0]), Vec::new(), unresolved_rows());
    let left = ([0.0, 0.0], [1.0, 2.0]);
    assert!(input_outcome(
        &picked,
        0,
        verified_child(2, left, Vec::new()),
        verified_child(2, left, Vec::new()),
    )
    .is_err());
    assert_eq!(
        list.grouped_queue_status().unwrap(),
        GroupedQueueStatus::Pending
    );
}

#[test]
fn grouped_input_cover_rejects_gap() {
    let (list, picked) = one_parent(1, ([0.0, 0.0], [2.0, 2.0]), Vec::new(), unresolved_rows());
    assert!(input_outcome(
        &picked,
        0,
        verified_child(2, ([0.0, 0.0], [0.9, 2.0]), Vec::new()),
        verified_child(2, ([1.1, 0.0], [2.0, 2.0]), Vec::new()),
    )
    .is_err());
    assert_eq!(
        list.grouped_queue_status().unwrap(),
        GroupedQueueStatus::Pending
    );
}

#[test]
fn grouped_input_cover_rejects_depth_spoof() {
    let (list, picked) = one_parent(4, ([0.0, 0.0], [2.0, 2.0]), Vec::new(), unresolved_rows());
    assert!(input_outcome(
        &picked,
        0,
        verified_child(5, ([0.0, 0.0], [1.0, 2.0]), Vec::new()),
        verified_child(6, ([1.0, 0.0], [2.0, 2.0]), Vec::new()),
    )
    .is_err());
    assert_eq!(
        list.grouped_queue_status().unwrap(),
        GroupedQueueStatus::Pending
    );
}

#[test]
fn grouped_phase_cover_requires_exact_parent_history_and_opposite_phases() {
    let parent_history = vec![("old".to_string(), 1, true, None)];
    let (list, picked) = one_parent(
        3,
        ([0.0, 0.0], [1.0, 1.0]),
        parent_history.clone(),
        unresolved_rows(),
    );
    let mut first_history = parent_history;
    first_history.push(("relu".to_string(), 7, true, None));
    let mut second_history = vec![("wrong".to_string(), 1, true, None)];
    second_history.push(("relu".to_string(), 7, false, None));
    assert!(phase_outcome(
        &picked,
        0,
        verified_child(4, ([0.0, 0.0], [1.0, 1.0]), first_history),
        verified_child(4, ([0.0, 0.0], [1.0, 1.0]), second_history),
    )
    .is_err());
    assert_eq!(
        list.grouped_queue_status().unwrap(),
        GroupedQueueStatus::Pending
    );
}

#[test]
fn grouped_phase_cover_rejects_same_phase_or_split_point_mismatch() {
    let parent_box = ([0.0, 0.0], [1.0, 1.0]);
    let (same_phase_list, same_phase_pick) =
        one_parent(1, parent_box, Vec::new(), unresolved_rows());
    assert!(phase_outcome(
        &same_phase_pick,
        0,
        verified_child(
            2,
            parent_box,
            vec![("relu".to_string(), 7, true, Some(0.25))],
        ),
        verified_child(
            2,
            parent_box,
            vec![("relu".to_string(), 7, true, Some(0.25))],
        ),
    )
    .is_err());
    assert_eq!(
        same_phase_list.grouped_queue_status().unwrap(),
        GroupedQueueStatus::Pending
    );

    let (point_list, point_pick) = one_parent(1, parent_box, Vec::new(), unresolved_rows());
    assert!(phase_outcome(
        &point_pick,
        0,
        verified_child(
            2,
            parent_box,
            vec![("relu".to_string(), 7, true, Some(0.25))],
        ),
        verified_child(
            2,
            parent_box,
            vec![("relu".to_string(), 7, false, Some(0.5))],
        ),
    )
    .is_err());
    assert_eq!(
        point_list.grouped_queue_status().unwrap(),
        GroupedQueueStatus::Pending
    );
}

#[test]
fn grouped_child_keep_mask_cannot_act_as_outcome_authority() {
    let parent_box = ([0.0, 0.0], [2.0, 2.0]);
    let (list, picked) = one_parent(1, parent_box, Vec::new(), unresolved_rows());
    let mut forged = single_processed(2, ([0.0, 0.0], [1.0, 2.0]), Vec::new());
    forged.keep_mask[0] = false;
    let forged_child = TestChild {
        processed: forged,
        disposition: TestChildDisposition::Verified(row_bounds(&verified_rows())),
    };
    assert!(input_outcome(
        &picked,
        0,
        forged_child,
        verified_child(2, ([1.0, 0.0], [2.0, 2.0]), Vec::new()),
    )
    .is_err());
    assert_eq!(
        list.grouped_queue_status().unwrap(),
        GroupedQueueStatus::Pending
    );
}

#[test]
fn grouped_cross_parent_children_are_rejected() {
    let mut list = DomainList::new_grouped(
        grouped_config(TreeTraversal::BreadthFirst, 0),
        grouped_layout(),
    )
    .unwrap();
    let boxes = [([0.0, 0.0], [2.0, 2.0]), ([10.0, 0.0], [12.0, 2.0])];
    add_grouped_roots_for_test(
        &mut list,
        processed(&[1, 1], &boxes, &[Vec::new(), Vec::new()], vec![true, true]),
        row_bounds(&[unresolved_rows()[0], unresolved_rows()[0]]),
        vec![TestGroupedRootDisposition::Queued; 2],
    )
    .unwrap();
    let picked = list.pick_out_grouped(2).unwrap();
    let first_outcome = input_outcome(
        &picked,
        1,
        verified_child(2, ([10.0, 0.0], [11.0, 2.0]), Vec::new()),
        verified_child(2, ([11.0, 0.0], [12.0, 2.0]), Vec::new()),
    )
    .unwrap();
    let second_outcome = input_outcome(
        &picked,
        0,
        verified_child(2, ([0.0, 0.0], [1.0, 2.0]), Vec::new()),
        verified_child(2, ([1.0, 0.0], [2.0, 2.0]), Vec::new()),
    )
    .unwrap();
    let resolutions = vec![
        resolution(&picked, 0, first_outcome),
        resolution(&picked, 1, second_outcome),
    ];
    assert!(list.resolve_grouped_batch(picked, resolutions).is_err());
}

#[test]
fn grouped_evaluator_rejects_swapped_child_geometry_and_rows() {
    let parent_box = ([0.0, 0.0], [2.0, 2.0]);
    let (list, picked) = one_parent(1, parent_box, Vec::new(), unresolved_rows());
    let left = single_processed(2, ([0.0, 0.0], [1.0, 2.0]), Vec::new());
    let right = single_processed(2, ([1.0, 0.0], [2.0, 2.0]), Vec::new());
    let [left_token, _right_token] = picked
        .mint_input_bisection_tokens(0, &left, &right)
        .unwrap();

    // This is the old false-authority shape: a sound row proof computed for
    // the left child is paired with the right child's geometry. The token was
    // minted for the left child, so sealing fails before a verdict exists.
    assert!(evaluate_grouped_verified_for_test(
        left_token,
        picked.layout(),
        right,
        row_bounds(&verified_rows()),
    )
    .is_err());
    assert_eq!(
        list.grouped_queue_status().unwrap(),
        GroupedQueueStatus::Pending
    );
}

#[test]
fn grouped_resolution_rejects_slot_swapped_opaque_children() {
    let parent_box = ([0.0, 0.0], [2.0, 2.0]);
    let (mut list, picked) = one_parent(1, parent_box, Vec::new(), unresolved_rows());
    let left = verified_child(2, ([0.0, 0.0], [1.0, 2.0]), Vec::new());
    let right = verified_child(2, ([1.0, 0.0], [2.0, 2.0]), Vec::new());
    let [left_token, right_token] = picked
        .mint_input_bisection_tokens(0, &left.processed, &right.processed)
        .unwrap();
    let evaluated_left = evaluate_child(left_token, picked.layout(), left).unwrap();
    let evaluated_right = evaluate_child(right_token, picked.layout(), right).unwrap();
    let outcome = GroupedParentOutcome::input_bisection(evaluated_right, evaluated_left);
    let resolutions = vec![resolution(&picked, 0, outcome)];

    // Geometry alone still forms an exact cover in reversed order. Rejection
    // therefore discriminates the ordered token binding, not the cover check.
    assert!(list.resolve_grouped_batch(picked, resolutions).is_err());
    assert_eq!(
        list.grouped_queue_status().unwrap(),
        GroupedQueueStatus::Pending
    );
}

#[test]
fn grouped_evaluator_rejects_identical_layout_cross_spec_rows() {
    let parent_box = ([0.0, 0.0], [2.0, 2.0]);
    let (list, picked) = one_parent(1, parent_box, Vec::new(), unresolved_rows());
    let left = single_processed(2, ([0.0, 0.0], [1.0, 2.0]), Vec::new());
    let right = single_processed(2, ([1.0, 0.0], [2.0, 2.0]), Vec::new());
    let [left_token, _right_token] = picked
        .mint_input_bisection_tokens(0, &left, &right)
        .unwrap();
    let other_spec = grouped_layout_for(b"canonical grouped objective B");
    assert_eq!(other_spec.thresholds(), picked.layout().thresholds());
    assert_eq!(other_spec.clause_sizes(), picked.layout().clause_sizes());
    assert_ne!(
        other_spec.spec_fingerprint(),
        picked.layout().spec_fingerprint()
    );

    // Shape, clauses, thresholds, and child geometry are identical. Only the
    // objective/spec fingerprint differs.
    assert!(evaluate_grouped_verified_for_test(
        left_token,
        &other_spec,
        left,
        row_bounds(&verified_rows()),
    )
    .is_err());
    assert_eq!(
        list.grouped_queue_status().unwrap(),
        GroupedQueueStatus::Pending
    );
}

#[test]
fn grouped_row_swap_is_rejected_by_payload_bound_lease() {
    let mut list = DomainList::new_grouped(
        grouped_config(TreeTraversal::BreadthFirst, 0),
        grouped_layout(),
    )
    .unwrap();
    add_grouped_roots_for_test(
        &mut list,
        processed(
            &[1, 1],
            &[([0.0, 0.0], [1.0, 1.0]); 2],
            &[Vec::new(), Vec::new()],
            vec![true, true],
        ),
        row_bounds(&[unresolved_rows()[0], verified_rows()[0]]),
        vec![TestGroupedRootDisposition::Queued; 2],
    )
    .unwrap();
    let mut picked = list.pick_out_grouped(2).unwrap();
    let resolutions = vec![
        resolution(&picked, 0, GroupedParentOutcome::UnresolvedDropped),
        resolution(&picked, 1, GroupedParentOutcome::Verified),
    ];
    picked.swap_rows_for_test(0, 1);
    assert!(list.resolve_grouped_batch(picked, resolutions).is_err());
    assert_eq!(
        list.grouped_queue_status().unwrap(),
        GroupedQueueStatus::Pending
    );
}

#[test]
fn grouped_duplicate_domain_id_is_rejected() {
    let mut list = DomainList::new_grouped(
        grouped_config(TreeTraversal::BreadthFirst, 0),
        grouped_layout(),
    )
    .unwrap();
    add_grouped_roots_for_test(
        &mut list,
        processed(
            &[1, 1],
            &[([0.0, 0.0], [1.0, 1.0]); 2],
            &[Vec::new(), Vec::new()],
            vec![true, true],
        ),
        row_bounds(&[verified_rows()[0], verified_rows()[0]]),
        vec![TestGroupedRootDisposition::Queued; 2],
    )
    .unwrap();
    let picked = list.pick_out_grouped(2).unwrap();
    let cloned_id = picked.domain_ids()[0];
    let resolutions = vec![
        GroupedParentResolution::new(cloned_id, GroupedParentOutcome::Verified),
        GroupedParentResolution::new(cloned_id, GroupedParentOutcome::Verified),
    ];
    assert!(list.resolve_grouped_batch(picked, resolutions).is_err());
}

#[test]
fn grouped_cross_queue_lease_is_rejected() {
    let (first, first_pick) =
        one_parent(1, ([0.0, 0.0], [1.0, 1.0]), Vec::new(), unresolved_rows());
    let (mut second, _second_pick) =
        one_parent(1, ([0.0, 0.0], [1.0, 1.0]), Vec::new(), unresolved_rows());
    let resolutions = vec![resolution(
        &first_pick,
        0,
        GroupedParentOutcome::UnresolvedDropped,
    )];
    assert!(second
        .resolve_grouped_batch(first_pick, resolutions)
        .is_err());
    assert_eq!(
        first.grouped_queue_status().unwrap(),
        GroupedQueueStatus::Pending
    );
    assert_eq!(
        second.grouped_queue_status().unwrap(),
        GroupedQueueStatus::Pending
    );
}

#[test]
fn grouped_certified_empty_requires_structural_phase_contradiction() {
    let predicate = ("relu".to_string(), 3, true, None);
    let parent_history = vec![predicate.clone()];
    let parent_box = ([0.0, 0.0], [1.0, 1.0]);
    let (mut list, picked) = one_parent(5, parent_box, parent_history.clone(), unresolved_rows());
    let mut active_history = parent_history.clone();
    active_history.push(predicate);
    let mut inactive_history = parent_history;
    inactive_history.push(("relu".to_string(), 3, false, None));
    let outcome = phase_outcome(
        &picked,
        0,
        verified_child(6, parent_box, active_history),
        certified_empty_child(6, parent_box, inactive_history),
    )
    .unwrap();
    let resolutions = vec![resolution(&picked, 0, outcome)];
    list.resolve_grouped_batch(picked, resolutions).unwrap();
    assert_eq!(
        list.grouped_queue_status().unwrap(),
        GroupedQueueStatus::ExhaustedVerified
    );
}

#[test]
fn grouped_unproven_certified_empty_is_rejected() {
    let parent_box = ([0.0, 0.0], [1.0, 1.0]);
    let (mut list, picked) = one_parent(1, parent_box, Vec::new(), unresolved_rows());
    let active_history = vec![("relu".to_string(), 3, true, None)];
    let inactive_history = vec![("relu".to_string(), 3, false, None)];
    let outcome = phase_outcome(
        &picked,
        0,
        verified_child(2, parent_box, active_history),
        certified_empty_child(2, parent_box, inactive_history),
    )
    .unwrap();
    let resolutions = vec![resolution(&picked, 0, outcome)];
    assert!(list.resolve_grouped_batch(picked, resolutions).is_err());
}

#[test]
fn grouped_layout_mismatch_and_scalar_apis_fail_without_mutating_queue() {
    let mut list = DomainList::new_grouped(
        grouped_config(TreeTraversal::BreadthFirst, 0),
        grouped_layout(),
    )
    .unwrap();
    let payload = single_processed(1, ([0.0, 0.0], [1.0, 1.0]), Vec::new());
    assert!(list.add(payload).is_err());
    assert!(list.is_empty());

    let malformed =
        PackedGroupedBounds::new(ArrayD::zeros(IxDyn(&[1, 3])), ArrayD::zeros(IxDyn(&[1, 3])));
    let root = single_processed(1, ([0.0, 0.0], [1.0, 1.0]), Vec::new());
    let token = list.mint_grouped_root_evaluation_token(&root).unwrap();
    assert!(evaluate_grouped_roots_for_test(
        &list,
        token,
        &grouped_layout(),
        root,
        malformed,
        vec![TestGroupedRootDisposition::Queued],
    )
    .is_err());
    assert!(list.is_empty());
}

#[test]
fn grouped_api_has_no_raw_queue_insertion_or_bounds_reexport() {
    let storage_source = include_str!("../storage.rs");
    let grouped_source = include_str!("../grouped.rs");
    let batched_domain_source = include_str!("../../../batched_domain.rs");
    let crate_source = include_str!("../../../lib.rs");

    assert!(!storage_source.contains("fn add_grouped("));
    assert!(!storage_source.contains("pub(super) fn add_impl("));
    assert!(storage_source.contains("pub(super) fn append_sealed_grouped_queued("));
    assert!(storage_source.contains("entry: SealedGroupedQueueEntry,"));
    assert!(!grouped_source.contains("pub(crate) struct PackedGroupedBounds"));
    assert!(!grouped_source.contains("pub(crate) fn new(row_lowers"));
    assert!(!batched_domain_source.contains("PackedGroupedBounds"));
    assert!(!crate_source.contains("PackedGroupedBounds"));
    assert!(!grouped_source.contains("GroupedChildOutcome"));
}
