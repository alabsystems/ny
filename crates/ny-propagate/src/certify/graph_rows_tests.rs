// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #vit-v3-bankable: the multi-objective (per-ROW) frame of the graph-lane
//! completeness ledger, the graph emptiness slack, and the cover witnesses.
//!
//! Every check counts what it checked; a precondition that does not hold
//! fails instead of passing vacuously.

use super::cover::{CoverLedger, CoverLiteral, CoverRefusal, CoverSite, LeafEvidence, LeafPath};
use super::graph::{
    authorize_graph_instance, graph_emptiness_evidence, graph_lower_evidence,
    reset_graph_cover_capture, take_graph_cover_capture, GraphCoverSlot, GraphCoverVerdict,
    GraphInstanceProperty, GraphObjective, GraphRecordKey, GraphRowsVerdict, RegionLiteral,
    CAPTURE_TEST_LOCK,
};
use super::token::TokenScope;
use super::value::{CertReceipt, Certified};
use crate::beta_crown::branching::{GraphNeuronConstraint, GraphSplitHistory};
use ny_tensor::BoundedTensor;

const NODES: [&str; 3] = ["input_linear", "relu_a", "relu_b"];

/// Two rows (a vit property has nine): `y0 - y1 > 0` and `y0 - y2 > -0.5`,
/// rows no other test uses (the capture is process-global).
fn rows() -> (Vec<Vec<f32>>, Vec<f32>) {
    (
        vec![
            vec![1.0, -1.0, 0.0, 0.0, 0.375],
            vec![1.0, 0.0, -1.0, 0.0, 0.375],
        ],
        vec![0.0, -0.5],
    )
}

fn boxed(lo: &[f32], hi: &[f32]) -> BoundedTensor {
    BoundedTensor::new(
        ndarray::Array1::from(lo.to_vec()).into_dyn(),
        ndarray::Array1::from(hi.to_vec()).into_dyn(),
    )
    .expect("well-formed box")
}

fn split(node: &str, neuron: usize, active: bool) -> GraphNeuronConstraint {
    GraphNeuronConstraint::new(node.to_string(), neuron, active, 1.0).expect("finite score")
}

fn history(splits: &[(&str, usize, bool)]) -> GraphSplitHistory {
    let mut h = GraphSplitHistory::new();
    for &(node, neuron, active) in splits {
        h.add_constraint(split(node, neuron, active));
    }
    h
}

fn root_box() -> BoundedTensor {
    boxed(&[-1.0, -1.0], &[1.0, 1.0])
}

/// Row 0 closes at the root; row 1 only on both children of `relu_a[0]`.
fn record_split_tree(slot: &GraphCoverSlot) {
    // Root: row 0 proved (0.25 > 0), row 1 not (-1 < -0.5).
    slot.note_rows_bounds(&history(&[]), &[(0.25, 3.0), (-1.0, 2.0)]);
    // Both children: row 1 proved; row 0 inherited (an extra fact).
    slot.note_rows_bounds(
        &history(&[("relu_a", 0, true)]),
        &[(0.25, 3.0), (-0.25, 1.0)],
    );
    slot.note_rows_bounds(
        &history(&[("relu_a", 0, false)]),
        &[(0.3, 3.0), (-0.4, 1.0)],
    );
}

/// A complete per-row tree proves every row; the witnesses are the SHALLOWEST
/// leaves that close each branch (the root for row 0 — not its inherited
/// extras — and both children for row 1); deposit puts ONE cover per row into
/// the capture, so the instance door entails one single-atom clause per row
/// (the vit property's shape) and mints an instance token.
#[test]
fn mo_cover_rows_frame_proves_per_row_covers_and_the_door_mints() {
    let _guard = CAPTURE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (objectives, thresholds) = rows();
    let slot = GraphCoverSlot::default();
    // Records before arming go nowhere.
    slot.note_rows_bounds(&history(&[]), &[(1.0, 2.0), (1.0, 2.0)]);
    let scope = slot.arm_rows("test_mo", NODES, &root_box(), &objectives, &thresholds);
    assert!(scope.owns_frame());
    assert!(slot.rows_armed());
    record_split_tree(&slot);
    let GraphRowsVerdict::Proved(proved) = scope.finish(true, None) else {
        panic!("a complete per-row tree must prove");
    };
    assert!(!slot.is_armed(), "finish disarms");
    let summary = proved.summary();
    assert_eq!(summary.rows, 2);
    assert_eq!(summary.records, 3);
    // Row 0's witness is the root (the children's row-0 facts are extras);
    // row 1's witnesses are the two children.
    let regions = proved.regions();
    assert_eq!(regions.len(), 3, "root + two children: {regions:?}");
    assert_eq!((regions[0].literals).len(), 0);
    assert_eq!(regions[0].rows, vec![0]);
    for region in &regions[1..] {
        assert_eq!(region.rows, vec![1], "{region:?}");
        assert!(matches!(
            region.literals.as_slice(),
            [RegionLiteral::Relu { node, neuron: 0, .. }] if node == "relu_a"
        ));
        assert!(!region.empty);
    }

    reset_graph_cover_capture();
    proved.deposit();
    let capture = take_graph_cover_capture();
    assert_eq!(capture.len(), 2, "one cover per row");
    let clauses: Vec<Vec<GraphObjective>> = objectives
        .iter()
        .zip(&thresholds)
        .map(|(row, &t)| vec![GraphObjective::new(row, t, false).expect("finite")])
        .collect();
    let property =
        GraphInstanceProperty::new(vec![-1.0, -1.0], vec![1.0, 1.0], clauses).expect("well formed");
    let token = authorize_graph_instance(&property, &capture).expect("every clause entailed");
    assert_eq!(token.scope(), TokenScope::Instance);
    assert_eq!(token.classes(), &[0, 1]);
}

/// RED-first for the dropped-child class (`#violdrop` survivor loss): the
/// lane closes row 1 on one child only (the sibling was dropped by a
/// bookkeeping path, never recorded) — the frame REFUSES naming row 1, where
/// the legacy door published `Verified`. A row whose bound never proves it is
/// likewise uncovered; an unverified result discards the frame.
#[test]
fn mo_cover_rows_frame_refuses_a_row_whose_child_was_dropped() {
    let (objectives, thresholds) = rows();
    let slot = GraphCoverSlot::default();
    let scope = slot.arm_rows("test_mo", NODES, &root_box(), &objectives, &thresholds);
    slot.note_rows_bounds(&history(&[]), &[(0.25, 3.0), (-1.0, 2.0)]);
    slot.note_rows_bounds(
        &history(&[("relu_a", 0, true)]),
        &[(0.25, 3.0), (-0.25, 1.0)],
    );
    match scope.finish(true, None) {
        GraphRowsVerdict::Refused { row, refusal } => {
            assert_eq!(row, 1, "row 0 is covered at the root");
            assert!(
                matches!(refusal, CoverRefusal::Uncovered { .. }),
                "{refusal:?}"
            );
        }
        other => panic!("a dropped child must refuse: {other:?}"),
    }

    // Both children present, but the second child's row-1 bound does not
    // clear its threshold (-0.6 < -0.5): row 1 is still uncovered.
    let scope = slot.arm_rows("test_mo", NODES, &root_box(), &objectives, &thresholds);
    slot.note_rows_bounds(&history(&[]), &[(0.25, 3.0), (-1.0, 2.0)]);
    slot.note_rows_bounds(
        &history(&[("relu_a", 0, true)]),
        &[(0.25, 3.0), (-0.25, 1.0)],
    );
    slot.note_rows_bounds(
        &history(&[("relu_a", 0, false)]),
        &[(0.3, 3.0), (-0.6, 1.0)],
    );
    assert!(matches!(
        scope.finish(true, None),
        GraphRowsVerdict::Refused { row: 1, .. }
    ));

    // A run that did not verify is discarded.
    let scope = slot.arm_rows("test_mo", NODES, &root_box(), &objectives, &thresholds);
    slot.note_rows_bounds(&history(&[]), &[(0.25, 3.0), (-1.0, 2.0)]);
    assert!(matches!(
        scope.finish(false, None),
        GraphRowsVerdict::NotArmed
    ));
    assert!(!slot.is_armed());
}

/// M2c's certified-infeasibility rule on the graph lane: an emptiness close is
/// a leaf ONLY with a finite positive slack (then for every row); without one
/// the frame gives up (UNBUILT) — an error is never a close.
#[test]
fn mo_cover_rows_frame_emptiness_needs_a_positive_slack() {
    let (objectives, thresholds) = rows();
    let slot = GraphCoverSlot::default();
    let parent = history(&[("relu_a", 0, true)]);

    // The sibling half-space `relu_b[1] <= 0` is empty with slack 0.125; the
    // other half closes both rows by bound; `relu_a[0] <= 0` closes by bound.
    let scope = slot.arm_rows("test_mo", NODES, &root_box(), &objectives, &thresholds);
    slot.note_rows_bounds(&history(&[]), &[(-1.0, 3.0), (-1.0, 2.0)]);
    slot.note_rows_bounds(&history(&[("relu_a", 0, false)]), &[(0.5, 3.0), (0.5, 3.0)]);
    slot.note_rows_bounds(
        &history(&[("relu_a", 0, true), ("relu_b", 1, true)]),
        &[(0.5, 3.0), (0.5, 3.0)],
    );
    slot.note_rows_empty(&parent, Some(&split("relu_b", 1, false)), Some(0.125));
    let GraphRowsVerdict::Proved(proved) = scope.finish(true, None) else {
        panic!("a bound + slack-bearing emptiness tree must prove");
    };
    let summary = proved.summary();
    assert_eq!(summary.empty_records, 1);
    assert_eq!(summary.empty_leaves, 2, "the empty leaf serves both rows");
    let empty_regions = proved.regions().iter().filter(|r| r.empty).count();
    assert_eq!(empty_regions, 1, "{:?}", proved.regions());

    // The same tree whose emptiness carried no slack, a zero slack, a
    // negative slack or a NaN: UNBUILT, never a proof, never a refusal.
    for slack in [
        None,
        Some(0.0),
        Some(-0.25),
        Some(f64::NAN),
        Some(f64::INFINITY),
    ] {
        let scope = slot.arm_rows("test_mo", NODES, &root_box(), &objectives, &thresholds);
        slot.note_rows_bounds(&history(&[]), &[(-1.0, 3.0), (-1.0, 2.0)]);
        slot.note_rows_bounds(&history(&[("relu_a", 0, false)]), &[(0.5, 3.0), (0.5, 3.0)]);
        slot.note_rows_bounds(
            &history(&[("relu_a", 0, true), ("relu_b", 1, true)]),
            &[(0.5, 3.0), (0.5, 3.0)],
        );
        slot.note_rows_empty(&parent, Some(&split("relu_b", 1, false)), slack);
        match scope.finish(true, None) {
            GraphRowsVerdict::Unbuilt(reason) => {
                assert_eq!(reason, "uncertified_emptiness", "slack={slack:?}");
            }
            other => panic!("slack={slack:?}: must be UNBUILT, got {other:?}"),
        }
    }
}

/// The frame keeps to its own records: a one-objective record on a rows frame
/// and a rows record on a one-objective frame both poison the frame (UNBUILT);
/// a nested arm of either kind poisons the outer frame and gets none; a
/// mismatched row count and an unknown split node give up.
#[test]
fn mo_cover_rows_frame_poisons_on_foreign_records_and_nested_arms() {
    let (objectives, thresholds) = rows();
    let slot = GraphCoverSlot::default();

    let scope = slot.arm_rows("test_mo", NODES, &root_box(), &objectives, &thresholds);
    record_split_tree(&slot);
    slot.note_bound(
        GraphRecordKey {
            objective: &objectives[0],
            threshold: 0.0,
            verify_upper: false,
        },
        &history(&[]),
        None,
        1.0,
        2.0,
    );
    assert!(matches!(
        scope.finish(true, None),
        GraphRowsVerdict::Unbuilt(reason) if reason == "foreign_record"
    ));

    let single = slot.arm_paths_named(
        "test_single",
        NODES,
        &root_box(),
        &objectives[0],
        0.0,
        false,
    );
    slot.note_rows_bounds(&history(&[]), &[(1.0, 2.0), (1.0, 2.0)]);
    assert!(matches!(
        single.finish(true, None),
        GraphCoverVerdict::Unbuilt(reason) if reason == "foreign_record"
    ));

    let outer = slot.arm_rows("test_mo", NODES, &root_box(), &objectives, &thresholds);
    let nested = slot.arm_rows("test_mo", NODES, &root_box(), &objectives, &thresholds);
    assert!(!nested.owns_frame());
    record_split_tree(&slot);
    assert!(matches!(
        nested.finish(true, None),
        GraphRowsVerdict::NotArmed
    ));
    assert!(matches!(
        outer.finish(true, None),
        GraphRowsVerdict::Unbuilt(reason) if reason == "nested_driver"
    ));

    let outer = slot.arm_rows("test_mo", NODES, &root_box(), &objectives, &thresholds);
    let nested = slot.arm_paths_named(
        "test_single",
        NODES,
        &root_box(),
        &objectives[0],
        0.0,
        false,
    );
    assert!(matches!(
        nested.finish(true, None),
        GraphCoverVerdict::NotArmed
    ));
    assert!(matches!(
        outer.finish(true, None),
        GraphRowsVerdict::Unbuilt(reason) if reason == "nested_driver"
    ));

    let scope = slot.arm_rows("test_mo", NODES, &root_box(), &objectives, &thresholds);
    slot.note_rows_bounds(&history(&[]), &[(1.0, 2.0)]);
    assert!(matches!(
        scope.finish(true, None),
        GraphRowsVerdict::Unbuilt(reason) if reason == "row_count_mismatch"
    ));

    let scope = slot.arm_rows("test_mo", NODES, &root_box(), &objectives, &thresholds);
    slot.note_rows_bounds(
        &history(&[("not_a_node", 0, true)]),
        &[(1.0, 2.0), (1.0, 2.0)],
    );
    assert!(matches!(
        scope.finish(true, None),
        GraphRowsVerdict::Unbuilt(reason) if reason == "unknown_split_node"
    ));

    let scope = slot.arm_rows("test_mo", NODES, &root_box(), &objectives, &thresholds);
    slot.note_unbuildable("sequential_lane");
    record_split_tree(&slot);
    assert!(matches!(
        scope.finish(true, None),
        GraphRowsVerdict::Unbuilt(reason) if reason == "sequential_lane"
    ));
    assert!(!slot.is_armed());
}

/// A root-terminal close rests on LOWERS only: a lower strictly past the
/// threshold is a leaf, anything else is not evidence.
#[test]
fn mo_cover_rows_frame_root_lowers_prove_exactly_their_rows() {
    assert!(graph_lower_evidence(0.25, 0.0).is_some());
    assert!(graph_lower_evidence(0.0, 0.0).is_none(), "strict");
    assert!(graph_lower_evidence(f32::NAN, 0.0).is_none());
    assert!(graph_lower_evidence(f32::INFINITY, 0.0).is_none());
    assert!(graph_lower_evidence(1.0, f32::NEG_INFINITY).is_none());

    let (objectives, thresholds) = rows();
    let slot = GraphCoverSlot::default();
    let scope = slot.arm_rows("test_mo", NODES, &root_box(), &objectives, &thresholds);
    slot.note_rows_lowers(&history(&[]), &[0.125, -0.25]);
    let GraphRowsVerdict::Proved(proved) = scope.finish(true, None) else {
        panic!("both lowers clear their thresholds");
    };
    assert_eq!(proved.summary().max_depth, 0);

    let scope = slot.arm_rows("test_mo", NODES, &root_box(), &objectives, &thresholds);
    slot.note_rows_lowers(&history(&[]), &[0.125, -0.75]);
    assert!(matches!(
        scope.finish(true, None),
        GraphRowsVerdict::Refused { row: 1, .. }
    ));
}

fn lit(node: u32, neuron: u32, positive: bool) -> CoverLiteral {
    CoverLiteral {
        site: CoverSite::GraphRelu { node, neuron },
        positive,
    }
}

fn path(lits: &[CoverLiteral]) -> Option<LeafPath> {
    LeafPath::from_literals(lits.to_vec())
}

/// The graph emptiness slack is its own gate's evidence: accepted under the
/// graph policy only with a strictly positive finite slack, refused under the
/// margin-row policy, and the margin-row slack stays refused under the graph
/// policy (each lane's evidence is accepted only by its own gate).
#[test]
fn mo_cover_graph_emptiness_is_accepted_only_by_the_graph_gate() {
    let pos = path(&[lit(0, 0, true)]);
    let neg = path(&[lit(0, 0, false)]);
    let bound = || {
        Some(LeafEvidence::Bound(Certified::new(
            0.5,
            CertReceipt::GraphLaneBound,
        )))
    };

    let mut graph = CoverLedger::for_graph_lane();
    graph.leaf(pos.clone(), graph_emptiness_evidence(0.25));
    graph.leaf(neg.clone(), bound());
    let cover = graph.prove(&[0]).expect("graph slack under the graph gate");
    assert_eq!(cover.empty_leaves(), 1);

    assert!(graph_emptiness_evidence(0.0).is_none());
    assert!(graph_emptiness_evidence(-1.0).is_none());
    assert!(graph_emptiness_evidence(f64::NAN).is_none());
    assert!(graph_emptiness_evidence(f64::INFINITY).is_none());

    let mut graph = CoverLedger::for_graph_lane();
    graph.leaf(
        pos.clone(),
        Some(LeafEvidence::Empty(Certified::new(
            0.25,
            CertReceipt::MarginRowEmptiness,
        ))),
    );
    graph.leaf(neg.clone(), bound());
    assert_eq!(
        graph.prove(&[0]),
        Err(CoverRefusal::UncertifiedLeaf { leaf: 0 })
    );

    let mut margin_row = CoverLedger::new();
    margin_row.leaf(pos, graph_emptiness_evidence(0.25));
    margin_row.leaf(
        neg,
        Some(LeafEvidence::Bound(Certified::new(
            0.5,
            CertReceipt::MarginRowOutward,
        ))),
    );
    assert_eq!(
        margin_row.prove(&[0]),
        Err(CoverRefusal::UncertifiedLeaf { leaf: 0 })
    );
}

/// The witnesses are exactly what the Shannon expansion used, and collecting
/// them never changes the decision: on random split trees with random extra
/// facts, `prove_with_witnesses` agrees with `prove` (accept and refuse), and
/// every accepted cover is still accepted when the ledger holds ONLY its
/// witnesses (so a replay of the witnesses alone covers the root).
#[test]
fn mo_cover_witnesses_alone_cover_the_root_and_never_change_the_decision() {
    let mut state = 0x5EED_1234_ABCD_0001_u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut accepted = 0usize;
    let mut refused = 0usize;
    for _ in 0..400 {
        // A random full split tree over sites (node 0..3, neuron 0..2).
        let mut leaves: Vec<Vec<CoverLiteral>> = vec![Vec::new()];
        let splits = 1 + next() % 5;
        for _ in 0..splits {
            let at = (next() % leaves.len() as u64) as usize;
            let base = leaves.swap_remove(at);
            let site = (next() % 4) as u32;
            let neuron = (next() % 3) as u32;
            if base
                .iter()
                .any(|l| l.site == CoverSite::GraphRelu { node: site, neuron })
            {
                leaves.push(base);
                continue;
            }
            let mut a = base.clone();
            a.push(lit(site, neuron, true));
            let mut b = base;
            b.push(lit(site, neuron, false));
            leaves.push(a);
            leaves.push(b);
        }
        let drop_one = next() % 3 == 0 && leaves.len() > 1;
        if drop_one {
            let at = (next() % leaves.len() as u64) as usize;
            leaves.swap_remove(at);
        }
        let mut ledger = CoverLedger::for_graph_lane();
        for leaf in &leaves {
            ledger.leaf(path(leaf), graph_emptiness_evidence(0.5));
            // An extra (true) fact below the leaf.
            if next() % 2 == 0 {
                let mut deeper = leaf.clone();
                let site = CoverSite::GraphRelu { node: 7, neuron: 0 };
                if !deeper.iter().any(|l| l.site == site) {
                    deeper.push(CoverLiteral {
                        site,
                        positive: next() % 2 == 0,
                    });
                    ledger.leaf(path(&deeper), graph_emptiness_evidence(0.5));
                }
            }
        }
        let plain = ledger.prove(&[0]);
        let with = ledger.prove_with_witnesses(&[0], 1 << 20, None);
        assert_eq!(plain.is_ok(), with.is_ok(), "the decision must not change");
        match with {
            Ok((_, witnesses)) => {
                accepted += 1;
                assert_ne!((witnesses).len(), 0);
                let mut only = CoverLedger::for_graph_lane();
                for &w in &witnesses {
                    let wp = ledger.witness_path(w).expect("witness index").clone();
                    assert!(ledger.witness_is_empty(w));
                    only.leaf(Some(wp), graph_emptiness_evidence(0.5));
                }
                assert!(
                    only.prove(&[0]).is_ok(),
                    "the witnesses alone must cover the root"
                );
            }
            Err(_) => refused += 1,
        }
    }
    assert!(
        accepted > 50 && refused > 20,
        "accepted={accepted} refused={refused}"
    );
}

/// review(vit-v3), RED under the mutant that records an emptiness at its
/// PARENT's region: a construction-time emptiness closes exactly the EMPTY
/// half-space (`parent ∧ extra`), never its sibling. Here `relu_a[0] <= 0`
/// closes both rows by bound and `relu_a[0] >= 0 ∧ relu_b[1] <= 0` is empty
/// with a positive slack, but `relu_a[0] >= 0 ∧ relu_b[1] >= 0` is NEVER
/// recorded (the lane dropped it). Every row must be REFUSED: a frame that
/// wrote the empty half-space as `relu_a[0] >= 0` alone would "prove" a root
/// the lane never closed. Recording the missing sibling then proves (so the
/// refusal is about that region and nothing else).
#[test]
fn mo_cover_rows_frame_emptiness_closes_only_its_half_space() {
    let (objectives, thresholds) = rows();
    let slot = GraphCoverSlot::default();
    let parent = history(&[("relu_a", 0, true)]);
    let record = |with_sibling: bool| {
        let scope = slot.arm_rows("test_mo", NODES, &root_box(), &objectives, &thresholds);
        slot.note_rows_bounds(&history(&[]), &[(-1.0, 3.0), (-1.0, 2.0)]);
        slot.note_rows_bounds(&history(&[("relu_a", 0, false)]), &[(0.5, 3.0), (0.5, 3.0)]);
        slot.note_rows_empty(&parent, Some(&split("relu_b", 1, false)), Some(0.125));
        if with_sibling {
            slot.note_rows_bounds(
                &history(&[("relu_a", 0, true), ("relu_b", 1, true)]),
                &[(0.5, 3.0), (0.5, 3.0)],
            );
        }
        scope.finish(true, None)
    };
    match record(false) {
        GraphRowsVerdict::Refused { row, refusal } => {
            assert_eq!(row, 0, "the first row checked is the first uncovered one");
            assert!(
                matches!(refusal, CoverRefusal::Uncovered { .. }),
                "{refusal:?}"
            );
        }
        other => panic!(
            "an emptiness of `relu_a[0] >= 0 ∧ relu_b[1] <= 0` must not close \
             `relu_a[0] >= 0 ∧ relu_b[1] >= 0`: {other:?}"
        ),
    }
    let GraphRowsVerdict::Proved(proved) = record(true) else {
        panic!("with the sibling recorded the tree is complete");
    };
    let empty: Vec<_> = proved.regions().iter().filter(|r| r.empty).collect();
    assert_eq!(empty.len(), 1, "{:?}", proved.regions());
    assert_eq!(
        empty[0].literals.len(),
        2,
        "the empty witness is the half-space itself, not its parent: {:?}",
        empty[0]
    );
    assert!(!slot.is_armed());
}
