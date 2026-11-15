// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests of the graph-lane completeness ledger (#graph-bab-cover-ledger): the
//! overlap-tolerant path cover, the exact box cover and its clip slabs, the
//! lane's closure evidence, the per-verifier frame (containment of foreign
//! and nested records, unencodable histories, thread-count independence) and
//! the instance door.

use super::boxes::BoxLedger;
use super::cover::{CoverLedger, CoverLiteral, CoverRefusal, CoverSite, LeafEvidence, LeafPath};
use super::graph::{
    authorize_graph_instance, graph_bound_evidence, graph_clip_evidence, graph_infeasible_evidence,
    reset_graph_cover_capture, take_graph_cover_capture, GraphCoverSlot, GraphCoverVerdict,
    GraphInstanceProperty, GraphInstanceRefusal, GraphObjective, GraphRecordKey, CAPTURE_TEST_LOCK,
};
use super::token::{TokenScope, UnsatSource};
use super::value::{CertReceipt, Certified};
use crate::beta_crown::branching::{GenBabConstraint, GraphNeuronConstraint, GraphSplitHistory};
use crate::beta_crown::BetaCrownConfig;
use ny_tensor::BoundedTensor;

struct XorShift(u64);

impl XorShift {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn below(&mut self, n: usize) -> usize {
        usize::try_from(self.next_u64() % (n.max(1) as u64)).expect("small")
    }

    fn unit(&mut self) -> f32 {
        // A value in [0, 1) with 24 random bits: exact in binary32.
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

type Leaf = Vec<(CoverSite, bool)>;

fn relu(node: u32, neuron: u32) -> CoverSite {
    CoverSite::GraphRelu { node, neuron }
}

fn point(node: u32, neuron: u32, p: f32) -> CoverSite {
    CoverSite::GraphPoint {
        node,
        input: 0,
        neuron,
        point_bits: p.to_bits(),
    }
}

fn path(leaf: &[(CoverSite, bool)]) -> Option<LeafPath> {
    LeafPath::from_literals(
        leaf.iter()
            .map(|&(site, positive)| CoverLiteral { site, positive })
            .collect(),
    )
}

fn lane_bound(v: f64) -> Option<LeafEvidence> {
    Some(LeafEvidence::Bound(Certified::new(
        v,
        CertReceipt::GraphLaneBound,
    )))
}

fn graph_ledger(leaves: &[Leaf]) -> CoverLedger {
    let mut l = CoverLedger::for_graph_lane();
    for leaf in leaves {
        l.leaf(path(leaf), lane_bound(0.5));
    }
    l
}

fn site_pool() -> Vec<CoverSite> {
    (0..4u32)
        .flat_map(|n| (0..3u32).map(move |i| relu(n, i)))
        .chain([point(9, 0, 0.25), point(9, 0, -0.5), point(9, 1, 0.25)])
        .collect()
}

fn random_split_tree(rng: &mut XorShift) -> Vec<Leaf> {
    let pool = site_pool();
    let mut leaves = Vec::new();
    let mut stack: Vec<Leaf> = vec![Vec::new()];
    while let Some(node) = stack.pop() {
        let free: Vec<CoverSite> = pool
            .iter()
            .copied()
            .filter(|s| !node.iter().any(|(t, _)| t == s))
            .collect();
        if node.len() >= 6 || free.is_empty() || rng.below(3) == 0 {
            leaves.push(node);
            continue;
        }
        let site = free[rng.below(free.len())];
        for positive in [true, false] {
            let mut child = node.clone();
            child.push((site, positive));
            stack.push(child);
        }
    }
    leaves
}

fn random_path(rng: &mut XorShift) -> Leaf {
    let pool = site_pool();
    let mut leaf: Leaf = Vec::new();
    for _ in 0..rng.below(5) {
        let site = pool[rng.below(pool.len())];
        if !leaf.iter().any(|(s, _)| *s == site) {
            leaf.push((site, rng.below(2) == 0));
        }
    }
    leaf
}

/// A complete split tree is accepted in any recording order, with duplicated
/// leaves and with extra (true) facts about regions the tree never committed
/// — the retried and speculative evaluations the lane records.
#[test]
fn graph_cover_accepts_split_trees_with_extra_true_facts() {
    let mut rng = XorShift(0x5DEE_CE66_D1CE_4E5B);
    for case in 0..300 {
        let mut leaves = random_split_tree(&mut rng);
        assert!(
            graph_ledger(&leaves).proves_cover(&[0]),
            "case {case}: complete tree refused: {leaves:?}"
        );
        for _ in 0..rng.below(4) {
            let extra = random_path(&mut rng);
            leaves.push(extra);
        }
        if !leaves.is_empty() {
            let dup = leaves[rng.below(leaves.len())].clone();
            leaves.push(dup);
        }
        // Reverse, then rotate: a different recording order.
        leaves.reverse();
        let k = rng.below(leaves.len().max(1));
        leaves.rotate_left(k);
        let cover = graph_ledger(&leaves)
            .prove(&[0])
            .unwrap_or_else(|e| panic!("case {case}: tree plus extras refused: {e}"));
        assert_eq!(cover.leaves(), leaves.len());
        assert_eq!(cover.classes(), &[0]);
    }
}

/// Mutation pins: from a complete tree (no extras), dropping ANY leaf or
/// flipping ANY literal of ANY leaf leaves a region no leaf covers.
#[test]
fn graph_cover_refuses_every_dropped_or_flipped_leaf() {
    let mut rng = XorShift(0x2545_F491_4F6C_DD1D);
    let mut checked = 0usize;
    for case in 0..120 {
        let leaves = random_split_tree(&mut rng);
        if leaves.len() < 2 {
            continue;
        }
        for pick in 0..leaves.len() {
            let mut dropped = leaves.clone();
            dropped.remove(pick);
            assert!(
                matches!(
                    graph_ledger(&dropped).prove(&[0]),
                    Err(CoverRefusal::Uncovered { .. })
                ),
                "case {case}: dropping leaf {pick} still covered"
            );
            for lit in 0..leaves[pick].len() {
                let mut flipped = leaves.clone();
                flipped[pick][lit].1 = !flipped[pick][lit].1;
                assert!(
                    !graph_ledger(&flipped).proves_cover(&[0]),
                    "case {case}: flipping literal {lit} of leaf {pick} still covered"
                );
                checked += 1;
            }
        }
    }
    assert!(
        checked > 500,
        "only {checked} flips checked: the pin is VOID"
    );
}

/// JS F1's counterexample on the graph sites: siblings that split DIFFERENT
/// neurons leave `a on ∧ b off` uncovered, and an empty ledger proves nothing.
#[test]
fn graph_cover_refuses_siblings_that_split_different_neurons() {
    let (a, b) = (relu(0, 5), relu(1, 7));
    let l = graph_ledger(&[vec![(a, false)], vec![(b, true)]]);
    assert!(l.kraft_ok(), "the depth multiset {{1,1}} sums to 1");
    assert!(matches!(l.prove(&[0]), Err(CoverRefusal::Uncovered { .. })));
    assert_eq!(
        CoverLedger::for_graph_lane().prove(&[0]),
        Err(CoverRefusal::NoLeaves)
    );
    // Two GenBaB points on one neuron are two sites: `x ≥ 0.25` and
    // `x ≤ -0.5` leave `(-0.5, 0.25)` open.
    let l = graph_ledger(&[
        vec![(point(9, 0, 0.25), true)],
        vec![(point(9, 0, -0.5), false)],
    ]);
    assert!(!l.proves_cover(&[0]));
    // ... while both sides of ONE point cover the line.
    let l = graph_ledger(&[
        vec![(point(9, 0, 0.25), true)],
        vec![(point(9, 0, 0.25), false)],
    ]);
    assert!(l.proves_cover(&[0]));
}

/// Running out of work is `Budget`, never `Uncovered`: the lane reads it as
/// "not checked" (legacy door), not as a finding.
#[test]
fn graph_cover_budget_is_not_a_refusal() {
    let mut rng = XorShift(0x9E37_79B9_7F4A_7C15);
    let mut leaves = random_split_tree(&mut rng);
    while leaves.len() < 8 {
        leaves = random_split_tree(&mut rng);
    }
    let l = graph_ledger(&leaves);
    assert!(matches!(
        l.prove_within(&[0], 3, None),
        Err(CoverRefusal::Budget { .. })
    ));
    let past = std::time::Instant::now();
    // A deadline already passed is only consulted at a poll boundary, so a
    // small tree still finishes; the budget is what bounds the work.
    assert!(l.prove_within(&[0], usize::MAX, Some(past)).is_ok());
}

/// Each policy accepts only its own lane's evidence: a margin-row ledger
/// refuses graph-lane closes, a graph ledger refuses margin-row bounds,
/// emptiness slacks and nested subtree tokens, and a lane close must name a
/// graph-lane receipt.
#[test]
fn graph_policies_refuse_each_others_evidence() {
    let x = relu(0, 0);
    let pos = path(&[(x, true)]);
    let neg = path(&[(x, false)]);

    let mut mr = CoverLedger::new();
    mr.leaf(pos.clone(), lane_bound(0.5));
    mr.leaf(
        neg.clone(),
        Some(LeafEvidence::Bound(Certified::new(
            0.5,
            CertReceipt::MarginRowOutward,
        ))),
    );
    assert_eq!(
        mr.prove(&[0]),
        Err(CoverRefusal::UncertifiedLeaf { leaf: 0 })
    );
    let mut mr = CoverLedger::new();
    mr.leaf(pos.clone(), Some(graph_infeasible_evidence()));
    mr.leaf(
        neg.clone(),
        Some(LeafEvidence::Bound(Certified::new(
            0.5,
            CertReceipt::MarginRowOutward,
        ))),
    );
    assert_eq!(
        mr.prove(&[0]),
        Err(CoverRefusal::UncertifiedLeaf { leaf: 0 })
    );

    let mut g = CoverLedger::for_graph_lane();
    g.leaf(
        pos.clone(),
        Some(LeafEvidence::Bound(Certified::new(
            0.5,
            CertReceipt::MarginRowOutward,
        ))),
    );
    g.leaf(neg.clone(), lane_bound(0.5));
    assert_eq!(
        g.prove(&[0]),
        Err(CoverRefusal::UncertifiedLeaf { leaf: 0 })
    );
    let mut g = CoverLedger::for_graph_lane();
    g.leaf(
        pos.clone(),
        Some(LeafEvidence::Empty(Certified::new(
            0.5,
            CertReceipt::MarginRowEmptiness,
        ))),
    );
    g.leaf(neg.clone(), lane_bound(0.5));
    assert_eq!(
        g.prove(&[0]),
        Err(CoverRefusal::UncertifiedLeaf { leaf: 0 })
    );
    let mut g = CoverLedger::for_graph_lane();
    g.leaf(
        pos.clone(),
        Some(LeafEvidence::LaneClose(CertReceipt::FrozenDual)),
    );
    g.leaf(neg.clone(), lane_bound(0.5));
    assert_eq!(
        g.prove(&[0]),
        Err(CoverRefusal::UncertifiedLeaf { leaf: 0 })
    );
    // A zero margin is not a close.
    let mut g = CoverLedger::for_graph_lane();
    g.leaf(pos.clone(), lane_bound(0.0));
    g.leaf(neg.clone(), lane_bound(0.5));
    assert_eq!(
        g.prove(&[0]),
        Err(CoverRefusal::UncertifiedLeaf { leaf: 0 })
    );

    // The accepted shape: a bound leaf and a lane close.
    let mut g = CoverLedger::for_graph_lane();
    g.leaf(pos, Some(graph_infeasible_evidence()));
    g.leaf(neg, lane_bound(0.125));
    let cover = g.prove(&[0]).expect("bound plus lane close");
    assert_eq!(cover.lane_leaves(), 1);
    assert_eq!(cover.min_margin().to_bits(), 0.125f64.to_bits());
}

/// The evidence adapter IS the lane's closure rule (`domain_is_verified_for_mode`)
/// on every input, in both directions, including ties, inverted and
/// non-finite pairs; its margin is exact and strictly positive.
#[test]
fn graph_bound_evidence_is_the_lanes_closure_rule() {
    let specials = [
        0.0f32,
        -0.0,
        1.0,
        -1.0,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        f32::from_bits(1),
        f32::MAX,
        -f32::MAX,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        0.5,
        0.500_000_06,
    ];
    let mut rng = XorShift(0xD1B5_4A32_D192_ED03);
    let mut values: Vec<f32> = specials.to_vec();
    for _ in 0..40 {
        values.push(f32::from_bits(rng.next_u64() as u32));
    }
    let mut agreed = 0usize;
    for &lower in &values {
        for &upper in &values {
            for &threshold in specials.iter().take(8) {
                for upper_mode in [false, true] {
                    let rule = BetaCrownConfig::domain_is_verified_for_mode(
                        upper_mode, lower, upper, threshold,
                    );
                    let evidence = graph_bound_evidence(upper_mode, lower, upper, threshold);
                    assert_eq!(
                        evidence.is_some(),
                        rule,
                        "lower={lower:e} upper={upper:e} thr={threshold:e} upper_mode={upper_mode}"
                    );
                    if let Some(LeafEvidence::Bound(c)) = evidence {
                        assert_eq!(c.receipt(), CertReceipt::GraphLaneBound);
                        assert!(c.value() > 0.0 && c.value().is_finite());
                        let exact = if upper_mode {
                            f64::from(threshold) - f64::from(upper)
                        } else {
                            f64::from(lower) - f64::from(threshold)
                        };
                        assert_eq!(c.value().to_bits(), exact.to_bits());
                    }
                    agreed += 1;
                }
            }
        }
    }
    assert!(agreed > 10_000, "only {agreed} cases compared");
    assert_eq!(
        graph_clip_evidence(),
        LeafEvidence::LaneClose(CertReceipt::GraphLaneClip)
    );
}

/// Upper-bound mode is the negated lower-bound claim, exactly; entailment
/// compares rows as values (the two zeros agree) and accepts a stricter
/// threshold only.
#[test]
fn graph_objective_normalizes_upper_mode_exactly() {
    let lower = GraphObjective::new(&[1.0, -0.0, 0.25], 0.5, false).expect("finite");
    let upper = GraphObjective::new(&[-1.0, 0.0, -0.25], -0.5, true).expect("finite");
    assert_eq!(lower, upper);
    assert!(lower.entails(&upper) && upper.entails(&lower));
    let looser = GraphObjective::new(&[1.0, 0.0, 0.25], 0.25, false).expect("finite");
    assert!(
        lower.entails(&looser),
        "0.5 > 0.25: the stricter proof excludes the atom"
    );
    assert!(!looser.entails(&lower), "0.25 does not reach 0.5");
    let other_row = GraphObjective::new(&[1.0, 0.0, 0.5], 0.5, false).expect("finite");
    assert!(!lower.entails(&other_row));
    let shorter = GraphObjective::new(&[1.0, 0.0], 0.5, false).expect("finite");
    assert!(!lower.entails(&shorter));
    assert!(GraphObjective::new(&[], 0.0, false).is_none());
    assert!(GraphObjective::new(&[f32::NAN], 0.0, false).is_none());
    assert!(GraphObjective::new(&[1.0], f32::INFINITY, true).is_none());
}

// ---------------------------------------------------------------------------
// The box cover.
// ---------------------------------------------------------------------------

fn f64s(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| f64::from(x)).collect()
}

/// A random bisection tree over `[lo, hi]` with the lane's own midpoint
/// arithmetic, random clips (recorded as slabs) and the leaves' own boxes.
fn random_box_tree(
    rng: &mut XorShift,
    ledger: &mut BoxLedger,
    lo: &[f32],
    hi: &[f32],
    clips: bool,
    leaves: &mut Vec<(Vec<f32>, Vec<f32>)>,
) {
    let dims = lo.len();
    let mut stack: Vec<(Vec<f32>, Vec<f32>, usize)> = vec![(lo.to_vec(), hi.to_vec(), 0)];
    while let Some((mut l, mut h, depth)) = stack.pop() {
        if clips && depth > 0 && rng.below(4) == 0 {
            // Clip: shrink one side of one dimension to a point inside.
            let d = rng.below(dims);
            let (pre_l, pre_h) = (l.clone(), h.clone());
            let t = l[d] + (h[d] - l[d]) * rng.unit();
            if rng.below(2) == 0 {
                l[d] = t.min(h[d]);
            } else {
                h[d] = t.max(l[d]);
            }
            ledger.clip(
                &f64s(&pre_l),
                &f64s(&pre_h),
                &f64s(&l),
                &f64s(&h),
                &graph_clip_evidence(),
            );
        }
        if depth >= 7 || rng.below(3) == 0 {
            leaves.push((l, h));
            continue;
        }
        let d = rng.below(dims);
        if l[d].partial_cmp(&h[d]) != Some(std::cmp::Ordering::Less) {
            leaves.push((l, h));
            continue;
        }
        let mid = l[d] + (h[d] - l[d]) / 2.0;
        let mut left_h = h.clone();
        left_h[d] = mid;
        let mut right_l = l.clone();
        right_l[d] = mid;
        stack.push((l, left_h, depth + 1));
        stack.push((right_l, h, depth + 1));
    }
}

fn record_leaves(ledger: &mut BoxLedger, leaves: &[(Vec<f32>, Vec<f32>)]) {
    for (l, h) in leaves {
        ledger.leaf(&f64s(l), &f64s(h), lane_bound(0.75));
    }
}

/// Every random bisection tree (with clips recorded as slabs, extra true
/// facts, duplicates and a degenerate dimension) covers its root box.
#[test]
fn box_cover_accepts_bisection_trees_with_clips_and_extras() {
    let mut rng = XorShift(0xA076_1D64_78BD_642F);
    let lo = [-1.0f32, 0.0, 2.0, 5.0];
    let hi = [1.0f32, 0.75, 2.0, 7.5];
    for case in 0..200 {
        let mut ledger = BoxLedger::new(lo.len());
        let mut leaves = Vec::new();
        random_box_tree(&mut rng, &mut ledger, &lo, &hi, true, &mut leaves);
        for _ in 0..rng.below(3) {
            let d = rng.below(lo.len());
            let (mut l, mut h) = (lo.to_vec(), hi.to_vec());
            l[d] = lo[d] + (hi[d] - lo[d]) * rng.unit();
            h[d] = l[d].max(h[d] * rng.unit()).max(l[d]);
            leaves.push((l, h));
        }
        if !leaves.is_empty() {
            let dup = leaves[rng.below(leaves.len())].clone();
            leaves.push(dup);
        }
        leaves.reverse();
        record_leaves(&mut ledger, &leaves);
        let cover = ledger
            .prove(&f64s(&lo), &f64s(&hi), &[0], None)
            .unwrap_or_else(|e| panic!("case {case}: bisection tree refused: {e}"));
        assert!(cover.leaves() >= leaves.len());
        assert_eq!(cover.min_margin().to_bits(), 0.75f64.to_bits());
    }
}

/// Mutation pins, exact at the ulp: dropping any leaf box of a (clip-free)
/// tree is refused, a one-ulp gap between two boxes is refused, and the same
/// boxes touching at a shared face are accepted.
#[test]
fn box_cover_refuses_a_dropped_box_and_a_one_ulp_gap() {
    let mut rng = XorShift(0xE703_7ED1_A0B4_28DB);
    let lo = [0.0f32, -3.0, 1.0e-30];
    let hi = [1.0f32, 3.0, 1.0e-29];
    let mut checked = 0usize;
    for case in 0..60 {
        let mut probe = BoxLedger::new(lo.len());
        let mut leaves = Vec::new();
        random_box_tree(&mut rng, &mut probe, &lo, &hi, false, &mut leaves);
        assert_eq!(probe.recorded(), 0, "a clip-free tree records no slab");
        if leaves.len() < 2 {
            continue;
        }
        for pick in 0..leaves.len() {
            let mut ledger = BoxLedger::new(lo.len());
            let mut kept = leaves.clone();
            kept.remove(pick);
            record_leaves(&mut ledger, &kept);
            assert!(
                matches!(
                    ledger.prove(&f64s(&lo), &f64s(&hi), &[0], None),
                    Err(CoverRefusal::Uncovered { .. })
                ),
                "case {case}: dropping box {pick} still covered"
            );
            checked += 1;
        }
    }
    assert!(
        checked > 50,
        "only {checked} drops checked: the pin is VOID"
    );

    let m = 0.3f32;
    let gap = f32::from_bits(m.to_bits() + 1);
    let mut ledger = BoxLedger::new(1);
    ledger.leaf(&[0.0], &[f64::from(m)], lane_bound(1.0));
    ledger.leaf(&[f64::from(gap)], &[1.0], lane_bound(1.0));
    assert!(matches!(
        ledger.prove(&[0.0], &[1.0], &[0], None),
        Err(CoverRefusal::Uncovered { .. })
    ));
    let mut ledger = BoxLedger::new(1);
    ledger.leaf(&[0.0], &[f64::from(m)], lane_bound(1.0));
    ledger.leaf(&[f64::from(m)], &[1.0], lane_bound(1.0));
    assert!(ledger.prove(&[0.0], &[1.0], &[0], None).is_ok());
    // A leaf that stops one ulp short of the root's upper face.
    let mut ledger = BoxLedger::new(1);
    ledger.leaf(
        &[0.0],
        &[f64::from(f32::from_bits(1.0f32.to_bits() - 1))],
        lane_bound(1.0),
    );
    assert!(ledger.prove(&[0.0], &[1.0], &[0], None).is_err());
    // Malformed boxes are refused, never skipped.
    let mut ledger = BoxLedger::new(2);
    ledger.leaf(&[0.0], &[1.0], lane_bound(1.0));
    assert_eq!(
        ledger.prove(&[0.0, 0.0], &[1.0, 1.0], &[0], None),
        Err(CoverRefusal::BadBox { leaf: 0 })
    );
    let mut ledger = BoxLedger::new(1);
    ledger.leaf(&[0.5], &[0.25], lane_bound(1.0));
    assert_eq!(
        ledger.prove(&[0.0], &[1.0], &[0], None),
        Err(CoverRefusal::BadBox { leaf: 0 })
    );
    let mut ledger = BoxLedger::new(1);
    ledger.leaf(&[0.0], &[1.0], None);
    assert_eq!(
        ledger.prove(&[0.0], &[1.0], &[0], None),
        Err(CoverRefusal::UncertifiedLeaf { leaf: 0 })
    );
}

/// The clip slabs of `pre ∖ post` plus the clamped `pre ∩ post` cover `pre`
/// exactly, there are at most `2·D` of them, and none reaches into the
/// interior of `post`.
#[test]
fn box_clip_slabs_tile_the_excluded_part_exactly() {
    let mut rng = XorShift(0x8CB9_2BA7_2F3D_8DD7);
    let dims = 3;
    for case in 0..400 {
        let pre_lo: Vec<f64> = (0..dims).map(|_| f64::from(rng.unit()) - 0.5).collect();
        let pre_hi: Vec<f64> = pre_lo.iter().map(|l| l + f64::from(rng.unit())).collect();
        let post_lo: Vec<f64> = (0..dims)
            .map(|_| f64::from(rng.unit()) * 1.6 - 0.8)
            .collect();
        let post_hi: Vec<f64> = post_lo
            .iter()
            .map(|l| l + f64::from(rng.unit()) - 0.1)
            .collect();
        let mut ledger = BoxLedger::new(dims);
        let slabs = ledger.clip(&pre_lo, &pre_hi, &post_lo, &post_hi, &graph_clip_evidence());
        assert!(slabs <= 2 * dims, "case {case}: {slabs} slabs");
        // The clamped intersection the lane keeps searching.
        let keep_lo: Vec<f64> = (0..dims)
            .map(|d| post_lo[d].clamp(pre_lo[d], pre_hi[d]))
            .collect();
        let keep_hi: Vec<f64> = (0..dims)
            .map(|d| post_hi[d].clamp(keep_lo[d], pre_hi[d]))
            .collect();
        let mut full = BoxLedger::new(dims);
        full.clip(&pre_lo, &pre_hi, &post_lo, &post_hi, &graph_clip_evidence());
        full.leaf(&keep_lo, &keep_hi, lane_bound(1.0));
        assert!(
            full.prove(&pre_lo, &pre_hi, &[0], None).is_ok(),
            "case {case}: slabs plus the kept box do not cover pre"
        );
        // Without the kept box, a strictly interior point of it is uncovered.
        let open = (0..dims).all(|d| keep_lo[d] < keep_hi[d]);
        if open && slabs > 0 {
            assert!(
                ledger.prove(&pre_lo, &pre_hi, &[0], None).is_err(),
                "case {case}: slabs alone covered pre although post is open inside it"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// The per-verifier frame.
// ---------------------------------------------------------------------------

const NODES: [&str; 3] = ["input_linear", "relu_a", "relu_b"];

fn boxed(lo: &[f32], hi: &[f32]) -> BoundedTensor {
    BoundedTensor::new(
        ndarray::Array1::from(lo.to_vec()).into_dyn(),
        ndarray::Array1::from(hi.to_vec()).into_dyn(),
    )
    .expect("well-formed box")
}

fn relu_split(node: &str, neuron: usize, active: bool) -> GraphNeuronConstraint {
    GraphNeuronConstraint::new(node.to_string(), neuron, active, 1.0).expect("finite score")
}

fn history(splits: &[(&str, usize, bool)]) -> GraphSplitHistory {
    let mut h = GraphSplitHistory::new();
    for &(node, neuron, active) in splits {
        h.add_constraint(relu_split(node, neuron, active));
    }
    h
}

const OBJ: [f32; 2] = [1.0, -1.0];

fn key() -> GraphRecordKey<'static> {
    GraphRecordKey {
        objective: &OBJ,
        threshold: 0.0,
        verify_upper: false,
    }
}

/// A complete ReLU tree (one child closed by a bound, one by a certified
/// empty half-space the lane never built) is proved; the same tree missing a
/// child is REFUSED; a close whose bound does not verify is refused; a run
/// that did not verify is discarded.
#[test]
fn graph_frame_proves_a_complete_relu_tree_and_refuses_a_missing_child() {
    let root = boxed(&[-1.0, -1.0], &[1.0, 1.0]);
    let slot = GraphCoverSlot::default();
    assert!(!slot.is_armed());
    // Records before arming go nowhere.
    slot.note_bound(key(), &history(&[]), None, 1.0, 2.0);

    let scope = slot.arm_paths_named("test_relu", NODES, &root, &OBJ, 0.0, false);
    assert!(slot.is_armed());
    let parent = history(&[("relu_a", 0, true)]);
    slot.note_bound(key(), &history(&[("relu_a", 0, false)]), None, 0.25, 3.0);
    slot.note_bound(
        key(),
        &history(&[("relu_a", 0, true), ("relu_b", 1, true)]),
        None,
        0.5,
        1.0,
    );
    slot.note_infeasible(key(), &parent, Some(&relu_split("relu_b", 1, false)));
    match scope.finish(true, None) {
        GraphCoverVerdict::Proved(summary) => {
            assert_eq!(summary.leaves, 3);
            assert_eq!(summary.lane_leaves, 1);
            assert_eq!(summary.max_depth, 2);
            assert_eq!(summary.min_margin.to_bits(), 0.25f64.to_bits());
        }
        other => panic!("complete tree not proved: {other:?}"),
    }
    assert!(!slot.is_armed(), "finish disarms");

    // The same tree without the empty half-space: `a on ∧ b off` is open.
    let scope = slot.arm_paths_named("test_relu", NODES, &root, &OBJ, 0.0, false);
    slot.note_bound(key(), &history(&[("relu_a", 0, false)]), None, 0.25, 3.0);
    slot.note_bound(
        key(),
        &history(&[("relu_a", 0, true), ("relu_b", 1, true)]),
        None,
        0.5,
        1.0,
    );
    assert!(matches!(
        scope.finish(true, None),
        GraphCoverVerdict::Refused(CoverRefusal::Uncovered { .. })
    ));

    // A "close" whose bound does not clear the threshold is not evidence.
    let scope = slot.arm_paths_named("test_relu", NODES, &root, &OBJ, 0.0, false);
    slot.note_bound(key(), &history(&[]), None, -0.5, 1.0);
    assert!(matches!(
        scope.finish(true, None),
        GraphCoverVerdict::Refused(CoverRefusal::UncertifiedLeaf { .. })
    ));

    // A run that ended `Unknown` checks nothing; a dropped scope disarms.
    let scope = slot.arm_paths_named("test_relu", NODES, &root, &OBJ, 0.0, false);
    slot.note_bound(key(), &history(&[]), None, 1.0, 2.0);
    assert_eq!(scope.finish(false, None), GraphCoverVerdict::NotArmed);
    {
        let _scope = slot.arm_paths_named("test_relu", NODES, &root, &OBJ, 0.0, false);
        assert!(slot.is_armed());
    }
    assert!(!slot.is_armed(), "dropping the scope disarms");
}

/// Containment: a record for another objective, threshold or direction, or a
/// nested arm on the same verifier, leaves the frame UNBUILT (legacy door) —
/// never proved on foreign evidence, never refused.
#[test]
fn graph_frame_poisons_foreign_records_and_nested_arms() {
    let root = boxed(&[0.0], &[1.0]);
    let slot = GraphCoverSlot::default();
    let foreign = [
        GraphRecordKey {
            objective: &[1.0, -0.5],
            threshold: 0.0,
            verify_upper: false,
        },
        GraphRecordKey {
            objective: &OBJ,
            threshold: 0.125,
            verify_upper: false,
        },
        GraphRecordKey {
            objective: &OBJ,
            threshold: 0.0,
            verify_upper: true,
        },
    ];
    for other in foreign {
        let scope = slot.arm_paths_named("test_relu", NODES, &root, &OBJ, 0.0, false);
        slot.note_bound(key(), &history(&[]), None, 1.0, 2.0);
        slot.note_bound(other, &history(&[]), None, 1.0, 2.0);
        assert!(
            matches!(scope.finish(true, None), GraphCoverVerdict::Unbuilt(ref r) if r == "foreign_record"),
            "a foreign record must poison the frame: {other:?}"
        );
    }
    let outer = slot.arm_paths_named("test_relu", NODES, &root, &OBJ, 0.0, false);
    slot.note_bound(key(), &history(&[]), None, 1.0, 2.0);
    let inner = slot.arm_paths_named("test_relu", NODES, &root, &OBJ, 0.0, false);
    assert_eq!(inner.finish(true, None), GraphCoverVerdict::NotArmed);
    assert!(
        matches!(outer.finish(true, None), GraphCoverVerdict::Unbuilt(ref r) if r == "nested_driver")
    );
    // A box record on a path frame is a kind mismatch, not a proof.
    let scope = slot.arm_paths_named("test_relu", NODES, &root, &OBJ, 0.0, false);
    slot.note_box_bound(key(), &root, 1.0, 2.0);
    assert!(matches!(
        scope.finish(true, None),
        GraphCoverVerdict::Unbuilt(_)
    ));
}

/// Histories the ledger cannot write as half-space paths leave the frame
/// UNBUILT: an unknown node, a contradictory history, an explicit
/// unbuildable close. GenBaB points ARE encodable.
#[test]
fn graph_frame_leaves_unencodable_histories_unbuilt() {
    let root = boxed(&[0.0], &[1.0]);
    let slot = GraphCoverSlot::default();
    let scope = slot.arm_paths_named("test_relu", NODES, &root, &OBJ, 0.0, false);
    slot.note_bound(key(), &history(&[("not_a_node", 0, true)]), None, 1.0, 2.0);
    assert!(
        matches!(scope.finish(true, None), GraphCoverVerdict::Unbuilt(ref r) if r == "unknown_split_node")
    );
    let scope = slot.arm_paths_named("test_relu", NODES, &root, &OBJ, 0.0, false);
    slot.note_bound(
        key(),
        &history(&[("relu_a", 0, true), ("relu_a", 0, false)]),
        None,
        1.0,
        2.0,
    );
    assert!(
        matches!(scope.finish(true, None), GraphCoverVerdict::Unbuilt(ref r) if r == "contradictory_path")
    );
    let scope = slot.arm_paths_named("test_relu", NODES, &root, &OBJ, 0.0, false);
    slot.note_bound(key(), &history(&[]), None, 1.0, 2.0);
    slot.note_unbuildable("genbab_empty_child");
    assert!(
        matches!(scope.finish(true, None), GraphCoverVerdict::Unbuilt(ref r) if r == "genbab_empty_child")
    );
    // GenBaB: both sides of one point on `relu_b`'s input.
    let scope = slot.arm_paths_named("test_relu", NODES, &root, &OBJ, 0.0, false);
    for upper in [true, false] {
        let mut h = GraphSplitHistory::new();
        h.add_genbab_constraint(
            GenBabConstraint::new("relu_b".to_string(), 2, 0.375, upper, 1.0).expect("finite"),
        );
        slot.note_bound(key(), &h, None, 1.0, 2.0);
    }
    assert!(matches!(
        scope.finish(true, None),
        GraphCoverVerdict::Proved(_)
    ));
}

/// The same facts recorded from 1 thread or from 8 (in different orders)
/// give the same outcome and the same summary: the gate reads a set.
#[test]
fn graph_frame_is_thread_count_independent() {
    let root = boxed(&[0.0, 0.0], &[1.0, 1.0]);
    let mut rng = XorShift(0x94D0_49BB_1331_11EB);
    let mut facts: Vec<Vec<(&str, usize, bool)>> = Vec::new();
    // A complete depth-3 tree over (relu_a,0), (relu_b,1), (relu_a,1).
    for a in [true, false] {
        for b in [true, false] {
            for c in [true, false] {
                facts.push(vec![("relu_a", 0, a), ("relu_b", 1, b), ("relu_a", 1, c)]);
            }
        }
    }
    let run = |threads: usize, order: &[usize]| {
        let slot = GraphCoverSlot::default();
        let scope = slot.arm_paths_named("test_relu", NODES, &root, &OBJ, 0.0, false);
        std::thread::scope(|s| {
            for t in 0..threads {
                let slot = &slot;
                let facts = &facts;
                s.spawn(move || {
                    for (i, &f) in order.iter().enumerate() {
                        if i % threads == t {
                            slot.note_bound(key(), &history(&facts[f]), None, 0.5 + f as f32, 9.0);
                        }
                    }
                });
            }
        });
        scope.finish(true, None)
    };
    let serial: Vec<usize> = (0..facts.len()).collect();
    let reference = run(1, &serial);
    assert!(matches!(reference, GraphCoverVerdict::Proved(_)));
    for threads in [2, 3, 8] {
        let mut order = serial.clone();
        for i in (1..order.len()).rev() {
            order.swap(i, rng.below(i + 1));
        }
        assert_eq!(run(threads, &order), reference, "threads={threads}");
    }
}

/// The instance door: a token only when every clause is entailed by a
/// captured cover on a root box containing the declared box. Instance-scoped
/// and authorized; each missing piece refuses.
#[test]
fn graph_instance_door_needs_every_clause_on_a_containing_box() {
    let _guard = CAPTURE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    // Rows no other test uses, so a concurrently deposited cover cannot match.
    let row_a = [3.0f32, -7.0, 0.5, 0.0, 0.0];
    let row_b = [0.0f32, 0.0, -2.5, 11.0, 1.0];
    let engine_box = boxed(&[-1.0, -2.0], &[1.0, 2.0]);
    let prove = |row: &[f32], threshold: f32| {
        let slot = GraphCoverSlot::default();
        let scope = slot.arm_boxes("test_input_split", &engine_box, row, threshold, false);
        slot.note_box_bound(
            GraphRecordKey {
                objective: row,
                threshold,
                verify_upper: false,
            },
            &engine_box,
            threshold + 1.0,
            threshold + 2.0,
        );
        assert!(matches!(
            scope.finish(true, None),
            GraphCoverVerdict::Proved(_)
        ));
    };
    let atom = |row: &[f32], thr: f32| GraphObjective::new(row, thr, false).expect("finite");
    let declared = (vec![-0.5, -2.0], vec![1.0, 1.5]);

    reset_graph_cover_capture();
    prove(&row_a, 0.25);
    let capture = take_graph_cover_capture();
    assert!(!capture.is_empty());

    // One clause, the atom certified at a stricter threshold: a token.
    let property = GraphInstanceProperty::new(
        declared.0.clone(),
        declared.1.clone(),
        vec![vec![atom(&row_b, 0.0), atom(&row_a, 0.0)]],
    )
    .expect("well formed");
    let token = authorize_graph_instance(&property, &capture).expect("clause entailed");
    assert_eq!(token.source(), UnsatSource::Authorized);
    assert_eq!(token.scope(), TokenScope::Instance);
    assert!(token.publishable());
    assert_eq!(token.classes(), &[0]);

    // A second clause nobody proved: refused.
    let two = GraphInstanceProperty::new(
        declared.0.clone(),
        declared.1.clone(),
        vec![vec![atom(&row_a, 0.0)], vec![atom(&row_b, 0.0)]],
    )
    .expect("well formed");
    assert_eq!(
        authorize_graph_instance(&two, &capture),
        Err(GraphInstanceRefusal::ClauseUncovered { clause: 1 })
    );
    // A looser proof does not exclude a stricter atom.
    let strict = GraphInstanceProperty::new(
        declared.0.clone(),
        declared.1,
        vec![vec![atom(&row_a, 0.5)]],
    )
    .expect("well formed");
    assert!(authorize_graph_instance(&strict, &capture).is_err());
    // A declared box reaching outside the proved root box: refused.
    let wide = GraphInstanceProperty::new(
        vec![-0.5, -2.0],
        vec![1.0, 2.5],
        vec![vec![atom(&row_a, 0.0)]],
    )
    .expect("well formed");
    assert!(authorize_graph_instance(&wide, &capture).is_err());

    // Both clauses proved (two invocations): a token for both.
    reset_graph_cover_capture();
    prove(&row_a, 0.25);
    prove(&row_b, 0.0);
    let capture = take_graph_cover_capture();
    let token = authorize_graph_instance(&two, &capture).expect("both clauses entailed");
    assert_eq!(token.classes(), &[0, 1]);

    // Nothing captured (the capture was not armed): refused.
    prove(&row_a, 0.25);
    assert_eq!(
        authorize_graph_instance(&property, &take_graph_cover_capture()),
        Err(GraphInstanceRefusal::NothingCaptured)
    );
    // Malformed properties do not exist.
    assert!(GraphInstanceProperty::new(vec![0.0], vec![1.0], vec![]).is_none());
    assert!(GraphInstanceProperty::new(vec![0.0], vec![1.0], vec![vec![]]).is_none());
    assert!(
        GraphInstanceProperty::new(vec![1.0], vec![0.0], vec![vec![atom(&row_a, 0.0)]]).is_none()
    );
}

/// R6: when several captured covers entail the same clause, the token is a
/// function of the captured SET — the same classes and the same (best)
/// `min_margin` whichever order the drivers deposited them in.
#[test]
fn graph_instance_door_token_is_deposit_order_independent() {
    let _guard = CAPTURE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    // A row no other test uses, so a concurrently deposited cover cannot match.
    let row = [5.0f32, 0.0, -13.0];
    let root = boxed(&[-1.0, -1.0], &[1.0, 1.0]);
    let deposit = |margin: f32| {
        let slot = GraphCoverSlot::default();
        let scope = slot.arm_boxes("test_input_split", &root, &row, 0.0, false);
        slot.note_box_bound(
            GraphRecordKey {
                objective: &row,
                threshold: 0.0,
                verify_upper: false,
            },
            &root,
            margin,
            margin + 1.0,
        );
        assert!(matches!(
            scope.finish(true, None),
            GraphCoverVerdict::Proved(_)
        ));
    };
    let property = GraphInstanceProperty::new(
        vec![-0.5, -0.5],
        vec![0.5, 0.5],
        vec![vec![GraphObjective::new(&row, 0.0, false).expect("finite")]],
    )
    .expect("well formed");
    let mut tokens = Vec::new();
    for order in [[0.25f32, 0.75, 0.5], [0.5, 0.25, 0.75], [0.75, 0.5, 0.25]] {
        reset_graph_cover_capture();
        for margin in order {
            deposit(margin);
        }
        let capture = take_graph_cover_capture();
        // `>=`: a concurrently running driver test may deposit its own cover
        // (another row, which entails nothing here) while this capture is armed.
        assert!(capture.len() >= 3, "every proved frame is captured");
        let token = authorize_graph_instance(&property, &capture).expect("the clause is entailed");
        tokens.push((token.classes().to_vec(), token.min_margin().to_bits()));
    }
    assert!(
        tokens.iter().all(|t| *t == tokens[0]),
        "the token depends on the deposit order: {tokens:?}"
    );
    assert_eq!(
        tokens[0].1,
        0.75f64.to_bits(),
        "the best entailing cover's margin"
    );
}
