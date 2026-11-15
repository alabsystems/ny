// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests of the trusted facade: the path-based cover gate against the
//! `BranchTree.lean` `safe_of_leaves` shape (a leaf set proves the root iff it
//! is a complete split tree of certified leaves), the counterexamples a Kraft
//! sum accepts, the token's scopes and aggregation, and the margin-row
//! adapters. The source ratchets (profile, sinks, single constructor, math
//! mode, fused multiply-add) live in the crate's section-1.7 integration test.

use super::cover::{CoverLedger, CoverLiteral, CoverRefusal, CoverSite, LeafEvidence, LeafPath};
use super::exact::{
    authorize_scalar_emptiness, scalar_emptiness, ExactRefusal, ScalarAtom, ScalarRel,
    UNSAFE_CONJUNCTION,
};
use super::margin_row::{self as mr, ChildSplit};
use super::token::{
    authorize_unsat, legacy_uncovered_unsat, ratify, AuthorizeRefusal, ClassCover, GuardId,
    GuardReport, LegacyUnsatAuthority, TokenScope, UnsatSource,
};
use super::value::{closed, CertReceipt, Certified, SearchValue};
use crate::margin_row::rounding::RoundMode;

type Lits<'a> = &'a [(CoverSite, bool)];
type Leaf = Vec<(CoverSite, bool)>;

fn tr(layer: u32, pos: u32) -> CoverSite {
    CoverSite::Trunk { layer, pos }
}

fn hd(index: u32) -> CoverSite {
    CoverSite::Head { index }
}

fn path(lits: Lits<'_>) -> Option<LeafPath> {
    LeafPath::from_literals(
        lits.iter()
            .map(|&(site, positive)| CoverLiteral { site, positive })
            .collect(),
    )
}

fn bound(v: f64) -> Option<LeafEvidence> {
    Some(LeafEvidence::Bound(Certified::new(
        v,
        CertReceipt::MarginRowOutward,
    )))
}

fn ledger(leaves: &[Leaf]) -> CoverLedger {
    let mut l = CoverLedger::new();
    for leaf in leaves {
        l.leaf(path(leaf), bound(0.5));
    }
    l
}

const CLASSES: &[usize] = &[1, 2];

#[test]
fn cover_accepts_complete_trees_in_any_split_order() {
    let (x, y) = (tr(0, 0), tr(0, 1));
    let (h0, t01, t23) = (hd(0), tr(0, 1), tr(2, 3));
    let trees: Vec<Vec<Leaf>> = vec![
        vec![vec![]],
        vec![vec![(x, true)], vec![(x, false)]],
        vec![
            vec![(x, true)],
            vec![(x, false), (y, true)],
            vec![(x, false), (y, false)],
        ],
        // Head split at the root, DIFFERENT trunk sites below it: the case a
        // trunk-then-heads canonical order would misread.
        vec![
            vec![(h0, true), (t01, true)],
            vec![(h0, true), (t01, false)],
            vec![(h0, false), (t23, true)],
            vec![(h0, false), (t23, false)],
        ],
        // The same site split on both branches (either can be read as root).
        vec![
            vec![(x, true), (y, true)],
            vec![(x, true), (y, false)],
            vec![(x, false), (y, true)],
            vec![(x, false), (y, false)],
        ],
        // Leaves recorded in no particular order.
        vec![
            vec![(x, false), (y, false)],
            vec![(x, true)],
            vec![(x, false), (y, true)],
        ],
    ];
    for leaves in &trees {
        let l = ledger(leaves);
        let cover = l
            .prove(CLASSES)
            .unwrap_or_else(|e| panic!("complete tree {leaves:?} refused: {e}"));
        assert_eq!(cover.leaves(), leaves.len());
        assert_eq!(cover.classes(), CLASSES);
        assert!(l.kraft_ok(), "Kraft must hold on complete tree {leaves:?}");
        assert_eq!((cover.min_margin() - 0.5).abs(), 0.0);
    }
}

/// JS F1's two counterexamples: a Kraft sum passes both, the path gate
/// refuses both.
#[test]
fn cover_refuses_what_a_kraft_sum_accepts() {
    let (n5, n7, x, y) = (tr(1, 5), tr(1, 7), tr(0, 0), tr(0, 1));
    // Siblings that split DIFFERENT neurons: `n5 on ∧ n7 off` is uncovered.
    let different: Vec<Leaf> = vec![vec![(n5, false)], vec![(n7, true)]];
    let l = ledger(&different);
    assert!(l.kraft_ok(), "the depth multiset {{1,1}} sums to 1");
    assert!(matches!(
        l.prove(CLASSES),
        Err(CoverRefusal::NoCommonSplit { depth: 0, .. })
    ));
    // A duplicated leaf plus a missing sibling at the same depth.
    let dup: Vec<Leaf> = vec![
        vec![(x, true), (y, true)],
        vec![(x, true), (y, true)],
        vec![(x, false)],
    ];
    let l = ledger(&dup);
    assert!(l.kraft_ok(), "1/4 + 1/4 + 1/2 = 1");
    assert!(matches!(
        l.prove(CLASSES),
        Err(CoverRefusal::Overlap { .. })
    ));
}

#[test]
fn cover_refuses_missing_extra_and_overlapping_leaves() {
    let (x, y) = (tr(0, 0), tr(0, 1));
    assert!(matches!(
        CoverLedger::new().prove(CLASSES),
        Err(CoverRefusal::NoLeaves)
    ));
    let cases: Vec<Vec<Leaf>> = vec![
        vec![vec![(x, true)]],
        vec![vec![(x, true)], vec![(x, false), (y, true)]],
        vec![
            vec![(x, true)],
            vec![(x, false)],
            vec![(x, false), (y, true)],
        ],
        vec![vec![], vec![(x, true)]],
        vec![vec![], vec![]],
    ];
    for leaves in &cases {
        assert!(
            !ledger(leaves).proves_cover(CLASSES),
            "non-cover {leaves:?} accepted"
        );
    }
}

#[test]
fn cover_refuses_bad_paths_and_uncertified_leaves() {
    let x = tr(0, 0);
    // One site on both sides is not a path.
    assert!(path(&[(x, true), (x, false)]).is_none());
    // A repeated literal collapses.
    assert_eq!(path(&[(x, true), (x, true)]), path(&[(x, true)]));

    let mut l = CoverLedger::new();
    l.leaf(None, bound(0.5));
    l.leaf(path(&[(x, false)]), bound(0.5));
    assert_eq!(l.prove(CLASSES), Err(CoverRefusal::BadPath { leaf: 0 }));

    let mut l = CoverLedger::new();
    l.leaf(path(&[(x, true)]), None);
    l.leaf(path(&[(x, false)]), bound(0.5));
    assert_eq!(
        l.prove(CLASSES),
        Err(CoverRefusal::UncertifiedLeaf { leaf: 0 })
    );

    // Strict: a zero margin admits a counterexample.
    let mut l = CoverLedger::new();
    l.leaf(path(&[(x, true)]), bound(0.0));
    l.leaf(path(&[(x, false)]), bound(0.5));
    assert_eq!(
        l.prove(CLASSES),
        Err(CoverRefusal::UncertifiedLeaf { leaf: 0 })
    );

    // A bound leaf must carry a bound receipt, an emptiness leaf an emptiness one.
    let mut l = CoverLedger::new();
    l.leaf(
        path(&[(x, true)]),
        Some(LeafEvidence::Bound(Certified::new(
            0.5,
            CertReceipt::MarginRowEmptiness,
        ))),
    );
    l.leaf(path(&[(x, false)]), bound(0.5));
    assert!(!l.proves_cover(CLASSES));
    let mut l = CoverLedger::new();
    l.leaf(
        path(&[(x, true)]),
        Some(LeafEvidence::Empty(Certified::new(
            0.5,
            CertReceipt::MarginRowEmptiness,
        ))),
    );
    l.leaf(path(&[(x, false)]), bound(0.25));
    let cover = l.prove(CLASSES).expect("emptiness leaf plus bound leaf");
    assert_eq!(cover.empty_leaves(), 1);
    assert_eq!(cover.min_margin().to_bits(), 0.25f64.to_bits());
}

/// Deterministic xorshift, so the property test needs no RNG crate API.
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

    fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n.max(1)
    }
}

fn random_tree(rng: &mut XorShift) -> Vec<Leaf> {
    let pool: Vec<CoverSite> = (0..6u32)
        .map(|i| tr(i % 3, i))
        .chain((0..4u32).map(hd))
        .collect();
    let mut leaves = Vec::new();
    let mut stack: Vec<Leaf> = vec![Vec::new()];
    while let Some(node) = stack.pop() {
        let free: Vec<CoverSite> = pool
            .iter()
            .copied()
            .filter(|s| !node.iter().any(|(t, _)| t == s))
            .collect();
        let split = node.len() < 6 && !free.is_empty() && rng.below(3) != 0;
        if !split {
            leaves.push(node);
            continue;
        }
        let site = free[usize::try_from(rng.below(free.len() as u64)).expect("small")];
        for positive in [true, false] {
            let mut child = node.clone();
            child.push((site, positive));
            stack.push(child);
        }
    }
    leaves
}

/// Property: every random split tree (random sites, trunk and head mixed,
/// different sites on different branches, leaves in DFS order) is accepted,
/// and each of three mutations — drop a leaf, duplicate a leaf, flip one
/// literal of one leaf — is refused.
#[test]
fn cover_property_random_split_trees_accept_and_mutations_refuse() {
    let mut rng = XorShift(0x9E37_79B9_7F4A_7C15);
    for case in 0..300 {
        let leaves = random_tree(&mut rng);
        assert!(
            ledger(&leaves).proves_cover(CLASSES),
            "case {case}: complete tree refused: {leaves:?}"
        );
        if leaves.len() < 2 {
            continue;
        }
        let pick = usize::try_from(rng.below(leaves.len() as u64)).expect("small");

        let mut dropped = leaves.clone();
        dropped.remove(pick);
        assert!(
            !ledger(&dropped).proves_cover(CLASSES),
            "case {case}: dropping leaf {pick} still covered"
        );

        let mut duplicated = leaves.clone();
        duplicated.push(leaves[pick].clone());
        assert!(
            !ledger(&duplicated).proves_cover(CLASSES),
            "case {case}: duplicating leaf {pick} still covered"
        );

        let mut flipped = leaves;
        let lit = usize::try_from(rng.below(flipped[pick].len() as u64)).expect("small");
        flipped[pick][lit].1 = !flipped[pick][lit].1;
        assert!(
            !ledger(&flipped).proves_cover(CLASSES),
            "case {case}: flipping literal {lit} of leaf {pick} still covered"
        );
    }
}

#[test]
fn subtree_leaf_requires_a_subdomain_token_for_every_class() {
    let root_cover = mr::root_class(RoundMode::Outward, 1, 0.5).expect("closes");
    let root_cover2 = mr::root_class(RoundMode::Outward, 2, 0.75).expect("closes");
    let guards = GuardReport::new();
    let nested = authorize_unsat(
        TokenScope::Subdomain,
        &[1, 2],
        &[root_cover.clone(), root_cover2.clone()],
        &guards,
    )
    .expect("nested frame closes at its root");
    let mut l = CoverLedger::new();
    l.leaf(Some(LeafPath::root()), Some(LeafEvidence::Subtree(nested)));
    let cover = l.prove(&[1, 2]).expect("subtree covers its frame");
    assert_eq!(cover.subtree_leaves(), 1);
    assert_eq!(cover.min_margin().to_bits(), 0.5f64.to_bits());
    assert_eq!(
        l.prove(&[1, 2, 3]),
        Err(CoverRefusal::SubtreeMismatch { leaf: 0 })
    );

    // An instance-scoped token is not a sub-domain proof.
    let instance = authorize_unsat(
        TokenScope::Instance,
        &[1, 2],
        &[root_cover, root_cover2],
        &guards,
    )
    .expect("instance");
    let mut l = CoverLedger::new();
    l.leaf(
        Some(LeafPath::root()),
        Some(LeafEvidence::Subtree(instance)),
    );
    assert_eq!(
        l.prove(&[1, 2]),
        Err(CoverRefusal::SubtreeMismatch { leaf: 0 })
    );
}

#[test]
fn unsat_token_requires_every_class_and_clear_guards() {
    let guards = GuardReport::new();
    let r1 = mr::root_class(RoundMode::Outward, 1, 0.3).expect("closes");
    let tree = ledger(&[vec![(tr(0, 0), true)], vec![(tr(0, 0), false)]])
        .prove(&[2])
        .expect("cover");
    let token = authorize_unsat(
        TokenScope::Instance,
        &[2, 1],
        &[r1.clone(), ClassCover::Tree(tree.clone())],
        &guards,
    )
    .expect("every class covered");
    assert!(token.publishable());
    assert_eq!(token.source(), UnsatSource::Authorized);
    assert_eq!(token.classes(), &[1, 2]);
    assert_eq!(token.min_margin().to_bits(), 0.3f64.to_bits());

    assert_eq!(
        authorize_unsat(TokenScope::Instance, &[1, 2, 3], &[r1.clone()], &guards),
        Err(AuthorizeRefusal::Uncovered { class: 2 })
    );
    assert_eq!(
        authorize_unsat(TokenScope::Instance, &[], &[r1.clone()], &guards),
        Err(AuthorizeRefusal::NoClasses)
    );
    let mut tripped = GuardReport::new();
    tripped.trip(GuardId::Containment, "box excludes sample 3");
    assert!(matches!(
        authorize_unsat(TokenScope::Instance, &[1], &[r1.clone()], &tripped),
        Err(AuthorizeRefusal::GuardTripped(_))
    ));
    // A root bound that does not close covers nothing.
    assert!(mr::root_class(RoundMode::Outward, 1, 0.0).is_none());
    assert!(mr::root_class(RoundMode::Parity, 1, 0.3).is_none());

    // Class-part tokens aggregate; a sub-domain token is not a class part.
    let part = authorize_unsat(
        TokenScope::ClassPart,
        &[2],
        &[ClassCover::Tree(tree)],
        &guards,
    )
    .expect("class part");
    assert!(!part.publishable());
    let whole = authorize_unsat(
        TokenScope::Instance,
        &[1, 2],
        &[r1.clone(), ClassCover::Part(part)],
        &guards,
    )
    .expect("aggregate");
    assert!(whole.publishable());
    let sub = authorize_unsat(TokenScope::Subdomain, &[2], &[r1], &guards);
    assert!(
        sub.is_err(),
        "class 2 is not covered by a class-1 root close"
    );

    // ratify is downgrade-only.
    assert!(ratify(whole.clone(), &guards).is_ok());
    assert!(ratify(whole, &tripped).is_err());
}

/// Only an instance-scoped token may reach RESULTS_FILE (`ny-cli`'s
/// `UnsatReceipt::from_token` publishes exactly `publishable()`): the SAME
/// covers authorized at class-part or sub-domain scope are not publishable.
/// Pinned here because no other crate can mint these tokens.
#[test]
fn unsat_token_scopes_publish_only_at_instance() {
    let guards = GuardReport::new();
    let cover = mr::root_class(RoundMode::Outward, 1, 0.5).expect("closes");
    for (scope, publishable) in [
        (TokenScope::Instance, true),
        (TokenScope::ClassPart, false),
        (TokenScope::Subdomain, false),
    ] {
        let token = authorize_unsat(scope, &[1], std::slice::from_ref(&cover), &guards)
            .expect("the class is covered");
        assert_eq!(token.scope(), scope);
        assert_eq!(token.source(), UnsatSource::Authorized);
        assert_eq!(token.publishable(), publishable, "{scope:?}");
    }
}

#[test]
fn unsat_token_scope_fails_closed_by_default() {
    assert_eq!(TokenScope::default(), TokenScope::Subdomain);
    let legacy = legacy_uncovered_unsat(LegacyUnsatAuthority::TestFixture);
    assert!(legacy.publishable());
    assert_eq!(
        legacy.source(),
        UnsatSource::LegacyUncovered(LegacyUnsatAuthority::TestFixture)
    );
    assert!(legacy.min_margin().is_nan());
}

#[test]
fn closed_is_strict_finite_and_mode_free() {
    let c = |v: f64| Certified::new(v, CertReceipt::MarginRowOutward);
    assert!(!closed(&c(0.0)));
    assert!(!closed(&c(-0.0)));
    assert!(closed(&c(f64::from_bits(1))));
    assert!(closed(&c(1e-300)));
    assert!(!closed(&c(-1e-300)));
    assert!(!closed(&c(f64::INFINITY)));
    assert!(!closed(&c(f64::NEG_INFINITY)));
    assert!(!closed(&c(f64::NAN)));
    assert_eq!(SearchValue::new(0.5).hint().to_bits(), 0.5f64.to_bits());
}

/// The margin-row closure decision in Outward mode is bit-for-bit the
/// historical predicate (`b.is_finite() && b > 0.0`), so no Outward verdict can
/// move; the historical Parity branch (`b >= 0.0`) is gone.
#[test]
fn margin_row_closes_matches_the_historical_outward_predicate() {
    let mut values = vec![
        0.0,
        -0.0,
        f64::from_bits(1),
        -f64::from_bits(1),
        f64::MIN_POSITIVE,
        1e-300,
        1e-12,
        0.5,
        -0.5,
        f64::MAX,
        f64::MIN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
    ];
    let mut rng = XorShift(0x0123_4567_89AB_CDEF);
    for _ in 0..10_000 {
        values.push(f64::from_bits(rng.next_u64()));
    }
    for v in values {
        let historical_outward = v.is_finite() && v > 0.0;
        assert_eq!(
            mr::closes(RoundMode::Outward, v),
            historical_outward,
            "{v:e}"
        );
        assert_eq!(
            mr::bound_evidence(RoundMode::Outward, v).is_some(),
            historical_outward,
            "{v:e}"
        );
        assert!(!mr::closes(RoundMode::Parity, v), "parity closed {v:e}");
        assert!(mr::bound_evidence(RoundMode::Parity, v).is_none());
        assert_eq!(
            mr::empty_evidence(RoundMode::Outward, v).is_some(),
            historical_outward,
            "{v:e}"
        );
    }
}

#[test]
fn leaf_path_is_relative_to_the_frame_root_heads() {
    let trunk = [(0usize, 1usize, 1i8), (2, 0, -1)];
    let heads = [(5usize, 1i8), (7, -1)];
    let p = mr::leaf_path(&trunk, &heads, &[(5, 1)], None).expect("path");
    assert_eq!(
        p.literals(),
        path(&[(tr(0, 1), true), (tr(2, 0), false), (hd(7), false)])
            .expect("expected")
            .literals()
    );
    let child = mr::leaf_path(&trunk, &heads, &[(5, 1)], Some(ChildSplit::Head((9, 1))))
        .expect("child path");
    assert_eq!(child.depth(), 4);
    let trunk_child = mr::leaf_path(&trunk, &heads, &[], Some(ChildSplit::Trunk((3, 3, -1))))
        .expect("trunk child");
    assert_eq!(trunk_child.depth(), 5);
    // The frame root's heads must be a prefix.
    assert!(mr::leaf_path(&trunk, &heads, &[(7, -1)], None).is_none());
    assert!(mr::leaf_path(&trunk, &[], &[(5, 1)], None).is_none());
    // A split without a side is not a literal.
    assert!(mr::leaf_path(&[(0, 1, 0)], &[], &[], None).is_none());
    // The child re-splitting a fixed site on the other side is refused.
    assert!(mr::leaf_path(&trunk, &heads, &[], Some(ChildSplit::Trunk((0, 1, -1)))).is_none());
}

#[test]
fn frozen_dual_class_rechecks_the_level_ledger() {
    use crate::margin_row::zono_bab::{
        ClassReport, ClassRun, FrozenDualStateMode, LevelTrace, OpenReason, SearchReport,
        SearchVerdict,
    };
    let report =
        |verdict: SearchVerdict, levels: Vec<LevelTrace>, root: f64, slack: f64| ClassReport {
            class: 4,
            lane_root: -0.1,
            result: Ok(ClassRun {
                state_gens: 1,
                unstable: 1,
                depth_cap: 1,
                build_ms: 0.0,
                state_bytes: 0,
                build_estimate_bytes: 0,
                build_columns: 0,
                streamed_rows: false,
                root_g: root,
                search: SearchReport {
                    verdict,
                    root_certified: root,
                    cert_min_slack: slack,
                    levels,
                    ..SearchReport::empty()
                },
                // #fd-shared-state: the adapter re-derives the cover from the
                // search alone, whichever way the state was built.
                state: FrozenDualStateMode::Shared,
                root_loss: Some(0.25),
                proof: None,
                progressive: None,
            }),
        };
    let lv = |depth: usize, frontier: u64, pruned: u64| LevelTrace {
        depth,
        frontier,
        pruned,
    };
    // Complete: level 1 has 2 nodes, prunes 1; level 2 has 2, prunes both.
    let good = report(
        SearchVerdict::Closed,
        vec![lv(1, 2, 1), lv(2, 2, 2)],
        -0.2,
        1e-7,
    );
    match mr::frozen_dual_class(&good) {
        Some(ClassCover::FrozenDual { class, cert }) => {
            assert_eq!(class, 4);
            assert_eq!(cert.value().to_bits(), 1e-7f64.to_bits());
            assert_eq!(cert.receipt(), CertReceipt::FrozenDual);
        }
        other => panic!("expected a frozen-dual cover, got {other:?}"),
    }
    // A `Closed` verdict whose ledger does not close is refused.
    let lying = report(
        SearchVerdict::Closed,
        vec![lv(1, 2, 1), lv(2, 1, 1)],
        -0.2,
        1e-7,
    );
    assert!(mr::frozen_dual_class(&lying).is_none());
    // Root-closed (no level): the root's own certified value decides.
    let root_closed = report(SearchVerdict::Closed, Vec::new(), 0.01, 0.01);
    assert!(mr::frozen_dual_class(&root_closed).is_some());
    let root_zero = report(SearchVerdict::Closed, Vec::new(), 0.0, 0.0);
    assert!(mr::frozen_dual_class(&root_zero).is_none());
    // Open is open.
    let open = report(
        SearchVerdict::Open(OpenReason::Deadline),
        vec![lv(1, 2, 1), lv(2, 2, 2)],
        -0.2,
        1e-7,
    );
    assert!(mr::frozen_dual_class(&open).is_none());
}

// ---------------------------------------------------------------------------
// #spine-legacy-lanes: exact specification-emptiness certificates.
// ---------------------------------------------------------------------------

fn atom(rel: ScalarRel, constant: f64) -> ScalarAtom {
    ScalarAtom { rel, constant }
}

/// Every contradicting shape the pair check must accept, and every
/// satisfiable one it must refuse, including the strict tie, the non-strict
/// tie, signed zeros and non-finite constants.
#[test]
fn exact_emptiness_accepts_exactly_the_contradicting_pairs() {
    use ScalarRel::{Ge, Gt, Le, Lt};
    // The isomorphic complement: t > eps and t < -eps.
    let iso = scalar_emptiness(&[atom(Gt, 0.05), atom(Lt, -0.05)]).expect("empty");
    assert_eq!(iso.lower(), atom(Gt, 0.05));
    assert_eq!(iso.upper(), atom(Lt, -0.05));
    assert!(iso.slack() > 0.0 && iso.slack() <= 0.1);
    // Non-strict pair with a gap: t >= 1 and t <= 0.
    assert!(scalar_emptiness(&[atom(Ge, 1.0), atom(Le, 0.0)]).is_some());
    // Ties: empty exactly when a strict atom is involved.
    for (lower, upper, empty) in [
        (Gt, Lt, true),
        (Gt, Le, true),
        (Ge, Lt, true),
        (Ge, Le, false),
    ] {
        let got = scalar_emptiness(&[atom(lower, 0.5), atom(upper, 0.5)]);
        assert_eq!(got.is_some(), empty, "{lower:?} 0.5 vs {upper:?} 0.5");
        if let Some(proof) = got {
            assert_eq!(proof.slack().to_bits(), 0.0f64.to_bits());
        }
    }
    // Signed zeros are the same real number: t > 0 and t < -0 is empty.
    assert!(scalar_emptiness(&[atom(Gt, 0.0), atom(Lt, -0.0)]).is_some());
    // Satisfiable shapes: the crafted wrong-sign region, a single side, two
    // lowers, two uppers.
    assert!(scalar_emptiness(&[atom(Gt, -0.05), atom(Lt, 0.05)]).is_none());
    assert!(scalar_emptiness(&[atom(Gt, 0.05)]).is_none());
    assert!(scalar_emptiness(&[atom(Gt, 5.0), atom(Ge, 7.0)]).is_none());
    assert!(scalar_emptiness(&[atom(Lt, 5.0), atom(Le, -7.0)]).is_none());
    assert!(scalar_emptiness(&[]).is_none());
    // Any non-finite constant refuses the whole group (fail closed), even
    // beside a genuinely contradicting pair.
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(
            scalar_emptiness(&[atom(Gt, 1.0), atom(Lt, 0.0), atom(Le, bad)]).is_none(),
            "{bad}"
        );
    }
    // The best pair is kept: the largest slack wins.
    let best = scalar_emptiness(&[atom(Gt, 0.25), atom(Ge, 2.0), atom(Lt, 0.0), atom(Le, -1.0)])
        .expect("empty");
    assert_eq!(best.lower(), atom(Ge, 2.0));
    assert_eq!(best.upper(), atom(Le, -1.0));
    assert!(best.slack() > 2.9 && best.slack() <= 3.0);
    // An overflowing difference reports f64::MAX, never inf or NaN.
    let wide = scalar_emptiness(&[atom(Ge, f64::MAX), atom(Le, -f64::MAX)]).expect("empty");
    assert_eq!(wide.slack().to_bits(), f64::MAX.to_bits());
}

/// The pair check against brute force: a group of atoms on a dyadic grid is
/// satisfiable iff some candidate point (every constant, every midpoint
/// between consecutive constants, and one point beyond each end) satisfies
/// every atom, because the solution set is an interval whose ends are
/// constants. `scalar_emptiness` must return `Some` exactly for the empty
/// groups — soundness AND completeness — on 20,000 random groups.
#[test]
fn exact_emptiness_agrees_with_a_brute_force_witness_search() {
    const RELS: [ScalarRel; 4] = [ScalarRel::Lt, ScalarRel::Le, ScalarRel::Gt, ScalarRel::Ge];
    let holds = |a: ScalarAtom, t: f64| match a.rel {
        ScalarRel::Lt => t < a.constant,
        ScalarRel::Le => t <= a.constant,
        ScalarRel::Gt => t > a.constant,
        ScalarRel::Ge => t >= a.constant,
    };
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        state >> 33
    };
    let (mut empty_seen, mut feasible_seen) = (0usize, 0usize);
    for _ in 0..20_000 {
        let n = 1 + (next() % 5) as usize;
        let atoms: Vec<ScalarAtom> = (0..n)
            .map(|_| {
                let rel = RELS[(next() % 4) as usize];
                // Constants on a 0.5 grid in [-2, 2], so every midpoint is exact.
                let constant = f64::from(i32::try_from(next() % 9).expect("small")) * 0.5 - 2.0;
                atom(rel, constant)
            })
            .collect();
        let mut consts: Vec<f64> = atoms.iter().map(|a| a.constant).collect();
        consts.sort_by(f64::total_cmp);
        consts.dedup();
        let mut candidates = consts.clone();
        candidates.extend(consts.windows(2).map(|w| f64::midpoint(w[0], w[1])));
        candidates.push(consts[0] - 1.0);
        candidates.push(consts[consts.len() - 1] + 1.0);
        let feasible = candidates
            .iter()
            .any(|&t| atoms.iter().all(|&a| holds(a, t)));
        let proof = scalar_emptiness(&atoms);
        assert_eq!(proof.is_none(), feasible, "atoms {atoms:?}");
        if let Some(proof) = proof {
            // The certificate's own pair is contradictory at every candidate.
            assert!(candidates
                .iter()
                .all(|&t| !(holds(proof.lower(), t) && holds(proof.upper(), t))));
            empty_seen += 1;
        } else {
            feasible_seen += 1;
        }
    }
    assert!(
        empty_seen > 1_000 && feasible_seen > 1_000,
        "the generator must exercise both outcomes: empty={empty_seen} feasible={feasible_seen}"
    );
}

/// The verifying entry: every group re-checked, instance scope, one class,
/// the LARGEST group slack as G5 (every group covers the one class, and
/// `authorize_unsat` keeps a class's best cover), and a named refusal for each
/// failure.
#[test]
fn exact_emptiness_token_requires_every_group_and_is_instance_scoped() {
    use ScalarRel::{Gt, Lt};
    let iso = |eps: f64| vec![atom(Gt, eps), atom(Lt, -eps)];
    let token = authorize_scalar_emptiness(&[iso(0.5), iso(0.25)]).expect("both groups empty");
    assert!(token.publishable());
    assert_eq!(token.scope(), TokenScope::Instance);
    assert_eq!(token.source(), UnsatSource::Authorized);
    assert_eq!(token.classes(), &[UNSAFE_CONJUNCTION]);
    // Aggregation keeps the best cover per class: here the larger slack.
    assert!(token.min_margin() > 0.9 && token.min_margin() <= 1.0);

    assert_eq!(authorize_scalar_emptiness(&[]), Err(ExactRefusal::NoGroups));
    assert_eq!(
        authorize_scalar_emptiness(&[iso(0.5), Vec::new()]),
        Err(ExactRefusal::EmptyGroup { group: 1 })
    );
    assert_eq!(
        authorize_scalar_emptiness(&[iso(0.5), iso(-0.5)]),
        Err(ExactRefusal::Feasible { group: 1 })
    );
    assert_eq!(
        authorize_scalar_emptiness(&[vec![atom(Gt, f64::NAN), atom(Lt, 0.0)]]),
        Err(ExactRefusal::NonFinite { group: 0 })
    );
    // The cover it builds is the facade's own: a feasible group can never be
    // smuggled in beside an empty one.
    assert!(authorize_scalar_emptiness(&[iso(0.5), vec![atom(Gt, 0.0)]]).is_err());
}

/// The legacy-uncovered census is a closed list: every variant appears once in
/// `ALL`, with a unique token. The match below is exhaustive, so a new variant
/// fails to compile here until it is added to the census (and to `ny-cli`'s
/// door manifest, whose ceiling may only go down).
#[test]
fn legacy_uncovered_census_lists_every_authority_once() {
    fn listed(a: LegacyUnsatAuthority) -> bool {
        match a {
            LegacyUnsatAuthority::BetaCrownVerdict
            | LegacyUnsatAuthority::CzHybridLane
            | LegacyUnsatAuthority::RelationalImplication
            | LegacyUnsatAuthority::NonlinearIntervalBab
            | LegacyUnsatAuthority::CctsdbSubbox
            | LegacyUnsatAuthority::TllStructure
            | LegacyUnsatAuthority::TestFixture => LegacyUnsatAuthority::ALL.contains(&a),
        }
    }
    let mut tokens: Vec<&str> = LegacyUnsatAuthority::ALL
        .iter()
        .map(|a| a.token())
        .collect();
    assert!(LegacyUnsatAuthority::ALL.iter().all(|&a| listed(a)));
    tokens.sort_unstable();
    tokens.dedup();
    assert_eq!(tokens.len(), LegacyUnsatAuthority::ALL.len());
    for a in LegacyUnsatAuthority::ALL {
        let token = legacy_uncovered_unsat(a);
        assert_eq!(token.source(), UnsatSource::LegacyUncovered(a));
        assert_eq!(token.source().token(), a.token());
        assert!(token.classes().is_empty() && token.min_margin().is_nan());
    }
}

/// ENGINE_PROGRAM §1.7 B1, the certify half of the proof-hint stream: a
/// PROVED ledger streams EVERY recorded leaf, in recording order, with its
/// canonical path, its close kind (bound / empty / nested-subtree) and the
/// certified margin it closed with (the subtree's is its token's minimum), into
/// any sink — and, under hint capture, every recorded leaf also becomes a v2
/// NODE (absolute ONNX-level literals, even when its hint was never pushed).
/// Streaming is write-only: the gate's answer over the same ledger is
/// unchanged afterwards and nothing closes the stream but the entry point.
/// The ny-cli bundle test (`proof_bundle_streams_leaves_and_finalises_
/// only_after_an_unsat`) covers the other half, the writer.
#[test]
fn proof_sink_streams_every_recorded_leaf_and_moves_no_cover() {
    use super::sink::{
        force_hint_capture_for_test, stream_cover_into, HintLedger, LeafClose, NodeClose, NullSink,
        RecordingSink, SiteMap,
    };

    let (x, y) = (tr(0, 0), tr(1, 2));
    let guards = GuardReport::new();
    let nested = authorize_unsat(
        TokenScope::Subdomain,
        CLASSES,
        &[
            mr::root_class(RoundMode::Outward, 1, 0.5).expect("closes"),
            mr::root_class(RoundMode::Outward, 2, 0.75).expect("closes"),
        ],
        &guards,
    )
    .expect("nested frame closes at its root");
    let nested_min = nested.min_margin();
    let mut l = CoverLedger::new();
    l.leaf(path(&[(x, true)]), bound(0.5));
    l.leaf(
        path(&[(y, true), (x, false)]),
        Some(LeafEvidence::Empty(Certified::new(
            0.25,
            CertReceipt::MarginRowEmptiness,
        ))),
    );
    l.leaf(
        path(&[(x, false), (y, false)]),
        Some(LeafEvidence::Subtree(nested)),
    );
    let before = l
        .prove(CLASSES)
        .expect("a complete tree of certified leaves");

    let mut rec = RecordingSink::default();
    assert_eq!(stream_cover_into(&l, &mut rec), 3, "one record per leaf");
    let got: Vec<(Vec<(CoverSite, bool)>, LeafClose, u64)> = rec
        .leaves
        .iter()
        .map(|r| {
            (
                r.path
                    .literals()
                    .iter()
                    .map(|lit| (lit.site, lit.positive))
                    .collect(),
                r.close,
                r.margin.to_bits(),
            )
        })
        .collect();
    assert_eq!(
        got,
        vec![
            (vec![(x, true)], LeafClose::Bound, 0.5f64.to_bits()),
            // Canonical (site-sorted) path, whatever order it was built in.
            (
                vec![(x, false), (y, true)],
                LeafClose::Empty,
                0.25f64.to_bits()
            ),
            (
                vec![(x, false), (y, false)],
                LeafClose::Subtree,
                nested_min.to_bits()
            ),
        ],
        "records must be the ledger's leaves, in recording order"
    );
    assert_eq!(nested_min.to_bits(), 0.5f64.to_bits());
    assert!(
        rec.verdict.is_none(),
        "streaming leaves never closes the stream (the entry point does, after publication)"
    );
    assert_eq!(
        [LeafClose::Bound, LeafClose::Empty, LeafClose::Subtree].map(LeafClose::token),
        ["bound", "empty", "subtree"]
    );
    // The Competition sink consumes the identical stream and keeps nothing.
    assert_eq!(stream_cover_into(&l, &mut NullSink), 3);

    // v2: under capture every recorded leaf becomes a node with ABSOLUTE
    // literals (trunk layer 0 -> Relu op 3 of unstable list [40]; layer 1 ->
    // op 6 of [50, 51, 52]); these leaves pushed no hint, so they are
    // streamed from their paths and claim the frame's classes.
    force_hint_capture_for_test(Some(true));
    let hints = HintLedger::new();
    force_hint_capture_for_test(None);
    let sites = SiteMap::new(vec![(3, vec![40]), (6, vec![50, 51, 52])], 9, Vec::new());
    let mut v2 = RecordingSink::default();
    hints.stream_into(&l, CLASSES, &sites, &mut v2);
    assert_eq!(v2.nodes.len(), 3, "one node per proved leaf");
    let kinds: Vec<NodeClose> = v2.nodes.iter().map(|n| n.close).collect();
    assert_eq!(
        kinds,
        [NodeClose::Bound, NodeClose::Empty, NodeClose::Subtree]
    );
    assert_eq!(v2.nodes[0].lits.len(), 1);
    assert_eq!(
        (v2.nodes[0].lits[0].site.op, v2.nodes[0].lits[0].site.index),
        (3, 40)
    );
    assert_eq!(v2.nodes[0].dis.as_deref(), Some(CLASSES));
    assert_eq!(
        v2.nodes[1].dis, None,
        "an emptiness node claims every clause"
    );
    assert_eq!(
        v2.nodes[2].dis.as_deref(),
        Some(&[][..]),
        "a subtree node claims nothing"
    );
    assert!(v2.nodes.iter().all(|n| n.beta.len() == n.lits.len()));

    // Write-only: the gate's answer over the same ledger is unchanged.
    let after = l.prove(CLASSES).expect("streaming cannot un-prove a cover");
    let shape = |c: &super::cover::TreeCover| {
        (
            c.classes().to_vec(),
            c.leaves(),
            c.max_depth(),
            c.empty_leaves(),
            c.subtree_leaves(),
            c.min_margin().to_bits(),
        )
    };
    assert_eq!(shape(&after), shape(&before));
    assert_eq!(l.recorded(), 3);
}

/// #graph-bab-cover-ledger x B2.1: the v2 emitter never states a checkable
/// certificate for a graph-lane leaf. A graph-site path has no translation in
/// a margin-row frame's site map, and a margin-free `Lane` close has no bound
/// the bundle can carry, so both stream as `opaque` nodes claiming every class
/// (the offline checker can only report those claims unconfirmed).
#[test]
fn graph_lane_leaves_stream_as_opaque_v2_nodes() {
    use super::sink::{force_hint_capture_for_test, HintLedger, NodeClose, RecordingSink, SiteMap};

    let g = CoverSite::GraphRelu { node: 2, neuron: 5 };
    let p = CoverSite::GraphPoint {
        node: 4,
        input: 0,
        neuron: 1,
        point_bits: 0.5f32.to_bits(),
    };
    let mut l = CoverLedger::new();
    // A graph-site path with a certified bound: untranslatable.
    l.leaf(path(&[(g, true)]), bound(0.5));
    // A graph-site path with a margin-free lane close.
    l.leaf(
        path(&[(g, false), (p, true)]),
        Some(LeafEvidence::LaneClose(CertReceipt::GraphLaneClip)),
    );
    // A translatable margin-row path whose evidence is a lane close.
    l.leaf(
        path(&[(tr(0, 0), true)]),
        Some(LeafEvidence::LaneClose(CertReceipt::GraphLaneInfeasible)),
    );
    force_hint_capture_for_test(Some(true));
    let hints = HintLedger::new();
    force_hint_capture_for_test(None);
    let sites = SiteMap::new(vec![(3, vec![40])], 9, Vec::new());
    let mut v2 = RecordingSink::default();
    hints.stream_into(&l, CLASSES, &sites, &mut v2);
    assert_eq!(v2.nodes.len(), 3, "every recorded leaf still gets a node");
    for n in &v2.nodes {
        assert_eq!(n.close, NodeClose::Opaque, "{n:?}");
        assert_eq!(n.dis.as_deref(), Some(CLASSES), "{n:?}");
        assert!(n.margin.is_nan(), "{n:?}");
    }
    assert!(v2.nodes[0].lits.is_empty() && v2.nodes[1].lits.is_empty());
    assert_eq!(v2.nodes[2].lits.len(), 1);
    assert_eq!(
        (v2.nodes[2].lits[0].site.op, v2.nodes[2].lits[0].site.index),
        (3, 40)
    );
}

/// #graph-bab-cover-ledger x B2.1, the HINTED arm (the test above reaches only
/// the unhinted cover-record arm): a leaf hint with a margin-free `Lane` close
/// streams as an `opaque` node claiming EVERY class, not just the hint's own
/// `dis`, keeping its translated literals and β; the cover record it names is
/// not streamed a second time.
#[test]
fn graph_lane_hinted_leaf_streams_as_opaque_claiming_every_class() {
    use super::sink::{
        force_hint_capture_for_test, FrameSplit, HintLedger, LeafClose, LeafHint, NodeClose,
        RecordingSink, SiteMap,
    };

    let mut l = CoverLedger::new();
    l.leaf(
        path(&[(tr(0, 0), true)]),
        Some(LeafEvidence::LaneClose(CertReceipt::GraphLaneInfeasible)),
    );
    force_hint_capture_for_test(Some(true));
    let mut hints = HintLedger::new();
    force_hint_capture_for_test(None);
    assert!(
        hints.capturing(),
        "the ledger must capture, or the test is VOID"
    );
    hints.push_leaf(LeafHint {
        splits: vec![FrameSplit::Trunk {
            layer: 0,
            pos: 0,
            positive: true,
        }],
        beta: vec![0.25],
        close: LeafClose::Lane,
        dis: vec![CLASSES[0]],
        margin: f64::NAN,
        lift: false,
        empty_split: None,
    });
    let sites = SiteMap::new(vec![(3, vec![40])], 9, Vec::new());
    let mut v2 = RecordingSink::default();
    hints.stream_into(&l, CLASSES, &sites, &mut v2);
    assert_eq!(
        v2.nodes.len(),
        1,
        "the hinted leaf's cover record is not re-streamed: {:?}",
        v2.nodes
    );
    let n = &v2.nodes[0];
    assert_eq!(n.close, NodeClose::Opaque, "{n:?}");
    assert_eq!(n.src, "tree", "{n:?}");
    assert_eq!(
        n.dis.as_deref(),
        Some(CLASSES),
        "a lane close claims every class: {n:?}"
    );
    assert_eq!(n.lits.len(), 1, "{n:?}");
    assert_eq!(
        (n.lits[0].site.op, n.lits[0].site.index, n.lits[0].positive),
        (3, 40, true)
    );
    assert_eq!(n.beta, vec![0.25]);
    assert!(n.margin.is_nan(), "{n:?}");
}

/// review(b2-fd): a wave leaf's OWN slope table is bounded per frame — the
/// ledger keeps tables until `LEAF_ALPHA_TABLE_BYTES` (lowered here), and a
/// leaf past the ceiling still streams, as a plain leaf under the frame's
/// root α table (no own `hint.alpha` record). Under the default ceiling both
/// leaves keep their own tables. Without the ceiling the ledger grows as
/// leaves × searched sites with no admission.
#[test]
fn leaf_alpha_tables_fall_back_to_the_frame_table_past_the_ledger_ceiling() {
    use super::sink::{
        force_hint_capture_for_test, FrameSplit, HintLedger, LeafClose, LeafHint, NodeClose,
        RecordingSink, SiteMap,
    };

    let x = tr(0, 0);
    let mut l = CoverLedger::new();
    l.leaf(path(&[(x, true)]), bound(0.5));
    l.leaf(path(&[(x, false)]), bound(0.25));
    let leaf = |positive: bool, margin: f64| LeafHint {
        splits: vec![FrameSplit::Trunk {
            layer: 0,
            pos: 0,
            positive,
        }],
        beta: vec![0.0],
        close: LeafClose::Bound,
        dis: vec![CLASSES[0]],
        margin,
        lift: false,
        empty_split: None,
    };
    let table = || vec![(3u32, vec![40u32], vec![0.5f64])];
    let sites = SiteMap::new(vec![(3, vec![40])], 9, Vec::new());
    let run = |cap: Option<u64>| {
        force_hint_capture_for_test(Some(true));
        let mut hints = HintLedger::new();
        force_hint_capture_for_test(None);
        assert!(hints.capturing(), "the ledger must capture: VOID");
        if let Some(cap) = cap {
            hints.set_leaf_alpha_cap_for_test(cap);
        }
        hints.push_leaf_with_alpha(leaf(true, 0.5), table());
        hints.push_leaf_with_alpha(leaf(false, 0.25), table());
        let mut v2 = RecordingSink::default();
        hints.stream_into(&l, CLASSES, &sites, &mut v2);
        v2
    };
    // Default ceiling: both leaves stream their own table.
    let full = run(None);
    assert_eq!(full.nodes.len(), 2, "{:?}", full.nodes);
    assert_eq!(full.alphas.len(), 2, "one own table per leaf");
    assert!(full.nodes.iter().all(|n| n.alpha.is_some()));
    // One table's worth: the second leaf falls back, and still streams.
    let capped = run(Some(80));
    assert_eq!(
        capped.nodes.len(),
        2,
        "a leaf past the ceiling still streams"
    );
    assert_eq!(
        capped.alphas.len(),
        1,
        "only the first leaf keeps its table"
    );
    assert!(capped.nodes[0].alpha.is_some());
    assert_eq!(
        capped.nodes[1].alpha, None,
        "past the ceiling: the frame's root table (none here)"
    );
    assert!(capped
        .nodes
        .iter()
        .all(|n| n.close == NodeClose::Bound && n.dis.as_deref() == Some(&CLASSES[..1])));
}

/// ENGINE_PROGRAM §1.7 B1: hint capture is OFF unless the entry point's Proof
/// arm switched it on — no ledger in a process without a bundle sink captures
/// (and every push is then a no-op before anything is built).
#[test]
fn hint_capture_is_off_without_a_bundle_sink() {
    use super::sink::{hint_capture_enabled, proof_sink_installed, HintLedger};
    assert!(
        !proof_sink_installed(),
        "no ny-propagate test installs a process sink"
    );
    assert!(!hint_capture_enabled());
    let l = HintLedger::new();
    assert!(!l.capturing());
    assert!(!HintLedger::default().capturing());
}

/// review(eng14): the claim of a route that publishes `unsat` OUTSIDE every
/// frame (`stream_opaque_claim`, the dense input-head route's emitter) is
/// exactly ONE `opaque` node over the WHOLE region (no literals) carrying the
/// route token and the token's classes; it streams only while capturing, and
/// an empty class list streams nothing. Mutation targets: the capture test,
/// the `dis`, the `src`, the empty-region literals.
#[test]
fn stream_opaque_claim_is_one_whole_region_opaque_node_only_while_capturing() {
    use super::sink::{
        force_hint_capture_for_test, stream_opaque_claim, with_thread_sink, AlphaRecord,
        FdStateRecord, LeafRecord, NodeClose, NodeRecord, ProofSink, SitesRecord, VerdictRecord,
    };
    use std::sync::{Arc, Mutex};

    #[derive(Clone, Default)]
    struct Shared(Arc<Mutex<Vec<NodeRecord>>>);
    impl ProofSink for Shared {
        fn leaf(&mut self, _leaf: &LeafRecord) {}
        fn node(&mut self, node: &NodeRecord) {
            self.0.lock().unwrap().push(node.clone());
        }
        fn alpha(&mut self, _alpha: &AlphaRecord) {}
        fn sites(&mut self, _sites: &SitesRecord) {}
        fn fd_state(&mut self, _state: &FdStateRecord) {}
        fn finish(&mut self, _verdict: &VerdictRecord) {}
    }

    let run = |capture: bool, classes: Vec<usize>| {
        let sink = Shared::default();
        force_hint_capture_for_test(Some(capture));
        let ((), _) = with_thread_sink(Box::new(sink.clone()), || {
            stream_opaque_claim(classes, "input_head");
        });
        force_hint_capture_for_test(None);
        let nodes = sink.0.lock().unwrap().clone();
        nodes
    };
    assert!(
        run(false, vec![1]).is_empty(),
        "capture off streams nothing"
    );
    assert!(run(true, Vec::new()).is_empty(), "no class, no claim");
    let nodes = run(true, vec![1]);
    assert_eq!(nodes.len(), 1, "{nodes:?}");
    let n = &nodes[0];
    assert_eq!(n.close, NodeClose::Opaque, "{n:?}");
    assert_eq!(n.src, "input_head", "{n:?}");
    assert_eq!(n.dis.as_deref(), Some(&[1usize][..]), "{n:?}");
    assert!(
        n.lits.is_empty() && n.xlits.is_empty(),
        "whole region: {n:?}"
    );
    assert_eq!(n.region, 0, "{n:?}");
    assert!(
        n.margin.is_nan() && n.beta.is_empty() && n.alpha.is_none(),
        "{n:?}"
    );
}
