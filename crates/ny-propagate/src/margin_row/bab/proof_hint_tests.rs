// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Proof bundle v2 emitter tests (ENGINE_PROGRAM §1.7 B1, B2.1 brief §3):
//! the margin-row lane's leaf, retirement and pre-cover hints.

use std::sync::{Arc, Mutex};

use super::{
    head_raised_classes, head_route_pre_hint, hint_splits, pre_hint, proof_pre_hint, HeadFix,
    HeadRaised, TrunkSplit,
};
use crate::certify::margin_row::{self as cert_mr, ChildSplit};
use crate::certify::{
    force_hint_capture_for_test, AbsLit, AbsSite, AlphaRecord, ClassCover, CoverLedger, FrameSplit,
    HintLedger, LeafClose, LeafHint, LeafRecord, NodeClose, NodeRecord, PreHint, ProofSink,
    RetireHint, SiteMap, SitesRecord, VerdictRecord,
};
use crate::margin_row::rounding::RoundMode;

#[derive(Clone, Default)]
struct Shared(Arc<Mutex<Vec<NodeRecord>>>);

impl ProofSink for Shared {
    fn leaf(&mut self, _leaf: &LeafRecord) {}

    fn node(&mut self, node: &NodeRecord) {
        if let Ok(mut v) = self.0.lock() {
            v.push(node.clone());
        }
    }

    fn alpha(&mut self, _alpha: &AlphaRecord) {}

    fn sites(&mut self, _sites: &SitesRecord) {}

    fn fd_state(&mut self, _state: &crate::certify::FdStateRecord) {}

    fn finish(&mut self, _verdict: &VerdictRecord) {}
}

/// Two trunk layers (ops 1 and 4, unstable lists) and the head Relu (op 7).
fn site_map() -> SiteMap {
    SiteMap::new(
        vec![(1, vec![10, 11, 12]), (4, vec![20, 21])],
        7,
        Vec::new(),
    )
}

fn capturing_ledger() -> HintLedger {
    force_hint_capture_for_test(Some(true));
    let l = HintLedger::new();
    force_hint_capture_for_test(None);
    assert!(l.capturing());
    l
}

fn stream(l: HintLedger, cover: &CoverLedger, classes: &[usize]) -> Vec<NodeRecord> {
    let sink = Shared::default();
    let mut boxed = sink.clone();
    l.stream_into(cover, classes, &site_map(), &mut boxed);
    let v = sink.0.lock().expect("lock").clone();
    v
}

fn lit(op: u32, index: u32, positive: bool) -> AbsLit {
    AbsLit {
        site: AbsSite { op, index },
        positive,
    }
}

/// The branch order survives the lane's two split vectors: the mask says
/// which splits were head splits, β stays aligned with its split, the fresh
/// child split goes last with β 0, and the root's inherited head fixes are
/// not part of the frame's path.
#[test]
fn leaf_hints_carry_branch_order_literals_and_aligned_betas() {
    // Frame root entered with one head fix (index 9); then, in branch order:
    // trunk (0,1,+), head (3,-), trunk (1,0,-), head (4,+).
    let trunk: Vec<TrunkSplit> = vec![(0, 1, 1), (1, 0, -1)];
    let heads: Vec<HeadFix> = vec![(9, 1), (3, -1), (4, 1)];
    let order: u128 = 0b1010; // splits 1 and 3 are head splits
    let betas = [0.25, 0.5];
    let head_betas = [9.0, 0.75, 1.5];
    let (splits, beta) = hint_splits(
        &trunk,
        &heads,
        1,
        order,
        &betas,
        &head_betas,
        Some(ChildSplit::Trunk((0, 2, 1))),
    )
    .expect("splits");
    assert_eq!(
        splits,
        vec![
            FrameSplit::Trunk {
                layer: 0,
                pos: 1,
                positive: true
            },
            FrameSplit::Head {
                index: 3,
                positive: false
            },
            FrameSplit::Trunk {
                layer: 1,
                pos: 0,
                positive: false
            },
            FrameSplit::Head {
                index: 4,
                positive: true
            },
            FrameSplit::Trunk {
                layer: 0,
                pos: 2,
                positive: true
            },
        ]
    );
    assert_eq!(beta, vec![0.25, 0.75, 0.5, 1.5, 0.0]);
    // An inconsistent mask falls back to trunk-then-head (never drops a split).
    let (fallback, _) =
        hint_splits(&trunk, &heads, 1, 0b1, &betas, &head_betas, None).expect("fallback");
    assert_eq!(fallback.len(), 4);
    assert!(matches!(fallback[0], FrameSplit::Trunk { .. }));
    assert!(matches!(fallback[3], FrameSplit::Head { .. }));
    // Streamed: absolute literals in branch order, β aligned, prefix first.
    let mut l = capturing_ledger();
    l.set_prefix(vec![lit(4, 21, true)]);
    l.push_leaf(LeafHint {
        splits,
        beta,
        close: LeafClose::Bound,
        dis: vec![2, 5],
        margin: 0.125,
        lift: true,
        empty_split: None,
    });
    let nodes = stream(l, &CoverLedger::new(), &[2, 5]);
    assert_eq!(nodes.len(), 1);
    let n = &nodes[0];
    assert_eq!(
        n.lits,
        vec![
            lit(4, 21, true),
            lit(1, 11, true),
            lit(7, 3, false),
            lit(4, 20, false),
            lit(7, 4, true),
            lit(1, 12, true),
        ]
    );
    assert_eq!(n.beta, vec![0.0, 0.25, 0.75, 0.5, 1.5, 0.0]);
    assert_eq!(n.close, NodeClose::Bound);
    assert_eq!(n.dis, Some(vec![2, 5]));
    assert!(n.lift);
}

/// A retirement is a claim on the EVALUATED domain's region — an ancestor of
/// every leaf below it — for the newly retired classes, and the leaves below
/// no longer claim them.
#[test]
fn retire_hints_name_the_proving_ancestor_region() {
    let parent = vec![FrameSplit::Trunk {
        layer: 0,
        pos: 0,
        positive: true,
    }];
    let mut child = parent.clone();
    child.push(FrameSplit::Head {
        index: 2,
        positive: false,
    });
    let mut l = capturing_ledger();
    l.push_retire(RetireHint {
        splits: parent,
        beta: vec![0.5],
        dis: vec![3],
        margin: f64::NAN,
    });
    l.push_leaf(LeafHint {
        splits: child,
        beta: vec![0.5, 0.0],
        close: LeafClose::Bound,
        dis: vec![1],
        margin: 0.01,
        lift: false,
        empty_split: None,
    });
    let nodes = stream(l, &CoverLedger::new(), &[1, 3]);
    let retire = nodes
        .iter()
        .find(|n| n.close == NodeClose::Retire)
        .expect("retire node");
    let leaf = nodes
        .iter()
        .find(|n| n.close == NodeClose::Bound)
        .expect("leaf node");
    assert_eq!(retire.dis, Some(vec![3]));
    assert_eq!(retire.lits, vec![lit(1, 10, true)]);
    assert!(
        leaf.lits.starts_with(&retire.lits),
        "the retirement's region must be an ancestor of the leaf's"
    );
    assert_eq!(leaf.dis, Some(vec![1]));
}

/// Every class a frame's token relies on is CLAIMED in the stream: a root
/// discharge is a `root` node at the frame prefix, a class covered by a route
/// that streams no leaves (the wave engine's per-class tree covers) is an
/// `opaque` claim, and a class part claims nothing here (its own frame
/// streamed when it was authorized). A proved leaf whose hint was lost is
/// still streamed, from its path, over-claiming the frame's classes.
#[test]
fn pre_covers_and_opaque_routes_are_always_claimed() {
    let root = cert_mr::root_class(RoundMode::Outward, 4, 0.5).expect("root close");
    assert_eq!(
        pre_hint(&root),
        Some(PreHint::Root {
            class: 4,
            margin: 0.5
        })
    );
    // A wave-style tree cover in `pre`.
    let mut ledger = CoverLedger::new();
    ledger.leaf(
        cert_mr::leaf_path(&[(0, 0, 1)], &[], &[], None),
        cert_mr::bound_evidence(RoundMode::Outward, 0.25),
    );
    ledger.leaf(
        cert_mr::leaf_path(&[(0, 0, -1)], &[], &[], None),
        cert_mr::bound_evidence(RoundMode::Outward, 0.25),
    );
    let tree = ledger.prove(&[6]).expect("a complete tree");
    assert_eq!(
        pre_hint(&ClassCover::Tree(tree)),
        Some(PreHint::Opaque {
            classes: vec![6],
            route: "wave"
        })
    );
    // Streamed claims: root + opaque at the prefix, and the two proved
    // leaves (their hints were never pushed) from their paths.
    let mut l = capturing_ledger();
    l.set_prefix(vec![lit(4, 20, false)]);
    l.push_pre(PreHint::Root {
        class: 4,
        margin: 0.5,
    });
    l.push_pre(PreHint::Opaque {
        classes: vec![6],
        route: "wave",
    });
    let nodes = stream(l, &ledger, &[6, 8]);
    let root_node = nodes
        .iter()
        .find(|n| n.close == NodeClose::Root)
        .expect("root claim");
    assert_eq!(root_node.dis, Some(vec![4]));
    assert_eq!(root_node.lits, vec![lit(4, 20, false)]);
    let opaque = nodes
        .iter()
        .find(|n| n.close == NodeClose::Opaque)
        .expect("opaque claim");
    assert_eq!(opaque.dis, Some(vec![6]));
    let unhinted: Vec<&NodeRecord> = nodes.iter().filter(|n| n.src == "tree_unhinted").collect();
    assert_eq!(unhinted.len(), 2, "both proved leaves are streamed");
    for n in unhinted {
        assert_eq!(
            n.dis,
            Some(vec![6, 8]),
            "a lost hint over-claims the frame's classes"
        );
        assert_eq!(n.lits.len(), 2);
        assert_eq!(n.lits[0], lit(4, 20, false));
    }
}

/// B2.3 x head routes (eng12b): a `Root` cover whose certified bound a head
/// route raised — the direct all-head prepass (`merge_direct_prelude`) or the
/// head-group refinement (`refine_head_groups_with_direct`) — is never
/// streamed as a `close: root` node carrying the head route's margin, which
/// the checker's root rung does not compute. A class the ordinary root
/// evaluation closed on its own keeps that evaluation's checkable claim with
/// ITS margin; any other is the head route's opaque claim, under the route
/// that raised it last. Dark (identical bounds) nothing is raised and every
/// cover is `proof_pre_hint`'s, unchanged. Mutation targets: emitting
/// `pre_hint` for a raised class; the raised margin in place of the root's.
#[test]
fn head_route_raised_root_covers_stream_opaque_or_the_roots_own_claim() {
    let mode = RoundMode::Outward;
    let adv = [0usize, 2, 3, 5];
    // 0: unraised, closed by the root. 2: open at the root, closed by the
    // direct merge. 3: open after the merge, closed by head groups. 5: the
    // root closed it at 0.25, head groups raised it to 0.5.
    let root = [0.1, -0.2, -0.3, 0.25];
    let merged = [0.1, 0.05, -0.3, 0.25];
    let fin = [0.1, 0.05, 0.4, 0.5];
    let raised = head_raised_classes(mode, &adv, &root, &merged, &fin);
    assert_eq!(
        raised,
        vec![
            HeadRaised {
                class: 2,
                route: "direct_head",
                root_margin: None
            },
            HeadRaised {
                class: 3,
                route: "head_group",
                root_margin: None
            },
            HeadRaised {
                class: 5,
                route: "head_group",
                root_margin: Some(0.25)
            },
        ]
    );
    assert!(
        head_raised_classes(mode, &adv, &root, &root, &root).is_empty(),
        "dark: identical bounds raise nothing"
    );
    assert!(
        head_raised_classes(mode, &adv, &[f64::NAN; 4], &fin, &fin).is_empty(),
        "a non-finite root bound is never raised"
    );
    assert!(
        head_raised_classes(mode, &adv, &root[..2], &merged, &fin).len() <= 2,
        "a misaligned entry is never raised"
    );
    let cover = |class: usize, b: f64| cert_mr::root_class(mode, class, b).expect("root close");
    // Unraised: proof_pre_hint's root claim, unchanged.
    assert_eq!(head_route_pre_hint(&cover(0, 0.1), &raised), None);
    assert!(matches!(
        proof_pre_hint(&cover(0, 0.1), &mut Vec::new(), &[], true),
        Some(PreHint::Root { class: 0, .. })
    ));
    assert_eq!(
        head_route_pre_hint(&cover(2, 0.05), &raised),
        Some(PreHint::Opaque {
            classes: vec![2],
            route: "direct_head"
        })
    );
    assert_eq!(
        head_route_pre_hint(&cover(3, 0.4), &raised),
        Some(PreHint::Opaque {
            classes: vec![3],
            route: "head_group"
        })
    );
    assert_eq!(
        head_route_pre_hint(&cover(5, 0.5), &raised),
        Some(PreHint::Root {
            class: 5,
            margin: 0.25
        }),
        "the root evaluation's own margin, not the head route's"
    );
    // Streamed: the opaque claim is an `opaque` node under the head route's
    // token, never a `root` node.
    let mut l = capturing_ledger();
    l.push_pre(head_route_pre_hint(&cover(3, 0.4), &raised).expect("raised"));
    let nodes = stream(l, &CoverLedger::new(), &[]);
    assert_eq!(nodes.len(), 1);
    assert_eq!(nodes[0].close, NodeClose::Opaque);
    assert_eq!(nodes[0].src, "head_group");
    assert_eq!(nodes[0].dis, Some(vec![3]));
}

/// B2.3: a frozen-dual close whose prune trace is INCOMPLETE (the record cap
/// ran out before every certified prune was recorded) stays the class's
/// `close: fd` CLAIM: `proof_pre_hint` never turns it into its prunes and
/// leaves the trace unconsumed, so the frame streams one `fd` node and no
/// `hint.fd_state` / `fd_prune` record (a partial prune set streamed as the
/// class's certificate would leave cells the checker reads as a claimed-cover
/// gap: INVALID on a correct unsat). Positive control: the same class with a
/// COMPLETE trace streams its state record and both prunes instead of the
/// claim; and a non-root frame keeps the claim even then.
#[test]
fn incomplete_fd_trace_is_never_streamed_as_a_certificate() {
    use crate::certify::fd_proof::FdProofParts;
    use crate::certify::{FdLambdaRecord, FdProofTrace, RecordingSink};
    use crate::margin_row::zono_bab::{
        Candidate, ClassReport, ClassRun, FrozenDualStateMode, LevelTrace, PrunedNode,
        SearchReport, SearchVerdict,
    };
    use crate::margin_row::zono_state::Side;

    let class = 4usize;
    // The tier closed the class by ONE split: two children, both pruned.
    let report = ClassReport {
        class,
        lane_root: -0.1,
        result: Ok(ClassRun {
            state_gens: 3,
            unstable: 1,
            depth_cap: 1,
            build_ms: 0.0,
            state_bytes: 0,
            build_estimate_bytes: 0,
            build_columns: 0,
            streamed_rows: false,
            root_g: -0.2,
            search: SearchReport {
                verdict: SearchVerdict::Closed,
                root_certified: -0.2,
                certified_prunes: 2,
                cert_min_slack: 1e-3,
                levels: vec![LevelTrace {
                    depth: 1,
                    frontier: 2,
                    pruned: 2,
                }],
                ..SearchReport::empty()
            },
            state: FrozenDualStateMode::PerClass,
            root_loss: None,
            proof: None,
            progressive: None,
        }),
    };
    let cover =
        cert_mr::frozen_dual_class(&report).expect("a closed ledger is a frozen-dual cover");
    let prune = |side: Side| PrunedNode {
        sides: vec![side],
        l0: vec![0.0],
        l1: vec![0.0],
        certified: 1e-3,
        candidate: Candidate::Floor,
    };
    let trace = |prunes: Vec<PrunedNode>, complete: bool| {
        FdProofTrace::new(FdProofParts {
            class,
            t: 0,
            state_sha256: "0".repeat(64),
            n_gens: 3,
            in_gens: 2,
            unstable: 1,
            relaxed: 1,
            root: -0.2,
            order: vec![AbsSite { op: 1, index: 10 }],
            lambda: vec![FdLambdaRecord {
                op: 1,
                idx: vec![10],
                val: vec![0.5],
                lo: vec![-1.0],
                hi: vec![1.0],
            }],
            prunes,
            root_closed: false,
            complete,
        })
    };
    let stream_pre = |pre: Option<PreHint>| -> RecordingSink {
        let mut l = capturing_ledger();
        if let Some(p) = pre {
            l.push_pre(p);
        }
        let mut rec = RecordingSink::default();
        l.stream_into(&CoverLedger::new(), &[class], &site_map(), &mut rec);
        rec
    };
    let claims = |rec: &RecordingSink| {
        rec.nodes
            .iter()
            .filter(|n| n.close == NodeClose::Fd && n.dis.as_deref() == Some(&[class][..]))
            .count()
    };
    let prune_nodes =
        |rec: &RecordingSink| rec.nodes.iter().filter(|n| n.src == "fd_prune").count();

    // The record cap hit: one of the two certified prunes was recorded.
    let partial = trace(vec![prune(Side::Off)], false);
    assert!(!partial.complete());
    let mut proofs = vec![partial.clone()];
    let hint = proof_pre_hint(&cover, &mut proofs, &[], true);
    assert_eq!(
        hint,
        Some(PreHint::FrozenDual {
            class,
            margin: 1e-3
        }),
        "an incomplete trace must leave the class's claim"
    );
    assert_eq!(
        proofs,
        vec![partial],
        "an incomplete trace must stay unconsumed"
    );
    let rec = stream_pre(hint);
    assert!(
        rec.fd_states.is_empty(),
        "an incomplete trace streamed its state record: {:?}",
        rec.fd_states
    );
    assert_eq!(
        prune_nodes(&rec),
        0,
        "an incomplete trace streamed its prunes"
    );
    assert_eq!(
        claims(&rec),
        1,
        "the class's close: fd claim is streamed once"
    );

    // Positive control: complete, it replaces the claim with its prunes.
    let mut proofs = vec![trace(vec![prune(Side::Off), prune(Side::On)], true)];
    let hint = proof_pre_hint(&cover, &mut proofs, &[], true);
    assert!(
        matches!(hint, Some(PreHint::FrozenDualProof(_))),
        "a complete trace on a root frame streams its prunes: {hint:?}"
    );
    assert!(proofs.is_empty(), "the streamed trace is consumed");
    let rec = stream_pre(hint);
    assert_eq!(rec.fd_states.len(), 1);
    assert_eq!(rec.fd_states[0].prunes, 2);
    assert_eq!(prune_nodes(&rec), 2);
    assert_eq!(claims(&rec), 0, "a streamed trace replaces the claim");

    // The tier runs on the outer frame only: a non-root frame keeps the claim.
    let mut proofs = vec![trace(vec![prune(Side::Off), prune(Side::On)], true)];
    assert!(matches!(
        proof_pre_hint(&cover, &mut proofs, &[], false),
        Some(PreHint::FrozenDual { .. })
    ));
    assert_eq!(proofs.len(), 1);
}
