// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Mixed conjunction regressions for the direct source publication barrier.

use super::*;
use crate::certify::{LeafPath, UnsatSource};
use crate::margin_row::head_group_source::HeadGroupSourceGuard;
use crate::margin_row::spec::{TwinOpSpec, TwinSpec};

fn spec() -> TwinSpec {
    TwinSpec {
        n_in: 1,
        ops: vec![
            TwinOpSpec::Conv {
                input: 0,
                weight: vec![1.0],
                bias: vec![2.0],
                bias_err: vec![0.0],
                weight_rel_err: 0.0,
                kernel: (1, 1, 1, 1),
                stride: (1, 1),
                pads: (0, 0, 0, 0),
                ishape: (1, 1, 1),
                oshape: (1, 1, 1),
            },
            TwinOpSpec::Relu { input: 1 },
            TwinOpSpec::Flatten { input: 2 },
            TwinOpSpec::Gemm {
                input: 3,
                weight: vec![1.0, -1.0],
                bias: vec![0.0; 2],
                bias_err: vec![0.0; 2],
                weight_rel_err: 0.0,
                shape: (2, 1),
            },
            TwinOpSpec::Relu { input: 4 },
            TwinOpSpec::Gemm {
                input: 5,
                weight: vec![0.0; 6],
                bias: vec![1.0, 0.0, 0.0],
                bias_err: vec![0.0; 3],
                weight_rel_err: 0.0,
                shape: (3, 2),
            },
        ],
    }
}

fn cfg() -> BabConfig {
    BabConfig {
        deadline: Some(Instant::now() + Duration::from_secs(10)),
        head_groups: super::super::HeadGroupPolicy::new(5.0, 2, 4),
        frozen_dual_tier: Some(false),
        ..BabConfig::default()
    }
}

fn mixed_ledger(
    root: &RootGates,
    prelude: &super::super::head_group::direct::DirectPrelude,
    deadline: Instant,
    tree: bool,
) -> Ledger {
    let first = cert_mr::root_class(root.mode, 1, prelude.bounds()[0]).unwrap();
    let mut ledger = Ledger {
        scope: TokenScope::Instance,
        required: Some(vec![1, 2]),
        pre: vec![first],
        direct_guard: Some(prelude.publication_guard(deadline)),
        ..Ledger::default()
    };
    if tree {
        ledger.cover.leaf(
            Some(LeafPath::root()),
            cert_mr::bound_evidence(root.mode, prelude.bounds()[1]),
        );
    } else {
        let second = cert_mr::root_class(root.mode, 2, prelude.bounds()[1]).unwrap();
        let part =
            authorize_unsat(TokenScope::ClassPart, &[2], &[second], &GuardReport::new()).unwrap();
        ledger.pre.push(ClassCover::Part(part));
    }
    ledger
}

#[test]
fn mixed_tree_and_classwise_parts_reject_expiry_after_authorizer_work() {
    let _capture = HeadGroupSourceGuard::enter(true);
    let net = TwinNet::compile(&spec()).unwrap();
    let root = RootGates::build(&net, &[-1.0], &[1.0], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let cfg = cfg();
    let direct = direct_root_attempt(&eng, 0, &[1, 2], &cfg).unwrap();
    let prelude = direct.prelude.as_ref().unwrap();
    assert!(prelude.complete());
    let re = root_eval(&eng, &net, 0, &[1, 2]).unwrap();
    for tree in [false, true] {
        let classes = if tree { &[2][..] } else { &[][..] };
        let mut control = mixed_ledger(&root, prelude, cfg.deadline.unwrap(), tree);
        let token = authorize_frame(&mut control, classes, tree, &net, &root, 0, &re, &cfg)
            .expect("same mixed conjunction must authorize with its live source");
        assert_eq!(token.source(), UnsatSource::Authorized);
        assert!(token.publishable() && token.covers_all(&[1, 2]));

        let mut late = mixed_ledger(&root, prelude, cfg.deadline.unwrap(), tree);
        // The injected instant changes only AFTER real G2/hint-stream work,
        // so an entry-only guard would incorrectly return the token here.
        DIRECT_EXPIRE_AFTER_AUTHORIZER.with(|fault| fault.set(true));
        let refusal =
            authorize_frame(&mut late, classes, tree, &net, &root, 0, &re, &cfg).unwrap_err();
        assert!(refusal.contains("direct head source/deadline/FP"));
        DIRECT_EXPIRE_AFTER_AUTHORIZER.with(|fault| assert!(!fault.get()));
    }
}

#[test]
fn final_mixed_ledger_refuses_foreign_original_class_order() {
    let _capture = HeadGroupSourceGuard::enter(true);
    let net = TwinNet::compile(&spec()).unwrap();
    let root = RootGates::build(&net, &[-1.0], &[1.0], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let cfg = cfg();
    let direct = direct_root_attempt(&eng, 0, &[1, 2], &cfg).unwrap();
    let prelude = direct.prelude.as_ref().unwrap();
    let re = root_eval(&eng, &net, 0, &[2, 1]).unwrap();
    let mut ledger = mixed_ledger(&root, prelude, cfg.deadline.unwrap(), false);
    ledger.required = Some(vec![2, 1]);
    assert!(
        authorize_frame(&mut ledger, &[], false, &net, &root, 0, &re, &cfg)
            .unwrap_err()
            .contains("direct head source/deadline/FP")
    );
}

#[test]
fn partial_max_merge_changes_returned_rows_without_changing_the_root_memo() {
    let _capture = HeadGroupSourceGuard::enter(true);
    let net = TwinNet::compile(&spec()).unwrap();
    let root = RootGates::build(&net, &[-1.0], &[1.0], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let cfg = cfg();
    let mut direct = direct_root_attempt(&eng, 0, &[1, 2], &cfg).unwrap();
    let stored = root_eval_store(&eng, &net, 0, &[1, 2]).unwrap();
    let prior = stored.dj.iter().map(|b| b.to_bits()).collect::<Vec<_>>();
    let mut returned = root_eval_lookup(&eng, &net, 0, &[1, 2]).unwrap();
    let guard = merge_direct_prelude(&eng, 0, &[1, 2], &mut returned, &cfg, Some(&mut direct));
    let after = root_eval_lookup(&eng, &net, 0, &[1, 2]).unwrap();
    assert!(
        guard.is_some(),
        "the direct baseline must actually strengthen the returned rows"
    );
    assert_eq!(
        after.dj.iter().map(|b| b.to_bits()).collect::<Vec<_>>(),
        prior
    );
    assert!(returned
        .dj
        .iter()
        .zip(&stored.dj)
        .all(|(new, old)| new >= old));
    assert!(returned
        .dj
        .iter()
        .zip(&stored.dj)
        .any(|(new, old)| new > old));
    assert!(Arc::ptr_eq(&returned.pack, &after.pack));
}

/// B2.3 x head routes (eng12b review), END TO END through both public lane
/// entries: a direct all-head prepass that closes the whole frame publishes
/// through `direct_root_outcome`, where NO root evaluation ran, so under
/// capture every class streams as an opaque `direct_head` claim and never as
/// a checkable `close: root` node carrying the head route's margin (which the
/// checker's root rung does not compute). The unit test in `proof_hint_tests`
/// pins the classification; this one pins the WIRING it relies on — the
/// `head_route_pre_hint` arm of `authorize_frame_view` and the `head_raised`
/// population of `direct_root_outcome` (mutation targets: either one makes
/// these nodes `root`). Capture off streams nothing and publishes the same
/// verdict bits (hint data only).
#[test]
fn direct_complete_cover_streams_only_opaque_direct_head_claims_end_to_end() {
    use crate::certify::{
        force_hint_capture_for_test, with_thread_sink, AlphaRecord, FdStateRecord, LeafRecord,
        NodeClose, NodeRecord, ProofSink, SitesRecord, VerdictRecord,
    };
    use std::sync::Mutex;

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
        fn fd_state(&mut self, _state: &FdStateRecord) {}
        fn finish(&mut self, _verdict: &VerdictRecord) {}
    }

    let _source = HeadGroupSourceGuard::enter(true);
    let net = TwinNet::compile(&spec()).unwrap();
    let root = RootGates::build(&net, &[-1.0], &[1.0], RoundMode::Outward, None).unwrap();
    for classwise in [false, true] {
        let run = |capture: bool| {
            let sink = Shared::default();
            force_hint_capture_for_test(Some(capture));
            let (out, _) = with_thread_sink(Box::new(sink.clone()), || {
                if classwise {
                    MarginRowBab::run_classwise(&net, &root, 0, &[1, 2], cfg())
                } else {
                    MarginRowBab::run(&net, &root, 0, &[1, 2], cfg())
                }
            });
            force_hint_capture_for_test(None);
            let nodes = sink.0.lock().unwrap().clone();
            (out, nodes)
        };
        let (off, off_nodes) = run(false);
        let (on, on_nodes) = run(true);
        let (
            MarginRowOutcome::Unsat(off_stats, off_token),
            MarginRowOutcome::Unsat(on_stats, on_token),
        ) = (&off, &on)
        else {
            panic!("classwise={classwise}: the direct prepass must close the fixture (else VOID)");
        };
        assert_eq!(
            on_stats.stop, "direct_head_closed_at_root",
            "classwise={classwise}: the direct route must be the publisher (else VOID)"
        );
        assert_eq!(off_stats.stop, on_stats.stop);
        assert_eq!(
            off_token.min_margin().to_bits(),
            on_token.min_margin().to_bits()
        );
        assert!(off_nodes.is_empty(), "capture off streams nothing");
        let mut claimed: Vec<usize> = Vec::new();
        for n in &on_nodes {
            assert_eq!(n.close, NodeClose::Opaque, "classwise={classwise}: {n:?}");
            assert_eq!(n.src, "direct_head", "classwise={classwise}: {n:?}");
            claimed.extend(n.dis.clone().unwrap_or_default());
        }
        claimed.sort_unstable();
        assert_eq!(
            claimed,
            vec![1, 2],
            "classwise={classwise}: every class claimed exactly once"
        );
    }
}
