// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Exercise the real scored-policy lane entry, including the mutable root
//! owners before BaB. Census assertions are run with the serial validation
//! harness, since the production census aggregates overlapping invocations.

use super::*;
use std::time::Duration;

fn unstable_trunk_constant_margins() -> TwinSpec {
    TwinSpec {
        n_in: 1,
        ops: vec![
            TwinOpSpec::Conv {
                input: 0,
                weight: vec![1.0],
                bias: vec![0.0],
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
                bias: vec![-0.5, 0.5],
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

fn alpha_policy() -> MarginRowRuntimePolicy {
    MarginRowRuntimePolicy {
        // Keep the ordinary-path control dark when production defaults arm
        // direct head groups. The armed sibling supplies its explicit policy.
        head_groups: None,
        root_alpha: RootAlphaRuntimePolicy {
            alpha_enabled: true,
            alpha_guard: true,
            alpha_secs: 1.0,
            alpha_iters: 1,
            ..RootAlphaRuntimePolicy::default()
        },
        ..MarginRowRuntimePolicy::default()
    }
}

fn call(policy: &MarginRowRuntimePolicy) -> MarginRowOutcome {
    run_margin_row_lane_with_policy(
        &unstable_trunk_constant_margins(),
        &[-1.0],
        &[1.0],
        0,
        &[2, 1],
        Some(Instant::now() + Duration::from_secs(10)),
        0,
        *policy,
    )
}

fn calls(line: &str, site: &str) -> usize {
    let prefix = format!("site.{site}=");
    line.split_whitespace()
        .find_map(|field| field.strip_prefix(&prefix))
        .unwrap_or_else(|| panic!("missing {site}: {line}"))
        .split(':')
        .next()
        .unwrap()
        .parse()
        .unwrap()
}

#[test]
fn armed_production_entry_closes_before_alpha_guard_and_identity_rows() {
    let mut policy = alpha_policy();
    policy.head_groups = HeadGroupPolicy::new(5.0, 2, 4);
    // Arm all four potential upstream owners. A complete direct token must
    // precede every one, while preserving their configured policy evidence.
    policy.root_alpha.rebake_enabled = true;
    policy.joint_refresh.rounds = 1;
    policy.root_certify_once.enabled = true;
    let out = call(&policy);
    let MarginRowOutcome::Unsat(stats, token) = out else {
        panic!("complete constant margins must authorize at the early entry");
    };
    assert_eq!(token.scope(), crate::certify::TokenScope::Instance);
    assert_eq!(stats.stop, "direct_head_closed_at_root");
    assert_eq!(stats.expansions, 0);
    assert_eq!(stats.root_closed_classes, 2);
    assert_eq!(stats.ledger_ok, Some(true));
    let evidence = stats.root_alpha_execution;
    assert!(evidence.eligible && evidence.alpha_enabled && evidence.alpha_guard);
    assert!(evidence.rebake_enabled);
    assert_eq!(evidence.standalone_attempted, 0);
    assert_eq!(evidence.standalone_completed, 0);
    assert_eq!(evidence.standalone_passes, 0);
    assert_eq!(evidence.standalone_guard_baseline_evals, 0);
    assert_eq!(evidence.standalone_guard_candidate_evals, 0);
    assert_eq!(evidence.rebake_attempted, 0);
    assert_eq!(evidence.rebake_completed, 0);
    assert_eq!(evidence.rebake_baseline_evals, 0);
    assert_eq!(evidence.rebake_alpha_passes, 0);
    assert!(!evidence.policy_mismatch);
    let line = census::snapshot_line("direct_early_entry_test");
    for site in ["y_rows", "root_eval", "root_ascent", "root_walk"] {
        assert_eq!(calls(&line, site), 0, "{line}");
    }
}

#[test]
fn dark_production_control_reaches_alpha_guard_root_and_identity_rows() {
    let out = call(&alpha_policy());
    let MarginRowOutcome::Unsat(stats, token) = out else {
        panic!("constant margins must also close through the ordinary path");
    };
    assert_eq!(token.scope(), crate::certify::TokenScope::Instance);
    assert_ne!(stats.stop, "direct_head_closed_at_root");
    assert!(stats.root_alpha_execution.eligible);
    assert!(stats.root_alpha_execution.alpha_enabled);
    assert!(stats.root_alpha_execution.alpha_guard);
    assert_eq!(stats.root_alpha_execution.standalone_attempted, 1);
    assert_eq!(stats.root_alpha_execution.standalone_completed, 0);
    // The published baseline guard ran, but search_alpha returns None when
    // every frozen margin already closes. completed(None) has no report from
    // which to recover the baseline counter. Its zero is not evidence that
    // no guard work occurred: require actual root/identity census arrival.
    assert_eq!(
        stats.root_alpha_execution.standalone_guard_baseline_evals,
        0
    );
    let line = census::snapshot_line("direct_dark_control_test");
    assert!(calls(&line, "root_eval") > 0, "{line}");
    assert!(calls(&line, "y_rows") > 0, "{line}");
}

#[test]
fn early_and_ordinary_authorizers_share_frozen_publication_and_policy_config() {
    let spec = unstable_trunk_constant_margins();
    let net = TwinNet::compile(&spec).unwrap();
    let root = RootGates::build(&net, &[-1.0], &[1.0], RoundMode::Outward, None).unwrap();
    let policy = alpha_policy();
    let deadline = Instant::now() + Duration::from_secs(10);
    let mut cfg = frozen_bab_config(
        &policy,
        &root,
        0,
        Some(deadline),
        0.01,
        None,
        0,
        &RootAlphaExecutionEvidence::skipped_by_direct_cover(policy.root_alpha, true),
    );
    let publish = cfg.effective_publish_deadline();
    let widths = (cfg.k_head, cfg.k_trunk, cfg.frontier, cfg.lru_cap);
    cfg.root_alpha_execution =
        RootAlphaExecutionEvidence::completed(policy.root_alpha, true, None, None);
    assert_eq!(cfg.deadline, Some(deadline));
    assert_eq!(cfg.effective_publish_deadline(), publish);
    assert_eq!((cfg.k_head, cfg.k_trunk, cfg.frontier, cfg.lru_cap), widths);
    assert!(cfg.root_alpha_execution.alpha_enabled);
    assert!(cfg.root_alpha_execution.alpha_guard);
}
