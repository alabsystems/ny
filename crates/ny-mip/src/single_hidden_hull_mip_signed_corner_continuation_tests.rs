// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Public native continuation control using the genuine fixture and owner.
use super::*;
use crate::single_hidden_hull::bab::{HullBabOptions, HullBabStop};
use crate::single_hidden_hull::HullLpStatus;

#[test]
fn corner_authentic_fresh_cache_closes_native_root_without_spending_its_lp() {
    let fixture = CrossFixture::new();
    let problem = fixture.source();
    // Fresh owner: neither ordinary cache.bind nor optional corner bind has
    // been called. Reusing the earlier test's owner would consume authority.
    let cuts = genuine_owner(&problem, &fixture, true);
    let owner = cuts.signed.as_ref().unwrap();
    assert!(owner.lower < -1.88);
    assert!(cuts.checked_lower < 0.);
    assert_eq!(cuts.proposal_lp_calls, 1);
    assert_eq!(cuts.native_funding_report().funded_calls, 1);
    let original = (
        owner.lower.to_bits(),
        owner.constant.to_bits(),
        owner.g.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
    );

    // Genuine original encoder binding publishes native origin rows and owns
    // the FIRST original Source replay. Keep its parts alive throughout use.
    let mut parts = fixture.original();
    cuts.append_to(&mut parts, &fixture.encoding(), end())
        .unwrap();
    assert_eq!(owner.stats().source_replays, 1);
    // The original second replay installs the ordinary cache and its optional
    // sidecar. Do not call bind or root_lower before the public continuation.
    let _old_proof = cuts.best_signed_root_proof(end());
    assert_eq!(owner.stats().source_replays, 2);
    let cache = cuts
        .signed_child_cache
        .get()
        .expect("completed second replay cache");
    let before = cache.stats();
    assert_eq!((before.collection_attempts, before.bind_attempts), (1, 0));
    assert_eq!(before.cached_masks, 12);
    assert_eq!(cuts.native_rows_report().attempted, 0);

    let continuation = cuts
        .continue_hull_bab_with_native_rows(
            end(),
            HullBabOptions {
                workers: 1,
                max_nodes: 1,
            },
        )
        .expect("public original-root continuation");
    let report = continuation.report();
    // This is an actual branch/call result through the opaque public proof
    // object, not a prediction from a standalone corner scalar.
    assert!(continuation.verified(), "{report:?}");
    assert_eq!(report.stop, HullBabStop::Proved);
    assert_eq!(
        (report.nodes, report.closed, report.open, report.max_depth),
        (1, 1, 0, 0)
    );
    assert_eq!((report.root_lp_calls, report.proposal_lp_calls), (0, 0));
    assert_eq!(report.root_lp_status, HullLpStatus::Skipped);
    assert_eq!(report.lp_failures, 0);
    assert_eq!(report.workers, 1);
    assert!(report.worst.is_none());

    // A Some value proves the native row capability reached bab_seed; absence
    // of native rows cannot masquerade as an intentionally skipped funded LP.
    let before_native = report
        .root_before_native
        .expect("published native capability reached root gate");
    assert!(before_native > 0.13 && rat(before_native) <= rat(17. / 128.));
    assert!(
        report.root_certified.is_some_and(|v| v <= 0.),
        "ordinary completed root already closed the fixture: {report:?}"
    );
    assert!(report.root_lp_value.is_none());
    let child = continuation.signed_child_report().unwrap();
    assert_eq!((child.source_replays, child.bind_attempts), (2, 1));
    assert_eq!((child.attempted_calls, child.completed_calls), (0, 0));
    // Existing lp_skips counts CHILD pricing exits, not this root-positive
    // gate; the authoritative root LP-skip evidence is the call census above.
    assert_eq!(child.lp_skips, 0);
    let native = cuts.native_rows_report();
    assert_eq!(
        (native.attempted, native.committed, native.priced),
        (0, 0, 0)
    );
    assert_eq!(cuts.native_funding_report().funded_calls, 1);
    assert_eq!(cuts.proposal_lp_calls, 1);
    assert_eq!(
        original,
        (
            owner.lower.to_bits(),
            owner.constant.to_bits(),
            owner.g.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        )
    );
    assert_eq!(owner.stats().source_replays, 2);
}
