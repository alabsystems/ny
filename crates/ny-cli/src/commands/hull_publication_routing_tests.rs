// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::commands::beta_crown::hull_publication::tests::{ordinary_dispatch, plain_capture};
use crate::commands::unsat_receipt::PublishedUnsatSource;
use ny_propagate::certify::work::{LeaseClass, WorkRequest};
use std::sync::Arc;

fn original_result() -> (
    VnncompResult,
    Arc<crate::commands::beta_crown::hull_publication::InstanceVerificationResources>,
) {
    let (captured, owner) = ordinary_dispatch(0.75, false, false);
    let result = translate_captured_verification(captured, || {
        panic!("sealed hull must not use graph/legacy JSON authority")
    });
    assert!(matches!(result, VnncompResult::Unsat(_)));
    (result, owner)
}

#[test]
fn hull_publication_dispatch_to_actual_result_writer_retains_source_hash() {
    let (captured, owner) = ordinary_dispatch(0.75, false, false);
    let value: serde_json::Value = serde_json::from_str(captured.json()).unwrap();
    let original_scope = value["hull_resource"]["original_scope"]
        .as_str()
        .unwrap()
        .to_owned();
    let result = translate_captured_verification(captured, || panic!("no JSON authority"));
    let VnncompResult::Unsat(receipt) = &result else {
        panic!("sealed replay required")
    };
    assert_eq!(
        receipt.published_summary().2,
        PublishedUnsatSource::OriginalHull
    );
    let authority = receipt.hull_authority().unwrap();
    let actual_scope =
        authority
            .scope()
            .iter()
            .fold(String::with_capacity(64), |mut text, byte| {
                use std::fmt::Write as _;
                write!(&mut text, "{byte:02x}").expect("writing to String is infallible");
                text
            });
    assert_eq!(original_scope, actual_scope);
    let output = tempfile::tempdir().unwrap();
    let results_file = output.path().join("result.txt");
    fs::write(&results_file, "unknown\n").unwrap();
    assert!(write_terminal_result(&results_file, None, result).unwrap());
    assert_eq!(fs::read_to_string(&results_file).unwrap(), "unsat\n");
    assert_eq!(owner.budget().receipt().unwrap().proposals_reserved, 1);
    authority.checkpoint().unwrap();
}

#[test]
fn hull_publication_cancel_between_translation_and_rename_preserves_placeholder() {
    let (result, owner) = original_result();
    let output = tempfile::tempdir().unwrap();
    let results_file = output.path().join("result.txt");
    fs::write(&results_file, "unknown\n").unwrap();
    owner.cancel();
    assert!(write_terminal_result(&results_file, None, result).is_err());
    assert_eq!(fs::read_to_string(&results_file).unwrap(), "unknown\n");
    assert!(!results_file.with_extension("result.tmp").exists());
}

#[test]
fn hull_publication_writer_cap_refuses_before_staging_and_preserves_prior_proof() {
    let (result, owner) = original_result();
    let output = tempfile::tempdir().unwrap();
    let results_file = output.path().join("result.txt");
    fs::write(&results_file, "unknown\n").unwrap();
    let live = owner.budget().receipt().unwrap().live_persistent_bytes;
    let mut hold = owner
        .budget()
        .lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                persistent_bytes: (32 << 20) - live - 1024,
                ..Default::default()
            },
        )
        .unwrap();
    hold.mark_materialized().unwrap();
    let hold = hold.commit_persistent().unwrap();
    assert!(write_results(&results_file, &result).is_err());
    assert_eq!(fs::read_to_string(&results_file).unwrap(), "unknown\n");
    assert!(!results_file.with_extension("result.tmp").exists());
    drop(hold);
    write_results(&results_file, &result).unwrap();
    assert_eq!(fs::read_to_string(&results_file).unwrap(), "unsat\n");
}

#[test]
fn hull_publication_plain_json_cannot_upgrade_to_original_replay_receipt() {
    let captured = plain_capture("{\"status\":\"verified\",\"hull_resource\":{\"original_replays\":1,\"certificate_proposals\":1}}");
    let result = translate_captured_verification(captured, || Err("no graph token".to_owned()));
    let VnncompResult::Unsat(receipt) = result else {
        panic!("preserve the existing legacy JSON door")
    };
    assert!(receipt.hull_authority().is_none());
    assert_eq!(
        receipt.published_summary().2,
        PublishedUnsatSource::Spine(ny_propagate::certify::UnsatSource::LegacyUncovered(
            LegacyUnsatAuthority::BetaCrownVerdict
        ))
    );
}

#[test]
fn hull_publication_queued_cancel_is_checked_on_coordinator_before_rename() {
    let (result, owner) = original_result();
    let VnncompResult::Unsat(receipt) = &result else {
        unreachable!()
    };
    let weak = Arc::downgrade(&receipt.hull_authority().unwrap());
    let output = tempfile::tempdir().unwrap();
    let results_file = output.path().join("result.txt");
    fs::write(&results_file, "unknown\n").unwrap();
    let (sender, receiver) = std::sync::mpsc::channel();
    let publisher = ShortGraceResultPublisher {
        results_file: results_file.clone(),
        sender,
    };
    let writer = std::thread::spawn(move || publisher.publish_terminal(result));
    let request = receiver
        .recv_timeout(std::time::Duration::from_secs(2))
        .unwrap();
    assert!(
        weak.upgrade().is_some(),
        "pending message must own the original replay"
    );
    owner.cancel();
    let (sender, receiver) = std::sync::mpsc::channel();
    sender.send(request).unwrap();
    drop(sender);
    let renames = std::sync::atomic::AtomicUsize::new(0);
    let outcome = await_short_grace_terminal_result_with_rename_and_record(
        &results_file,
        std::time::Instant::now() + std::time::Duration::from_secs(2),
        receiver,
        None,
        |_, _| {
            renames.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(())
        },
        |_| panic!("cancelled proof cannot reach watchdog commit"),
    );
    assert_eq!(outcome, ShortGraceCoordinatorOutcome::PublicationFailed);
    assert_eq!(renames.load(std::sync::atomic::Ordering::SeqCst), 0);
    assert!(writer.join().unwrap().is_err());
    assert!(weak.upgrade().is_none());
    assert_eq!(
        owner.budget().receipt().unwrap().live_persistent_bytes,
        4096
    );
    assert_eq!(fs::read_to_string(&results_file).unwrap(), "unknown\n");
}

#[test]
fn hull_publication_queued_expired_publication_window_cannot_rename() {
    let (result, _) = original_result();
    let output = tempfile::tempdir().unwrap();
    let results_file = output.path().join("result.txt");
    fs::write(&results_file, "unknown\n").unwrap();
    let (sender, receiver) = std::sync::mpsc::channel();
    let publisher = ShortGraceResultPublisher {
        results_file: results_file.clone(),
        sender,
    };
    let writer = std::thread::spawn(move || publisher.publish_terminal(result));
    let request = receiver
        .recv_timeout(std::time::Duration::from_secs(2))
        .unwrap();
    let staged = request.staged_path.clone();
    let (sender, receiver) = std::sync::mpsc::channel();
    sender.send(request).unwrap();
    drop(sender);
    let outcome = await_short_grace_terminal_result_with_rename_and_record(
        &results_file,
        std::time::Instant::now()
            .checked_sub(std::time::Duration::from_millis(1))
            .unwrap(),
        receiver,
        None,
        |_, _| panic!("expired publication must not rename"),
        |_| panic!("expired publication must not acknowledge"),
    );
    assert_eq!(outcome, ShortGraceCoordinatorOutcome::Deadline);
    assert!(!writer.join().unwrap().unwrap());
    assert!(!staged.exists());
    assert_eq!(fs::read_to_string(&results_file).unwrap(), "unknown\n");
}

#[cfg(unix)]
#[test]
fn hull_publication_short_grace_positive_retains_replay_through_real_ack() {
    let (result, owner) = original_result();
    let output = tempfile::tempdir().unwrap();
    let results_file = output.path().join("result.txt");
    fs::write(&results_file, "unknown\n").unwrap();
    let (sender, receiver) = std::sync::mpsc::channel();
    let publisher = ShortGraceResultPublisher {
        results_file: results_file.clone(),
        sender,
    };
    let writer = std::thread::spawn(move || publisher.publish_terminal(result));
    let hard_stop = canonical_monotonic_now()
        .expect("CLOCK_MONOTONIC")
        .checked_add(std::time::Duration::from_secs(2))
        .expect("small test deadline");
    let (ack_sender, ack_receiver) = std::sync::mpsc::sync_channel(1);
    let (control_sender, control_receiver) = std::sync::mpsc::sync_channel(1);
    let control = std::thread::spawn(move || {
        assert!(matches!(
            control_receiver.recv().unwrap(),
            ExternalWatchdogWriteRequest::PublicationCommit { .. }
        ));
        ack_sender.send(canonical_monotonic_now()).unwrap();
        let ExternalWatchdogWriteRequest::PublicationFinalize {
            write_observed_before_h,
            ..
        } = control_receiver.recv().unwrap()
        else {
            panic!("exact C/ACK/P sequence")
        };
        write_observed_before_h.send(true).unwrap();
    });
    let watchdog = ExternalWatchdogController {
        sender: control_sender,
        hard_stop,
        publication_acks: Arc::new(std::sync::Mutex::new(ack_receiver)),
    };
    let outcome = await_short_grace_terminal_result(
        &results_file,
        std::time::Instant::now() + std::time::Duration::from_secs(2),
        receiver,
        Some(&watchdog),
    );
    assert_eq!(outcome, ShortGraceCoordinatorOutcome::Published);
    control.join().unwrap();
    assert!(writer.join().unwrap().unwrap());
    assert_eq!(fs::read_to_string(&results_file).unwrap(), "unsat\n");
    assert_eq!(
        owner.budget().receipt().unwrap().live_persistent_bytes,
        4096
    );
}

/// B2 (ROUTER_PROGRAM 10.12): the original-hull `unsat` (`bab.single_hidden_hull:
/// root`, dark) published from its replay owner streamed NOTHING of its own, so
/// under the Proof profile its bundle carried a verdict for class 0 over
/// `nodes=0` (a cover gap the checker reports). With capture on, a refused
/// receipt (the owner cancelled after capture) claims nothing; a ratified one
/// streams exactly ONE `opaque` `original_hull` node over the whole region whose
/// classes are the receipt's own. Mutation target: the `stream_opaque_claim`
/// call in `translate_captured_verification` (removing it leaves zero nodes).
#[test]
fn original_hull_unsat_claims_exactly_its_receipt_classes_as_one_opaque_node() {
    use ny_propagate::certify::{
        with_thread_sink, AlphaRecord, FdStateRecord, LeafRecord, NodeClose, NodeRecord, ProofSink,
        SitesRecord, VerdictRecord,
    };
    use std::sync::Mutex;

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

    // The Proof arm's process flag (other proof-bundle tests of this binary
    // switch it on the same way); the thread sink isolates this test's nodes.
    crate::commands::proof_bundle::enable_hint_capture_for_test();

    // Refused: the owner is cancelled between capture and translation.
    let (captured, owner) = ordinary_dispatch(0.75, false, false);
    owner.cancel();
    let refused_sink = Shared::default();
    let (refused, _) = with_thread_sink(Box::new(refused_sink.clone()), || {
        translate_captured_verification(captured, || {
            panic!("sealed hull must not use graph/legacy JSON authority")
        })
    });
    assert!(
        !matches!(refused, VnncompResult::Unsat(_)),
        "a cancelled owner cannot publish"
    );
    let nodes = refused_sink.0.lock().unwrap().clone();
    assert!(
        nodes.is_empty(),
        "a refused receipt claims nothing: {nodes:?}"
    );

    // Ratified: the receipt publishes and claims exactly its classes.
    let (captured, _owner) = ordinary_dispatch(0.75, false, false);
    let sink = Shared::default();
    let (published, _) = with_thread_sink(Box::new(sink.clone()), || {
        translate_captured_verification(captured, || {
            panic!("sealed hull must not use graph/legacy JSON authority")
        })
    });
    let VnncompResult::Unsat(receipt) = &published else {
        panic!("the sealed hull replay must publish (else VOID)")
    };
    let (classes, _, source) = receipt.published_summary();
    assert_eq!(source, PublishedUnsatSource::OriginalHull);
    assert_eq!(classes, vec![0], "the hull's one clause");
    let nodes = sink.0.lock().unwrap().clone();
    assert_eq!(nodes.len(), 1, "{nodes:?}");
    let n = &nodes[0];
    assert_eq!(n.close, NodeClose::Opaque, "{n:?}");
    assert_eq!(n.src, "original_hull", "{n:?}");
    assert_eq!(n.dis.as_deref(), Some(classes.as_slice()), "{n:?}");
    assert!(
        n.lits.is_empty() && n.xlits.is_empty(),
        "whole region: {n:?}"
    );
}
