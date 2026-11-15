// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #attack-concurrent: the concurrent attack lane's publication protocol —
//! first verdict wins, a constructive witness outranks a bound-based `unsat`,
//! a `sat` reaches RESULTS_FILE only through the trusted gate, never two
//! verdicts, never a lost certified verdict.
//!
//! Every test runs on LOCAL precedence/claim state through the `_in` seams, so
//! no test depends on which other test touched the process globals first.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::SeqCst};
use std::sync::{Barrier, Mutex};

use super::*;
use crate::commands::beta_crown::ConcurrentSatPublication;

fn state() -> (Mutex<SatPrecedence>, AtomicBool) {
    (Mutex::new(SatPrecedence::default()), AtomicBool::new(false))
}

/// The gate's effect on the precedence record for an UPHELD candidate: the
/// adjudication marker is held across the gate and the confirmation is
/// latched inside it (`gate_sat_with_trusted_oracle` → `uphold_sat`).
fn gate_upholds(precedence: &Mutex<SatPrecedence>) {
    let _adjudication = SatAdjudicationGuard::enter_in(precedence);
    note_confirmed_sat_in(precedence);
}

/// SAT THEN UNSAT: the attack's confirmed `sat` is published first; the
/// proof's later `unsat` reaches RESULTS_FILE by neither door — the lane door
/// refuses on the confirmation, the main thread's claim is already taken, and
/// even the main path's verdict filter turns it into `unknown`.
#[test]
fn concurrent_attack_sat_outranks_a_later_proof_unsat() {
    let (precedence, claim) = state();
    let writes = Mutex::new(Vec::<&str>::new());
    gate_upholds(&precedence);
    assert_eq!(
        publish_attack_sat_in(&precedence, &claim, || {
            writes.lock().unwrap().push("sat");
            Ok(())
        }),
        AttackSatPublication::Published
    );
    assert_eq!(
        publish_lane_unsat_in(&precedence, &claim, || {
            writes.lock().unwrap().push("unsat");
            Ok(())
        }),
        LaneUnsatPublication::RefusedConfirmedSat
    );
    assert!(
        !claim_terminal_publication_in(&claim),
        "the main thread must find the single publication right taken"
    );
    assert_eq!(
        *writes.lock().unwrap(),
        vec!["sat"],
        "exactly one verdict, the sat"
    );
}

/// The main-thread half of the same rule: after a confirmed attack sat, an
/// `unsat` arriving at the main publication site is refused to `unknown`.
#[test]
fn proof_unsat_refused_after_confirmed_attack_sat() {
    let (precedence, claim) = state();
    gate_upholds(&precedence);
    assert_eq!(
        publish_attack_sat_in(&precedence, &claim, || Ok(())),
        AttackSatPublication::Published
    );
    let outward =
        refuse_unsat_against_confirmed_counterexample_in(&precedence, VnncompResult::test_unsat());
    assert!(
        matches!(outward, VnncompResult::Unknown),
        "a bound-based unsat must never be published against a confirmed witness"
    );
}

/// UNSAT WHILE THE ATTACK IS IN THE GATE: the proof's `unsat` defers while a
/// candidate is under adjudication; the candidate is DOWNGRADED (no
/// confirmation), the attack publishes nothing, and the certified `unsat` then
/// publishes — a deferral costs latency, never the verdict.
#[test]
fn proof_unsat_waits_for_inflight_adjudication_then_publishes_on_downgrade() {
    let (precedence, claim) = state();
    let writes = AtomicUsize::new(0);
    {
        let _adjudication = SatAdjudicationGuard::enter_in(&precedence);
        assert_eq!(
            publish_lane_unsat_in(&precedence, &claim, || {
                writes.fetch_add(1, SeqCst);
                Ok(())
            }),
            LaneUnsatPublication::RefusedAdjudicatingSat
        );
        // The gate downgrades: no confirmation is latched.
    }
    assert_eq!(writes.load(SeqCst), 0);
    // A downgraded candidate never reaches the publisher; were it to, the
    // missing confirmation refuses it.
    assert_eq!(
        publish_attack_sat_in(&precedence, &claim, || panic!("must not write")),
        AttackSatPublication::RefusedUnconfirmed
    );
    assert_eq!(
        publish_lane_unsat_in(&precedence, &claim, || {
            writes.fetch_add(1, SeqCst);
            Ok(())
        }),
        LaneUnsatPublication::Published,
        "the certified unsat must still be published"
    );
    assert_eq!(writes.load(SeqCst), 1);
}

/// A confirmed sat that finds the claim already taken is a CONFLICT: it is
/// refused before the write (never two verdicts) and the publisher reports
/// `ClaimTaken`, which `publish_concurrent_attack_sat` logs as an emergency.
#[test]
fn attack_sat_declines_when_claim_taken_and_records_conflict() {
    let (precedence, claim) = state();
    assert!(
        claim_terminal_publication_in(&claim),
        "another writer claims first"
    );
    gate_upholds(&precedence);
    assert_eq!(
        publish_attack_sat_in(&precedence, &claim, || panic!("must not write")),
        AttackSatPublication::RefusedClaimTaken
    );
    let declined = concurrent_attack_sat_pipeline(
        r#"{"status":"violated","counterexample_vnnlib":"((X_0 0.5)\n(Y_0 0.5))"}"#,
        || false,
        |witness| VnncompResult::Sat {
            witness: witness.map(str::to_string),
        },
        |_, _| {},
        |gated| gated,
        |_| AttackSatPublication::RefusedClaimTaken,
    );
    assert_eq!(declined.unwrap_err(), ConcurrentSatPublication::ClaimTaken);
}

/// SAT ONLY THROUGH THE GATE: without the gate's confirmation latch the
/// publisher refuses, writes nothing, and leaves the claim free.
#[test]
fn attack_sat_requires_the_gate_confirmation() {
    let (precedence, claim) = state();
    assert_eq!(
        publish_attack_sat_in(&precedence, &claim, || panic!("must not write")),
        AttackSatPublication::RefusedUnconfirmed
    );
    assert!(
        claim_terminal_publication_in(&claim),
        "a refused sat must not consume the publication right"
    );
}

/// A failed write gives the claim back (nothing reached disk), so the main
/// thread can still publish the verdict the row reached.
#[test]
fn attack_sat_write_failure_releases_the_claim() {
    let (precedence, claim) = state();
    gate_upholds(&precedence);
    assert!(matches!(
        publish_attack_sat_in(&precedence, &claim, || Err(anyhow!("disk full"))),
        AttackSatPublication::WriteFailed(_)
    ));
    assert!(claim_terminal_publication_in(&claim));
    assert!(
        ConcurrentSatPublication::WriteFailed(String::new()).returns_candidate_to_main(),
        "the candidate travels back to the main thread, which re-runs the gate"
    );
}

/// NEAR-SIMULTANEOUS: an attack thread (gate, then publish) and a proof
/// publisher race on the real primitives, with the candidate upheld on even
/// iterations and downgraded on odd ones. The proof side here is the
/// adjudication-aware LANE door (`publish_lane_unsat_in`, the margin-row
/// early exit's), retrying while a candidate is under adjudication. The MAIN
/// thread's door on a raced run is the synchronized one
/// (`publish_main_verdict_in`, pinned by the `main_door_*` tests below).
/// Every iteration ends with EXACTLY ONE verdict on disk; a downgraded
/// candidate always leaves the certified `unsat` published; an upheld one
/// either wins or — only when the proof published before the candidate even
/// entered the gate — is refused as a CONFLICT rather than written twice.
#[test]
fn near_simultaneous_attack_sat_and_proof_unsat_never_publish_two_verdicts() {
    let mut sat_won = 0usize;
    let mut unsat_won = 0usize;
    for iteration in 0..2_000 {
        let upheld = iteration % 2 == 0;
        let (precedence, claim) = state();
        let writes = Mutex::new(Vec::<&str>::new());
        let barrier = Barrier::new(2);
        let mut attack_outcome = None;
        std::thread::scope(|scope| {
            let attack = scope.spawn(|| {
                barrier.wait();
                let adjudication = SatAdjudicationGuard::enter_in(&precedence);
                if !upheld {
                    drop(adjudication);
                    return None;
                }
                note_confirmed_sat_in(&precedence);
                let outcome = publish_attack_sat_in(&precedence, &claim, || {
                    writes.lock().unwrap().push("sat");
                    Ok(())
                });
                drop(adjudication);
                Some(outcome)
            });
            barrier.wait();
            while publish_lane_unsat_in(&precedence, &claim, || {
                writes.lock().unwrap().push("unsat");
                Ok(())
            }) == LaneUnsatPublication::RefusedAdjudicatingSat
            {
                std::thread::yield_now();
            }
            attack_outcome = attack.join().expect("attack thread");
        });
        let writes = writes.into_inner().unwrap();
        assert_eq!(
            writes.len(),
            1,
            "iteration {iteration}: verdicts on disk = {writes:?}"
        );
        if upheld {
            match (writes[0], attack_outcome) {
                ("sat", Some(AttackSatPublication::Published)) => sat_won += 1,
                ("unsat", Some(AttackSatPublication::RefusedClaimTaken)) => unsat_won += 1,
                other => panic!("iteration {iteration}: illegal upheld outcome {other:?}"),
            }
        } else {
            assert_eq!(
                writes[0], "unsat",
                "iteration {iteration}: certified unsat lost"
            );
        }
    }
    assert!(
        sat_won > 0,
        "the race must exercise the sat-first order (got {sat_won})"
    );
    let _ = unsat_won;
}

/// The concurrent publisher replays the MAIN path's sat pipeline in the main
/// path's order: parse → QDQ latch → trusted gate → side-file record →
/// VNN-LIB 2.0 normalization (after the gate, as the entry point does) →
/// one-critical-section publication; and it stops at the first refusal.
#[test]
fn concurrent_publisher_replays_the_main_sat_pipeline_order() {
    let json = r#"{"status":"violated","counterexample_vnnlib":"((X_0 0.5)\n(Y_0 0.5))"}"#;
    let run = |qdq: bool, gate_upholds: bool, normalize_keeps: bool| {
        let steps = Mutex::new(Vec::<String>::new());
        let outcome = concurrent_attack_sat_pipeline(
            json,
            || {
                steps.lock().unwrap().push("qdq".into());
                qdq
            },
            |witness| {
                steps
                    .lock()
                    .unwrap()
                    .push(format!("gate:{}", witness.is_some()));
                if gate_upholds {
                    VnncompResult::Sat {
                        witness: witness.map(str::to_string),
                    }
                } else {
                    VnncompResult::Unknown
                }
            },
            |consulted, upheld| {
                steps
                    .lock()
                    .unwrap()
                    .push(format!("record:{consulted}:{upheld}"));
            },
            |gated| {
                steps.lock().unwrap().push("normalize".into());
                if normalize_keeps {
                    gated
                } else {
                    VnncompResult::Unknown
                }
            },
            |_| {
                steps.lock().unwrap().push("publish".into());
                AttackSatPublication::Published
            },
        );
        (outcome, steps.into_inner().unwrap())
    };

    let (outcome, steps) = run(false, true, true);
    assert!(matches!(outcome, Ok(VnncompResult::Sat { .. })));
    assert_eq!(
        steps,
        [
            "qdq",
            "gate:true",
            "record:true:true",
            "normalize",
            "publish"
        ]
    );

    let (outcome, steps) = run(true, true, true);
    assert_eq!(outcome.unwrap_err(), ConcurrentSatPublication::Withheld);
    assert_eq!(steps, ["qdq"], "a withheld verdict never reaches the gate");

    let (outcome, steps) = run(false, false, true);
    assert_eq!(outcome.unwrap_err(), ConcurrentSatPublication::Downgraded);
    assert_eq!(steps, ["qdq", "gate:true", "record:true:false"]);

    let (outcome, steps) = run(false, true, false);
    assert_eq!(outcome.unwrap_err(), ConcurrentSatPublication::Downgraded);
    assert_eq!(steps, ["qdq", "gate:true", "record:true:true", "normalize"]);

    // A JSON that is not a violation never reaches any stage.
    let steps = Mutex::new(0usize);
    let outcome = concurrent_attack_sat_pipeline(
        r#"{"status":"verified"}"#,
        || {
            *steps.lock().unwrap() += 1;
            false
        },
        |_| panic!("gate"),
        |_, _| panic!("record"),
        |_| panic!("normalize"),
        |_| panic!("publish"),
    );
    assert_eq!(outcome.unwrap_err(), ConcurrentSatPublication::Downgraded);
    assert_eq!(*steps.lock().unwrap(), 0);
}

/// The target is armed on the ordinary route only; unarmed, the publisher
/// answers `NotArmed` before doing anything (the race then hands the
/// candidate to the main thread exactly as the serial site does).
#[test]
fn unarmed_publisher_answers_not_armed() {
    assert!(ConcurrentSatPublication::NotArmed.returns_candidate_to_main());
    assert!(!ConcurrentSatPublication::Downgraded.returns_candidate_to_main());
    assert!(!ConcurrentSatPublication::ClaimTaken.returns_candidate_to_main());
    assert!(
        !ConcurrentSatPublication::DoorClosed.returns_candidate_to_main(),
        "a candidate refused at a closed door never returns: the main verdict is committed"
    );
}

/// A refusal at the closed attack door surfaces as `DoorClosed` from the
/// pipeline (never as a downgrade or a conflict), after the gate ran.
#[test]
fn attack_publication_after_the_door_closed_is_door_closed() {
    let declined = concurrent_attack_sat_pipeline(
        r#"{"status":"violated","counterexample_vnnlib":"((X_0 0.5)\n(Y_0 0.5))"}"#,
        || false,
        |witness| VnncompResult::Sat {
            witness: witness.map(str::to_string),
        },
        |_, _| {},
        |gated| gated,
        |_| AttackSatPublication::RefusedDoorClosed,
    );
    assert_eq!(declined.unwrap_err(), ConcurrentSatPublication::DoorClosed);
}

// ---------------------------------------------------------------------------
//  Adversarial review (review/attack-concurrency): delays injected INSIDE the
//  critical sections, and the one door that is not adjudication-aware.
// ---------------------------------------------------------------------------

/// A DELAY INSIDE THE ATTACK'S WRITE: the #sat-precedence lock is held across
/// the whole write, so a lane `unsat` that arrives while the attack's `sat` is
/// half-way to disk BLOCKS on the lock (it cannot interleave a second write)
/// and, once the write lands, is refused on the confirmation. Exactly one
/// verdict reaches the file, and it is the confirmed one. The ordering
/// assertion is causal (the lane returns only after the write finished), not
/// a wall-time threshold, so a descheduled test thread cannot make it flaky.
#[test]
fn a_lane_unsat_arriving_mid_write_blocks_then_is_refused() {
    let (precedence, claim) = state();
    let writes = Mutex::new(Vec::<&str>::new());
    let write_done = Mutex::new(None::<std::time::Instant>);
    let in_write = Barrier::new(2);
    let mut lane = None;
    let mut lane_returned = None;
    std::thread::scope(|scope| {
        let attack = scope.spawn(|| {
            gate_upholds(&precedence);
            publish_attack_sat_in(&precedence, &claim, || {
                in_write.wait();
                std::thread::sleep(std::time::Duration::from_millis(200));
                writes.lock().unwrap().push("sat");
                *write_done.lock().unwrap() = Some(std::time::Instant::now());
                Ok(())
            })
        });
        in_write.wait();
        lane = Some(publish_lane_unsat_in(&precedence, &claim, || {
            writes.lock().unwrap().push("unsat");
            Ok(())
        }));
        lane_returned = Some(std::time::Instant::now());
        assert_eq!(
            attack.join().expect("attack thread"),
            AttackSatPublication::Published
        );
    });
    assert_eq!(lane, Some(LaneUnsatPublication::RefusedConfirmedSat));
    let write_done = write_done.lock().unwrap().expect("the attack's write ran");
    assert!(
        lane_returned.expect("the lane ran") >= write_done,
        "the lane returned before the attack's write finished: the lock was not held across it"
    );
    assert_eq!(*writes.lock().unwrap(), vec!["sat"]);
}

/// THE DARK MAIN DOOR IS UNCHANGED, AND IS NOT ADJUDICATION-AWARE. On every
/// run where no race spawned the main publication site keeps its historical
/// door verbatim: the verdict filter reads only the LATCHED confirmation and
/// the claim is a bare swap, so neither defers while a candidate is inside
/// the trusted gate. That is why a raced run does NOT use it: the race's join
/// is bounded (follow-up (a)), so an attack adjudication may outlive the race,
/// and the raced run's main door is the synchronized
/// `publish_main_verdict_in` instead (the `main_door_*` tests). This pins the
/// dark door's semantics (default bit-identity).
#[test]
fn dark_main_door_reads_only_the_latched_confirmation() {
    let (precedence, claim) = state();
    let _gate_in_flight = SatAdjudicationGuard::enter_in(&precedence);
    let outward =
        refuse_unsat_against_confirmed_counterexample_in(&precedence, VnncompResult::test_unsat());
    assert!(
        matches!(outward, VnncompResult::Unsat(_)),
        "the main door does not defer on an in-flight adjudication"
    );
    assert!(
        claim_terminal_publication_in(&claim),
        "and its claim is a bare swap"
    );
}

// ---------------------------------------------------------------------------
//  Follow-up (b) (feat/attack-concurrency-followups): the main thread's door
//  SYNCHRONIZES with an in-flight concurrent-attack adjudication instead of
//  relying on the race's join — because (a) bounds that join, an abandoned
//  adjudication can now outlive the race.
// ---------------------------------------------------------------------------

/// The door primitives on local state.
fn door_state() -> (Mutex<SatPrecedence>, std::sync::Condvar, AtomicBool) {
    (
        Mutex::new(SatPrecedence::default()),
        std::sync::Condvar::new(),
        AtomicBool::new(false),
    )
}

/// An in-flight attack adjudication that ends in a DOWNGRADE 200 ms later:
/// the main door waits for it (causally: its write lands after the attack
/// left the door), then publishes the certified `unsat` — exactly one verdict.
#[test]
fn main_door_waits_for_an_inflight_attack_adjudication_then_publishes() {
    let (precedence, door, claim) = door_state();
    let writes = Mutex::new(Vec::<String>::new());
    let left_door = Mutex::new(None::<std::time::Instant>);
    let wrote_at = Mutex::new(None::<std::time::Instant>);
    let inflight = AttackPublicationGuard::enter_in(&precedence, &door).expect("door open");
    let mut report = None;
    std::thread::scope(|scope| {
        scope.spawn(|| {
            {
                let _gate = SatAdjudicationGuard::enter_in(&precedence);
                std::thread::sleep(std::time::Duration::from_millis(200));
                // Downgraded: nothing latched.
            }
            *left_door.lock().unwrap() = Some(std::time::Instant::now());
            drop(inflight);
        });
        report = Some(publish_main_verdict_in(
            &precedence,
            &door,
            &claim,
            VnncompResult::test_unsat(),
            Some(std::time::Instant::now() + std::time::Duration::from_secs(10)),
            |result| {
                *wrote_at.lock().unwrap() = Some(std::time::Instant::now());
                writes.lock().unwrap().push(result.token().to_string());
                Ok(true)
            },
        ));
    });
    let report = report.expect("the door ran");
    assert!(report.owns_publication);
    assert!(matches!(report.written, Some(Ok(true))));
    assert!(matches!(report.result, VnncompResult::Unsat(_)));
    assert_eq!(report.inflight_at_close, 0);
    let left = left_door.lock().unwrap().expect("the attack left the door");
    let wrote = wrote_at.lock().unwrap().expect("the main door wrote");
    assert!(
        wrote >= left,
        "the main door wrote while an attack adjudication was still in flight"
    );
    assert_eq!(*writes.lock().unwrap(), vec!["unsat".to_string()]);
}

/// An in-flight attack adjudication that UPHOLDS its witness: the main door
/// waits, the attack's confirmed `sat` is published first, and the door's
/// `unsat` is never written (in production the attack's publication ends the
/// process; here the door finds the claim taken).
#[test]
fn main_door_defers_to_an_upheld_inflight_attack_sat() {
    let (precedence, door, claim) = door_state();
    let writes = Mutex::new(Vec::<String>::new());
    let inflight = AttackPublicationGuard::enter_in(&precedence, &door).expect("door open");
    let mut attack = None;
    let mut report = None;
    std::thread::scope(|scope| {
        let handle = scope.spawn(|| {
            std::thread::sleep(std::time::Duration::from_millis(150));
            gate_upholds(&precedence);
            let outcome = publish_attack_sat_in(&precedence, &claim, || {
                writes.lock().unwrap().push("sat".to_string());
                Ok(())
            });
            drop(inflight);
            outcome
        });
        report = Some(publish_main_verdict_in(
            &precedence,
            &door,
            &claim,
            VnncompResult::test_unsat(),
            Some(std::time::Instant::now() + std::time::Duration::from_secs(10)),
            |result| {
                writes.lock().unwrap().push(result.token().to_string());
                Ok(true)
            },
        ));
        attack = Some(handle.join().expect("attack thread"));
    });
    assert_eq!(attack, Some(AttackSatPublication::Published));
    let report = report.expect("the door ran");
    assert!(!report.owns_publication, "the attack's sat owns the file");
    assert_eq!(*writes.lock().unwrap(), vec!["sat".to_string()]);
}

/// THE DOOR'S OWN DEADLINE. An adjudication still in flight at the door's
/// hard deadline is abandoned: the door CLOSES (latched), the certified
/// verdict publishes regardless, a late confirmation of the abandoned
/// candidate can no longer write, and no new attack publication may start.
#[test]
fn main_door_closes_the_attack_door_at_its_deadline_and_publishes_regardless() {
    let (precedence, door, claim) = door_state();
    let writes = Mutex::new(Vec::<String>::new());
    let inflight = AttackPublicationGuard::enter_in(&precedence, &door).expect("door open");
    let started = std::time::Instant::now();
    let report = publish_main_verdict_in(
        &precedence,
        &door,
        &claim,
        VnncompResult::test_unsat(),
        Some(started + std::time::Duration::from_millis(300)),
        |result| {
            writes.lock().unwrap().push(result.token().to_string());
            Ok(true)
        },
    );
    assert!(
        started.elapsed() >= std::time::Duration::from_millis(290),
        "the door must wait for the in-flight adjudication up to its deadline"
    );
    assert_eq!(report.inflight_at_close, 1);
    assert!(report.owns_publication);
    assert_eq!(*writes.lock().unwrap(), vec!["unsat".to_string()]);
    // The abandoned candidate confirms late: it may not write.
    gate_upholds(&precedence);
    assert_eq!(
        publish_attack_sat_in(&precedence, &claim, || panic!("must not write")),
        AttackSatPublication::RefusedDoorClosed
    );
    drop(inflight);
    assert!(
        AttackPublicationGuard::enter_in(&precedence, &door).is_none(),
        "a closed door admits no new attack publication"
    );
}

/// The #sat-precedence filter runs INSIDE the door's critical section: a
/// confirmation latched before the door turns its `unsat` into `unknown`.
#[test]
fn main_door_refuses_unsat_against_a_confirmation_latched_before_it() {
    let (precedence, door, claim) = door_state();
    gate_upholds(&precedence);
    let report = publish_main_verdict_in(
        &precedence,
        &door,
        &claim,
        VnncompResult::test_unsat(),
        Some(std::time::Instant::now() + std::time::Duration::from_secs(10)),
        |_| Ok(true),
    );
    assert!(matches!(report.result, VnncompResult::Unknown));
    assert!(report.owns_publication);
}

/// With NO attack publication in flight the synchronized door decides
/// exactly what the dark door decides (#sat-precedence filter, then the
/// claim, then the write) for every verdict, confirmation state and claim
/// state — and never waits.
#[test]
fn main_door_without_inflight_attacks_matches_the_dark_door() {
    let verdicts = || {
        [
            VnncompResult::test_unsat(),
            VnncompResult::Sat {
                witness: Some("((X_0 0.5))".to_string()),
            },
            VnncompResult::Unknown,
            VnncompResult::Timeout,
        ]
    };
    for (index, verdict) in verdicts().into_iter().enumerate() {
        for confirmed in [false, true] {
            for claim_taken in [false, true] {
                let (precedence, door, claim) = door_state();
                let (dark_precedence, _, dark_claim) = door_state();
                if confirmed {
                    gate_upholds(&precedence);
                    gate_upholds(&dark_precedence);
                }
                if claim_taken {
                    claim.store(true, SeqCst);
                    dark_claim.store(true, SeqCst);
                }
                // The dark door: filter, claim, write.
                let dark_result = refuse_unsat_against_confirmed_counterexample_in(
                    &dark_precedence,
                    verdicts()[index].clone(),
                );
                let dark_owns = claim_terminal_publication_in(&dark_claim);
                let started = std::time::Instant::now();
                let report = publish_main_verdict_in(
                    &precedence,
                    &door,
                    &claim,
                    verdict.clone(),
                    Some(started + std::time::Duration::from_secs(10)),
                    |_| Ok(true),
                );
                assert!(started.elapsed() < std::time::Duration::from_secs(1));
                assert_eq!(report.result.token(), dark_result.token());
                assert_eq!(report.owns_publication, dark_owns);
                assert_eq!(report.written.is_some(), dark_owns);
                assert_eq!(report.inflight_at_close, 0);
            }
        }
    }
}

// ---------------------------------------------------------------------------
//  review(attack-conc-followups): the attack-door entry is taken at
//  PREPARATION, on the attack thread inside the race scope — not by the
//  detached job — so the main door's "nothing in flight" is a fact by
//  construction, not by thread-spawn timing.
// ---------------------------------------------------------------------------

/// Leaked (`'static`) door state: a prepared job owns its entry and may run
/// on a detached thread, so the seam takes `'static` references.
fn leaked_door_state() -> (&'static Mutex<SatPrecedence>, &'static std::sync::Condvar) {
    (
        Box::leak(Box::new(Mutex::new(SatPrecedence::default()))),
        Box::leak(Box::new(std::sync::Condvar::new())),
    )
}

fn race_report() -> crate::commands::beta_crown::ConcurrentSatReport {
    crate::commands::beta_crown::ConcurrentSatReport {
        site: crate::commands::beta_crown::attack_concurrent::RaceSite::GraphUpfront,
        attack_wall: std::time::Duration::ZERO,
        attack_cpu: None,
        width_pool: 1,
    }
}

/// When [`recording_adjudicator`]'s gate left the attack door (recorded
/// BEFORE the entry is released, so a door write after it is causal).
static RECORDED_GATE_LEFT_DOOR: Mutex<Option<std::time::Instant>> = Mutex::new(None);

/// [`slow_downgrading_adjudicator`] that also records when it leaves the door.
fn recording_adjudicator(
    entered: AttackPublicationGuard<'static>,
    json: &str,
    _report: &crate::commands::beta_crown::ConcurrentSatReport,
) -> ConcurrentSatPublication {
    assert_eq!(
        json, "{\"status\":\"violated\"}",
        "the job adjudicates the prepared JSON"
    );
    std::thread::sleep(std::time::Duration::from_millis(300));
    *RECORDED_GATE_LEFT_DOOR.lock().unwrap() = Some(std::time::Instant::now());
    drop(entered);
    ConcurrentSatPublication::Downgraded
}

/// The adjudicator a prepared job runs: a 300 ms gate that downgrades,
/// releasing the door entry it was handed only when it returns.
fn slow_downgrading_adjudicator(
    entered: AttackPublicationGuard<'static>,
    json: &str,
    _report: &crate::commands::beta_crown::ConcurrentSatReport,
) -> ConcurrentSatPublication {
    assert_eq!(
        json, "{\"status\":\"violated\"}",
        "the job adjudicates the prepared JSON"
    );
    std::thread::sleep(std::time::Duration::from_millis(300));
    drop(entered);
    ConcurrentSatPublication::Downgraded
}

/// PREPARATION ENTERS THE DOOR. Right after `prepare_concurrent_attack_sat_in`
/// returns — before the job has run, been spawned, or even been looked at —
/// the publication is already counted in flight, so a main door reached at
/// that instant WAITS for it (its write lands after the job's gate ended).
/// RED on 0d091a88c's shape (entry inside the job): the count is 0 until the
/// detached thread happens to start the job, and a race that stopped waiting
/// before then let the door close over a candidate never adjudicated.
#[test]
fn prepared_candidate_is_inside_the_attack_door_before_its_job_runs() {
    use crate::commands::beta_crown::attack_concurrent::PreparedSat;
    let (precedence, door) = leaked_door_state();
    let claim = AtomicBool::new(false);
    let prepared = prepare_concurrent_attack_sat_in(
        precedence,
        door,
        "{\"status\":\"violated\"}".to_string(),
        race_report(),
        recording_adjudicator,
    );
    assert_eq!(
        lock_sat_precedence(precedence).attack_inflight,
        1,
        "a prepared candidate must be inside the attack door BEFORE its job runs"
    );
    let PreparedSat::Adjudicate(job) = prepared else {
        panic!("an open door prepares an adjudication job");
    };
    let wrote_at = Mutex::new(None::<std::time::Instant>);
    let mut report = None;
    std::thread::scope(|scope| {
        // The detached gate thread starts LATE (100 ms) — the window the
        // entry-in-the-job shape left open.
        scope.spawn(|| {
            std::thread::sleep(std::time::Duration::from_millis(100));
            assert_eq!(job(), ConcurrentSatPublication::Downgraded);
        });
        report = Some(publish_main_verdict_in(
            precedence,
            door,
            &claim,
            VnncompResult::test_unsat(),
            Some(std::time::Instant::now() + std::time::Duration::from_secs(10)),
            |_| {
                *wrote_at.lock().unwrap() = Some(std::time::Instant::now());
                Ok(true)
            },
        ));
    });
    let report = report.expect("the door ran");
    assert!(report.owns_publication);
    assert_eq!(report.inflight_at_close, 0);
    assert!(matches!(report.result, VnncompResult::Unsat(_)));
    let left = RECORDED_GATE_LEFT_DOOR
        .lock()
        .unwrap()
        .expect("the job's gate ran and left the door");
    let wrote = wrote_at.lock().unwrap().expect("the door wrote");
    assert!(
        wrote >= left,
        "the main door wrote before the prepared candidate's adjudication left the door"
    );
    assert_eq!(lock_sat_precedence(precedence).attack_inflight, 0);
}

/// A prepared job that is DROPPED unrun (its detached thread never took it)
/// releases its entry, so the door cannot wait on a phantom; and a door that
/// is already CLOSED refuses at preparation — no job, no entry.
#[test]
fn prepared_entry_is_released_unrun_and_refused_at_a_closed_door() {
    use crate::commands::beta_crown::attack_concurrent::PreparedSat;
    let (precedence, door) = leaked_door_state();
    let prepared = prepare_concurrent_attack_sat_in(
        precedence,
        door,
        "{\"status\":\"violated\"}".to_string(),
        race_report(),
        slow_downgrading_adjudicator,
    );
    assert!(matches!(prepared, PreparedSat::Adjudicate(_)));
    assert_eq!(lock_sat_precedence(precedence).attack_inflight, 1);
    drop(prepared);
    assert_eq!(
        lock_sat_precedence(precedence).attack_inflight,
        0,
        "an unrun job's entry must be released when it is dropped"
    );
    lock_sat_precedence(precedence).attack_door_closed = true;
    let refused = prepare_concurrent_attack_sat_in(
        precedence,
        door,
        "{\"status\":\"violated\"}".to_string(),
        race_report(),
        slow_downgrading_adjudicator,
    );
    assert!(matches!(
        refused,
        PreparedSat::Declined(ConcurrentSatPublication::DoorClosed)
    ));
    assert_eq!(lock_sat_precedence(precedence).attack_inflight, 0);
}
