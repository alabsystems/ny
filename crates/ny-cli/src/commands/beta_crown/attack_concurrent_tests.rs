// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #attack-concurrent: the race primitive, its window arithmetic, and its
//! admission chain, on injected attack / proof / publish closures.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::SeqCst};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use ndarray::{arr1, ArrayD};
use ny_propagate::{BabVerificationStatus, BetaCrownResult, PgdCancelToken};

use super::*;

fn result(status: BabVerificationStatus) -> BetaCrownResult {
    BetaCrownResult {
        result: status,
        domains_explored: 0,
        domains_verified: 0,
        cuts_generated: 0,
        max_depth_reached: 0,
        time_elapsed: Duration::ZERO,
        output_bounds: None,
    }
}

fn hit() -> AttackHit {
    Some((arr1(&[0.75_f32]).into_dyn(), arr1(&[0.75_f32]).into_dyn()))
}

fn window(window_ms: u64) -> RaceWindow {
    RaceWindow {
        slice: Duration::from_millis(window_ms / 2),
        window: Duration::from_millis(window_ms),
        host_threads: 2,
        width_pool: 2,
    }
}

fn plan(publish: PublishFn<'_>, window_ms: u64) -> RacePlan<'_> {
    let now = Instant::now();
    RacePlan {
        site: RaceSite::GraphUpfront,
        window: window(window_ms),
        site_start: now,
        slice_deadline: now + Duration::from_millis(window_ms / 2),
        overall_deadline: Some(now + Duration::from_millis(window_ms) + Duration::from_secs(60)),
        lane_start: now,
        publish,
    }
}

/// An adjudication job (the detached half of the publish hook).
fn adjudicate_with<F>(job: F) -> PreparedSat
where
    F: FnOnce() -> ConcurrentSatPublication + Send + 'static,
{
    PreparedSat::Adjudicate(Box::new(job))
}

fn pool() -> rayon::ThreadPool {
    build_pool(2).expect("a two-thread private pool")
}

/// An attack that polls its token and deadline like a PGD restart loop and
/// records how it stopped.
fn polling_attack<'a>(
    saw_cancel: &'a AtomicBool,
    ran_to_deadline: &'a AtomicBool,
) -> impl Fn(PgdCancelToken, Instant) -> Result<AttackHit> + Sync + 'a {
    move |token, deadline| loop {
        if token.is_cancelled() {
            saw_cancel.store(true, SeqCst);
            return Ok(None);
        }
        if Instant::now() >= deadline {
            ran_to_deadline.store(true, SeqCst);
            return Ok(None);
        }
        std::thread::sleep(Duration::from_millis(1));
    }
}

fn never_publish(_: &BetaCrownResult, _: &ConcurrentSatReport) -> PreparedSat {
    panic!("no candidate exists, so nothing may be published")
}

/// `W * k_a / (N + k_a) >= slice` for every width pair: while both sides are
/// busy the attack's CFS share is `k_a / (N + k_a)` of the machine, so the
/// stretched window gives it at least the CPU-seconds the serial slice gave
/// an attack of width `<= k_a` (rounding: at most one nanosecond per thread).
#[test]
fn stretched_window_conserves_attack_cpu_seconds() {
    for slice_ms in [1_u64, 37, 2_300, 44_400, 57_000] {
        let slice = Duration::from_millis(slice_ms);
        for host in 1..=32usize {
            for width in 1..=32usize {
                let w = stretched_window(slice, host, width);
                let lhs = w.as_nanos() * width as u128;
                let rhs = slice.as_nanos() * (host + width) as u128;
                assert!(
                    lhs + width as u128 > rhs,
                    "slice={slice_ms}ms N={host} k_a={width}: W={w:?} under-provisions"
                );
                assert!(w >= slice, "the window is never shorter than today's slice");
            }
        }
    }
    // The static split (k_a = N) doubles the slice.
    assert_eq!(
        stretched_window(Duration::from_secs(44), 20, 20),
        Duration::from_secs(88)
    );
}

/// The stretch is admitted only when the UNCAPPED window fits before
/// `overall_deadline - GATE_RESERVE`; no slice (unbounded, or expired) and no
/// overall deadline both refuse.
#[test]
fn stretch_must_fit_or_stand_down() {
    let start = Instant::now();
    let slice = start + Duration::from_secs(10);
    // W = 20 s; 20 + 5 <= 30 fits.
    let fits = plan_window(
        start,
        Some(slice),
        Some(start + Duration::from_secs(30)),
        4,
        4,
    )
    .expect("fits");
    assert_eq!(fits.window, Duration::from_secs(20));
    assert_eq!(fits.slice, Duration::from_secs(10));
    assert!((fits.stretch() - 2.0).abs() < 1e-9);
    // 20 + 5 > 24: refused, never capped.
    assert_eq!(
        plan_window(
            start,
            Some(slice),
            Some(start + Duration::from_secs(24)),
            4,
            4
        ),
        Err(AdmitRoute::StretchDoesNotFit)
    );
    assert_eq!(
        plan_window(start, Some(slice), None, 4, 4),
        Err(AdmitRoute::StretchDoesNotFit)
    );
    assert_eq!(
        plan_window(start, None, Some(start + Duration::from_secs(99)), 4, 4),
        Err(AdmitRoute::NoSlice)
    );
    assert_eq!(
        plan_window(
            start,
            Some(start),
            Some(start + Duration::from_secs(99)),
            4,
            4
        ),
        Err(AdmitRoute::NoSlice)
    );
}

fn admitted_facts() -> AdmissionFacts {
    AdmissionFacts {
        publisher_armed: true,
        ..AdmissionFacts::default()
    }
}

fn some_window() -> std::result::Result<RaceWindow, AdmitRoute> {
    Ok(window(100))
}

/// CR-X: the concurrent margin-row worker owns the concurrency; the attack
/// lane stands down (cifar100 / tinyimagenet stay bit-identical).
#[test]
fn margin_row_concurrent_stands_down_attack_concurrent() {
    let facts = AdmissionFacts {
        margin_row_lane_live: true,
        ..admitted_facts()
    };
    assert_eq!(
        admit(&facts, some_window()),
        Err(AdmitRoute::MarginRowOwnsConcurrency)
    );
}

/// An attack and a proof on the same accelerator stand down (vgg's wgpu proof
/// device, a GPU BaB beside a steered attack); an accelerated attack beside a
/// CPU proof, or a CPU attack beside anything, is not shared.
#[test]
fn shared_gpu_device_stands_down() {
    let facts = AdmissionFacts {
        shared_device: true,
        ..admitted_facts()
    };
    assert_eq!(admit(&facts, some_window()), Err(AdmitRoute::SharedDevice));

    let cpu = NamedEngine("faer-cpu");
    assert!(!engine_is_accelerator(&ny_core::NaiveCpuGemmEngine));
    let gpu = NamedEngine("wgpu-attack-steering");
    let cuda = NamedEngine("cuda-cublas-rn");
    assert!(!engine_is_accelerator(&cpu));
    assert!(engine_is_accelerator(&gpu) && engine_is_accelerator(&cuda));
    // Shared: both sides on an accelerator.
    assert!(shared_device(Some(&gpu), Some(&cuda), false));
    assert!(
        shared_device(Some(&gpu), None, true),
        "a GPU BaB is a GPU proof"
    );
    assert!(shared_device(Some(&cuda), Some(&cuda), false));
    // Not shared: a steered attack beside a CPU proof, or a CPU attack.
    assert!(!shared_device(Some(&gpu), None, false));
    assert!(!shared_device(Some(&gpu), Some(&cpu), false));
    assert!(!shared_device(Some(&cpu), Some(&cuda), true));
    assert!(!shared_device(None, Some(&cuda), true));
}

/// A provenance-only engine double for [`shared_gpu_device_stands_down`].
struct NamedEngine(&'static str);

impl ny_core::GemmEngine for NamedEngine {
    fn backend_provenance(&self) -> &'static str {
        self.0
    }

    fn gemm_f32(
        &self,
        _m: usize,
        _k: usize,
        _n: usize,
        _a: &[f32],
        _b: &[f32],
    ) -> ny_core::Result<Vec<f32>> {
        unreachable!("provenance-only double")
    }
}

/// A `Sign` net keeps the BNN lanes' serial order.
#[test]
fn bnn_net_stays_serial() {
    let facts = AdmissionFacts {
        sign_net: true,
        ..admitted_facts()
    };
    assert_eq!(admit(&facts, some_window()), Err(AdmitRoute::BnnSerial));
}

/// The admission chain's ORDER: publisher, margin-row, device, BNN, QDQ, then
/// the window; every refusal is reported by its own token.
#[test]
fn admission_chain_is_ordered_and_total() {
    let all_refusing = AdmissionFacts {
        publisher_armed: false,
        margin_row_lane_live: true,
        shared_device: true,
        sign_net: true,
        qdq_report_only: true,
    };
    assert_eq!(
        admit(&all_refusing, some_window()),
        Err(AdmitRoute::PublisherUnarmed)
    );
    let mut facts = AdmissionFacts {
        publisher_armed: true,
        ..all_refusing
    };
    assert_eq!(
        admit(&facts, some_window()),
        Err(AdmitRoute::MarginRowOwnsConcurrency)
    );
    facts.margin_row_lane_live = false;
    assert_eq!(admit(&facts, some_window()), Err(AdmitRoute::SharedDevice));
    facts.shared_device = false;
    assert_eq!(admit(&facts, some_window()), Err(AdmitRoute::BnnSerial));
    facts.sign_net = false;
    assert_eq!(admit(&facts, some_window()), Err(AdmitRoute::QdqReportOnly));
    facts.qdq_report_only = false;
    assert_eq!(
        admit(&facts, Err(AdmitRoute::StretchDoesNotFit)),
        Err(AdmitRoute::StretchDoesNotFit)
    );
    assert_eq!(admit(&facts, some_window()), Ok(window(100)));
    // A context-less site (interactive, nested, test faces) is unarmed.
    assert!(!ConcurrentAttackCtx::admission_facts(None, false, false).publisher_armed);
}

/// A DECIDED proof cancels the attack through its token, and the join is
/// prompt (one poll), not a full window.
#[test]
fn decided_proof_cancels_the_attack_within_one_poll() {
    let saw_cancel = AtomicBool::new(false);
    let ran_to_deadline = AtomicBool::new(false);
    let started = Instant::now();
    let out = run_race(
        plan(&never_publish, 60_000),
        pool(),
        polling_attack(&saw_cancel, &ran_to_deadline),
        || {
            std::thread::sleep(Duration::from_millis(50));
            Ok(result(BabVerificationStatus::Verified))
        },
    )
    .expect("race");
    assert!(matches!(out.result, BabVerificationStatus::Verified));
    assert!(
        saw_cancel.load(SeqCst),
        "the decided proof must cancel the attack"
    );
    assert!(!ran_to_deadline.load(SeqCst));
    assert!(
        started.elapsed() < Duration::from_secs(10),
        "the join must not wait out the 60 s window ({:?})",
        started.elapsed()
    );
}

/// An UNDECIDED proof that outlasts the stretched window never cancels the
/// attack: the attack keeps its whole window `W`, so no sat search time is
/// lost to a proof that gave up. (An undecided proof that gives up EARLY cuts
/// the attack at its serial-equivalent deadline instead:
/// `early_undecided_proof_shrinks_the_attack_to_its_serial_equivalent_window`.)
#[test]
fn late_undecided_proof_never_cancels_the_attack() {
    for undecided in [
        Ok(result(BabVerificationStatus::Timeout)),
        Ok(result(BabVerificationStatus::Unknown {
            reason: "test".to_string(),
        })),
        Err(anyhow::anyhow!("proof error")),
    ] {
        let saw_cancel = AtomicBool::new(false);
        let ran_to_deadline = AtomicBool::new(false);
        let started = Instant::now();
        let is_err = undecided.is_err();
        let (out, observed) = run_race_observed(
            plan(&never_publish, 300),
            pool(),
            polling_attack(&saw_cancel, &ran_to_deadline),
            move || {
                std::thread::sleep(Duration::from_millis(320));
                undecided
            },
        );
        let observed = observed.expect("the race ran");
        assert_eq!(observed.cancel, CancelReason::None);
        assert_eq!(observed.outcome, AttackOutcome::Deadline);
        assert_eq!(out.is_err(), is_err, "the proof's own result is returned");
        assert!(
            !saw_cancel.load(SeqCst),
            "an undecided proof must not cancel"
        );
        assert!(
            ran_to_deadline.load(SeqCst),
            "the attack keeps its whole window"
        );
        assert!(started.elapsed() >= Duration::from_millis(290));
    }
}

/// A proof that PANICS cancels the attack (Drop guard), so the panic
/// propagates after one poll instead of a full window.
#[test]
fn proof_panic_cancels_the_attack() {
    let saw_cancel = AtomicBool::new(false);
    let ran_to_deadline = AtomicBool::new(false);
    let started = Instant::now();
    let caught = catch_unwind(AssertUnwindSafe(|| {
        run_race(
            plan(&never_publish, 60_000),
            pool(),
            polling_attack(&saw_cancel, &ran_to_deadline),
            || -> Result<BetaCrownResult> { panic!("proof panicked") },
        )
    }));
    assert!(caught.is_err(), "the proof's panic propagates");
    assert!(saw_cancel.load(SeqCst));
    assert!(started.elapsed() < Duration::from_secs(10));
}

/// An attack panic is "no candidate": the proof's verdict stands.
#[test]
fn attack_panic_is_no_candidate_and_keeps_the_proof_verdict() {
    let out = run_race(
        plan(&never_publish, 60_000),
        pool(),
        |_token: PgdCancelToken, _deadline: Instant| -> Result<AttackHit> {
            panic!("attack panicked")
        },
        || Ok(result(BabVerificationStatus::Verified)),
    )
    .expect("race");
    assert!(matches!(out.result, BabVerificationStatus::Verified));
}

/// A candidate goes to the publisher exactly once, as the lane's `Violated`
/// result; a DOWNGRADE leaves the proof's (certified) verdict in charge.
#[test]
fn downgraded_candidate_keeps_the_proof_verdict() {
    let calls = AtomicUsize::new(0);
    let seen = Mutex::new(None);
    let publish = |candidate: &BetaCrownResult, report: &ConcurrentSatReport| {
        calls.fetch_add(1, SeqCst);
        *seen.lock().unwrap() = Some((candidate.clone(), report.site));
        adjudicate_with(|| ConcurrentSatPublication::Downgraded)
    };
    let out = run_race(
        plan(&publish, 60_000),
        pool(),
        |_token: PgdCancelToken, _deadline: Instant| Ok(hit()),
        || {
            std::thread::sleep(Duration::from_millis(20));
            Ok(result(BabVerificationStatus::Verified))
        },
    )
    .expect("race");
    assert!(
        matches!(out.result, BabVerificationStatus::Verified),
        "a downgraded candidate must never discard the certified verdict"
    );
    assert_eq!(calls.load(SeqCst), 1);
    let (candidate, site) = seen.lock().unwrap().clone().expect("published");
    assert_eq!(site, RaceSite::GraphUpfront);
    match candidate.result {
        BabVerificationStatus::Violated {
            counterexample,
            output,
        } => {
            assert_eq!(counterexample, vec![0.75]);
            assert_eq!(output, vec![0.75]);
        }
        other => panic!("the publisher must see the lane's Violated result, got {other:?}"),
    }
}

/// With no publisher (or a failed write) the candidate travels back to the
/// main thread as the lane's `Violated` result — today's serial hand-off — and
/// outranks the proof's result there, as the serial order would have.
#[test]
fn unadjudicated_candidate_returns_to_the_main_thread() {
    for declined in [
        ConcurrentSatPublication::NotArmed,
        ConcurrentSatPublication::WriteFailed("disk".to_string()),
    ] {
        let publish = move |_: &BetaCrownResult, _: &ConcurrentSatReport| {
            let answer = declined.clone();
            adjudicate_with(move || answer)
        };
        let out = run_race(
            plan(&publish, 60_000),
            pool(),
            |_token: PgdCancelToken, _deadline: Instant| Ok(hit()),
            || Ok(result(BabVerificationStatus::Timeout)),
        )
        .expect("race");
        assert!(matches!(out.result, BabVerificationStatus::Violated { .. }));
    }
}

/// The attack really runs on the PRIVATE pool (its par_iter work lands on
/// `attack-concurrent-*` workers), and it receives the stretched deadline and
/// a live token.
#[test]
fn attack_runs_in_the_private_pool_with_the_stretched_deadline() {
    let worker_named = AtomicBool::new(false);
    let deadline_seen = Mutex::new(None);
    let p = plan(&never_publish, 4_000);
    let expected = p.site_start + p.window.window;
    let _ = run_race(
        p,
        pool(),
        |token: PgdCancelToken, deadline: Instant| {
            *deadline_seen.lock().unwrap() = Some(deadline);
            assert!(!token.is_cancelled());
            let name = std::thread::current().name().map(str::to_owned);
            worker_named.store(
                name.is_some_and(|n| n.starts_with("attack-concurrent-")),
                SeqCst,
            );
            Ok(None)
        },
        || Ok(result(BabVerificationStatus::Timeout)),
    );
    assert!(
        worker_named.load(SeqCst),
        "the attack body runs on a private-pool worker"
    );
    assert_eq!(*deadline_seen.lock().unwrap(), Some(expected));
}

/// Unused-type guard: the candidate's `Violated` shape matches the serial
/// sites' (`domains_explored = 0`, the full counterexample and output).
#[test]
fn violated_result_matches_the_serial_shape() {
    let x: ArrayD<f32> = arr1(&[1.0_f32, 2.0]).into_dyn();
    let y: ArrayD<f32> = arr1(&[3.0_f32]).into_dyn();
    let r = violated_result(&x, &y, Instant::now());
    assert_eq!(r.domains_explored, 0);
    assert!(r.output_bounds.is_none());
    assert!(proof_decided(&Ok(r)));
    assert!(!proof_decided(&Ok(result(BabVerificationStatus::Timeout))));
}

// ---------------------------------------------------------------------------
//  Adversarial review (review/attack-concurrency): injected delays at every
//  hand-off point of the race — the attack's poll, the publisher's gate, the
//  proof's return — and the invariants the scored path relies on.
// ---------------------------------------------------------------------------

/// INSIDE THE BOUND, THE JOIN STILL WAITS. A publisher delayed at the gate
/// point (250 ms, well inside the bounded join's `GATE_RESERVE`) FINISHES
/// before `run_race` returns a decided proof's verdict — and a downgrade then
/// leaves that certified verdict in charge. (Past the bound the adjudication
/// is abandoned to the main thread's synchronized door:
/// `decided_proof_bounds_a_slow_gate_and_publishes_its_verdict`.)
#[test]
fn decided_proof_joins_an_inflight_publication_before_returning() {
    let publish_done = Arc::new(AtomicBool::new(false));
    let publish = |_: &BetaCrownResult, _: &ConcurrentSatReport| {
        let done = Arc::clone(&publish_done);
        adjudicate_with(move || {
            std::thread::sleep(Duration::from_millis(250));
            done.store(true, SeqCst);
            ConcurrentSatPublication::Downgraded
        })
    };
    let started = Instant::now();
    let out = run_race(
        plan(&publish, 60_000),
        pool(),
        |_token: PgdCancelToken, _deadline: Instant| Ok(hit()),
        || Ok(result(BabVerificationStatus::Verified)),
    )
    .expect("race");
    assert!(
        publish_done.load(SeqCst),
        "run_race returned while the attack thread was still inside the publisher"
    );
    assert!(started.elapsed() >= Duration::from_millis(250));
    assert!(
        matches!(out.result, BabVerificationStatus::Verified),
        "a downgraded in-flight candidate must leave the certified verdict, got {:?}",
        out.result
    );
}

/// An attack in a NON-POLLING stretch (a long batched forward) when the proof
/// decides: the cancel is observed late, the join waits for it, and the
/// decided verdict is returned intact — late, never lost. (The wait is bounded
/// only by the attack's own polling cadence; see the review's gap note.)
#[test]
fn attack_deaf_to_cancel_delays_but_never_replaces_a_decided_verdict() {
    let attack_returned = AtomicBool::new(false);
    let started = Instant::now();
    let out = run_race(
        plan(&never_publish, 60_000),
        pool(),
        |_token: PgdCancelToken, _deadline: Instant| {
            std::thread::sleep(Duration::from_millis(300));
            attack_returned.store(true, SeqCst);
            Ok(None)
        },
        || Ok(result(BabVerificationStatus::Verified)),
    )
    .expect("race");
    assert!(attack_returned.load(SeqCst), "the scope joined the attack");
    assert!(started.elapsed() >= Duration::from_millis(300));
    assert!(matches!(out.result, BabVerificationStatus::Verified));
}

/// A candidate that surfaces AFTER the proof decided (the attack missed the
/// cancel) still goes through the publisher exactly once — #sat-precedence: a
/// confirmed witness outranks a bound-based verdict whichever finished first.
/// Downgraded, the certified verdict stands; UNADJUDICATED (no publisher, a
/// failed write), the candidate travels to the main thread, which re-runs the
/// gate — the serial order's hand-off, never a silently dropped witness.
#[test]
fn candidate_after_the_proof_decided_is_still_adjudicated_once() {
    for (declined, expect_candidate) in [
        (ConcurrentSatPublication::Downgraded, false),
        (ConcurrentSatPublication::NotArmed, true),
        (
            ConcurrentSatPublication::WriteFailed("disk".to_string()),
            true,
        ),
    ] {
        let calls = AtomicUsize::new(0);
        let saw_cancel_before_publish = AtomicBool::new(false);
        let publish = |_: &BetaCrownResult, _: &ConcurrentSatReport| {
            calls.fetch_add(1, SeqCst);
            let answer = declined.clone();
            adjudicate_with(move || answer)
        };
        let out = run_race(
            plan(&publish, 60_000),
            pool(),
            |token: PgdCancelToken, _deadline: Instant| {
                // Deaf until well after the proof's instant decision.
                while !token.is_cancelled() {
                    std::thread::sleep(Duration::from_millis(1));
                }
                std::thread::sleep(Duration::from_millis(20));
                saw_cancel_before_publish.store(true, SeqCst);
                Ok(hit())
            },
            || Ok(result(BabVerificationStatus::Verified)),
        )
        .expect("race");
        assert!(saw_cancel_before_publish.load(SeqCst));
        assert_eq!(calls.load(SeqCst), 1, "{declined:?}: one adjudication");
        assert_eq!(
            matches!(out.result, BabVerificationStatus::Violated { .. }),
            expect_candidate,
            "{declined:?}: got {:?}",
            out.result
        );
        if !expect_candidate {
            assert!(matches!(out.result, BabVerificationStatus::Verified));
        }
    }
}

/// Deterministic xorshift for the stress test (no `rand` dependency).
struct StressRng(u64);

impl StressRng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// RANDOMIZED RACE STRESS: 400 races with injected delays at every hand-off
/// (attack poll cadence and hit time, publisher/gate latency, proof latency)
/// over every proof outcome and every publisher answer. On every race:
///
/// * the publisher is called at most once, and only for an attack hit;
/// * the attack closure AND the publisher have returned before `run_race`
///   does (every delay here is far inside the bounded join, so no
///   adjudication is abandoned);
/// * the token is cancelled for `ProofDecided` iff the proof decided; an
///   undecided proof cancels it only as `WindowShrunk`, and only at its
///   serial-equivalent deadline (never before it, never after `W`);
/// * the lane returns the candidate iff the publisher left it unadjudicated
///   (`NotArmed` / `WriteFailed`), and otherwise EXACTLY the proof's result —
///   a decided proof's verdict is never replaced by anything else.
#[test]
fn randomized_race_stress_keeps_every_invariant() {
    let answers = [
        ConcurrentSatPublication::Downgraded,
        ConcurrentSatPublication::Unconfirmed,
        ConcurrentSatPublication::SuppressedAtDeadline,
        ConcurrentSatPublication::Withheld,
        ConcurrentSatPublication::ClaimTaken,
        ConcurrentSatPublication::NotArmed,
        ConcurrentSatPublication::WriteFailed("injected".to_string()),
    ];
    let mut rng = StressRng(0x9E37_79B9_7F4A_7C15);
    let mut decided_races = 0usize;
    let mut candidate_returns = 0usize;
    for iteration in 0..400 {
        let attack_hits = rng.below(2) == 0;
        let attack_polls = rng.below(4) != 0;
        let attack_delay = Duration::from_micros(rng.below(4_000));
        let proof_delay = Duration::from_micros(rng.below(4_000));
        let publish_delay = Duration::from_micros(rng.below(4_000));
        let proof_kind = rng.below(5);
        let answer = answers[usize::try_from(rng.below(answers.len() as u64)).unwrap()].clone();

        let publish_calls = AtomicUsize::new(0);
        let publish_returned = Arc::new(AtomicBool::new(false));
        let attack_returned = AtomicBool::new(false);
        let token_seen: Mutex<Option<PgdCancelToken>> = Mutex::new(None);
        let publish = |_: &BetaCrownResult, _: &ConcurrentSatReport| {
            publish_calls.fetch_add(1, SeqCst);
            let returned = Arc::clone(&publish_returned);
            let answer = answer.clone();
            adjudicate_with(move || {
                std::thread::sleep(publish_delay);
                returned.store(true, SeqCst);
                answer
            })
        };
        let attack = |token: PgdCancelToken, deadline: Instant| -> Result<AttackHit> {
            *token_seen.lock().unwrap() = Some(token.clone());
            let hit_at = Instant::now() + attack_delay;
            let out = loop {
                if attack_polls && token.is_cancelled() {
                    break None;
                }
                if Instant::now() >= deadline {
                    break None;
                }
                if attack_hits && Instant::now() >= hit_at {
                    break hit();
                }
                std::thread::sleep(Duration::from_micros(200));
            };
            attack_returned.store(true, SeqCst);
            Ok(out)
        };
        let proof = || -> Result<BetaCrownResult> {
            std::thread::sleep(proof_delay);
            match proof_kind {
                0 => Ok(result(BabVerificationStatus::Verified)),
                1 => Ok(result(BabVerificationStatus::Violated {
                    counterexample: vec![0.25],
                    output: vec![0.25],
                })),
                2 => Ok(result(BabVerificationStatus::Timeout)),
                3 => Ok(result(BabVerificationStatus::Unknown {
                    reason: "stress".to_string(),
                })),
                _ => Err(anyhow::anyhow!("stress proof error")),
            }
        };
        let proof_decided_expected = proof_kind <= 1;
        // `run_race` owns its private pool, so each race gets a fresh one.
        let race_plan = plan(&publish, 12);
        let site_start = race_plan.site_start;
        let (out, observed) = run_race_observed(race_plan, pool(), attack, proof);
        let observed = observed.expect("the race ran");

        assert!(
            attack_returned.load(SeqCst),
            "iteration {iteration}: the attack outlived run_race"
        );
        let calls = publish_calls.load(SeqCst);
        assert!(calls <= 1, "iteration {iteration}: {calls} publications");
        if calls == 1 {
            assert!(
                attack_hits,
                "iteration {iteration}: published without a hit"
            );
            assert!(
                publish_returned.load(SeqCst),
                "iteration {iteration}: run_race returned inside the publisher"
            );
        }
        let token = token_seen.lock().unwrap().clone().expect("the attack ran");
        assert_eq!(observed.proof_decided, proof_decided_expected);
        assert_eq!(
            observed.cancel == CancelReason::ProofDecided,
            proof_decided_expected,
            "iteration {iteration}: cancel={:?} for proof kind {proof_kind}",
            observed.cancel
        );
        assert_eq!(
            token.is_cancelled(),
            observed.cancel != CancelReason::None,
            "iteration {iteration}: the token and the recorded reason disagree"
        );
        let stretched_end = site_start + Duration::from_millis(12);
        if proof_decided_expected {
            assert_eq!(observed.attack_deadline, stretched_end);
        } else {
            assert!(
                matches!(
                    observed.cancel,
                    CancelReason::None | CancelReason::WindowShrunk
                ),
                "iteration {iteration}: an undecided proof cancelled as {:?}",
                observed.cancel
            );
            // Never before the serial slice, never after W.
            assert!(observed.attack_deadline >= site_start + Duration::from_millis(6));
            assert!(observed.attack_deadline <= stretched_end);
        }
        assert_ne!(observed.adjudication, Adjudication::Abandoned);
        let candidate_expected = calls == 1 && answer.returns_candidate_to_main();
        match (&out, candidate_expected) {
            (Ok(r), true) => {
                candidate_returns += 1;
                match &r.result {
                    BabVerificationStatus::Violated { counterexample, .. } => assert_eq!(
                        counterexample,
                        &vec![0.75],
                        "iteration {iteration}: the ATTACK's candidate"
                    ),
                    other => panic!("iteration {iteration}: expected the candidate, got {other:?}"),
                }
            }
            (Ok(r), false) => {
                let matches_proof = match proof_kind {
                    0 => matches!(r.result, BabVerificationStatus::Verified),
                    1 => matches!(
                        &r.result,
                        BabVerificationStatus::Violated { counterexample, .. }
                            if counterexample == &vec![0.25]
                    ),
                    2 => matches!(r.result, BabVerificationStatus::Timeout),
                    3 => matches!(r.result, BabVerificationStatus::Unknown { .. }),
                    _ => false,
                };
                assert!(
                    matches_proof,
                    "iteration {iteration}: proof kind {proof_kind} became {:?}",
                    r.result
                );
            }
            (Err(_), expected) => assert!(
                proof_kind == 4 && !expected,
                "iteration {iteration}: an Err only for an erroring proof with no candidate"
            ),
        }
        if proof_decided_expected {
            decided_races += 1;
        }
    }
    assert!(
        decided_races > 50 && candidate_returns > 10,
        "the stress must exercise both branches (decided={decided_races}, \
         candidate_returns={candidate_returns})"
    );
}

// ---------------------------------------------------------------------------
//  Follow-ups (feat/attack-concurrency-followups): (a) a decided proof's join
//  is bounded even when the attack thread is inside a SLOW trusted gate;
//  (c) an EARLY undecided proof shrinks the attack back to its serial-
//  equivalent window instead of holding later lanes for the stretched one.
// ---------------------------------------------------------------------------

/// (a) THE JOIN TIMER. The attack's candidate reaches a gate that takes 3 s
/// (a cold ORT session on a large graph model); the proof decides at once;
/// the dispatch's authoritative deadline is 1.5 s away. The race must return
/// the certified verdict while the dispatch's deadline gate still admits it —
/// a gate that outlives the reserve must NOT turn a `Verified` decided in time
/// into `Timeout`.
#[test]
fn decided_proof_bounds_a_slow_gate_and_publishes_its_verdict() {
    let gate_done = Arc::new(AtomicBool::new(false));
    let start = Instant::now();
    let dispatch_deadline = start + Duration::from_millis(1_500);
    let publish = |_: &BetaCrownResult, _: &ConcurrentSatReport| {
        let gate_flag = Arc::clone(&gate_done);
        adjudicate_with(move || {
            std::thread::sleep(Duration::from_secs(3));
            gate_flag.store(true, SeqCst);
            ConcurrentSatPublication::Downgraded
        })
    };
    let race_plan = RacePlan {
        overall_deadline: Some(dispatch_deadline),
        ..plan(&publish, 60_000)
    };
    let (out, observed) = run_race_observed(
        race_plan,
        pool(),
        |_token: PgdCancelToken, _deadline: Instant| Ok(hit()),
        || Ok(result(BabVerificationStatus::Verified)),
    );
    let out = out.expect("race");
    let returned = start.elapsed();
    let observed = observed.expect("the race ran");
    assert_eq!(observed.adjudication, Adjudication::Abandoned);
    assert_eq!(observed.outcome, AttackOutcome::AdjudicationAbandoned);
    assert_eq!(observed.cancel, CancelReason::ProofDecided);
    assert!(
        observed
            .join_deadline
            .is_some_and(|join| Some(join) <= dispatch_deadline.checked_sub(JOIN_RESERVE)),
        "the join is capped by the dispatch deadline minus the reserve: {:?}",
        observed.join_deadline
    );
    let gated =
        super::super::dispatch::gate_result_at_deadline(out, start, Some(dispatch_deadline));
    assert!(
        matches!(gated.result, BabVerificationStatus::Verified),
        "a slow gate turned the certified verdict into {:?} (race returned after {returned:?})",
        gated.result
    );
    assert!(
        returned < Duration::from_millis(1_500),
        "the join waited out the slow gate ({returned:?})"
    );
    assert!(
        !gate_done.load(SeqCst),
        "the race returned only after the 3 s gate finished"
    );
    // The abandoned adjudication is not killed: it finishes detached (inside
    // the attack door, which the main thread's synchronized door waits on).
    let waited = Instant::now();
    while !gate_done.load(SeqCst) && waited.elapsed() < Duration::from_secs(10) {
        std::thread::sleep(Duration::from_millis(10));
    }
    assert!(
        gate_done.load(SeqCst),
        "the detached gate ran to completion"
    );
}

/// A PROOF PANIC with a candidate inside the trusted gate still WAITS for
/// that adjudication before the panic propagates (review(attack-conc-
/// followups)). Nothing on the scored path catches a proof panic: it ends the
/// process with the `unknown` placeholder on disk, so the in-flight candidate
/// is the only verdict the instance can still reach — an upheld one publishes
/// (and `_exit`s) from inside this wait, exactly as the serial order and the
/// reviewed base did. Abandoning it at the unwind (the follow-up's first
/// version, an `Instant::now()` join deadline) let the panic kill the
/// detached gate mid-adjudication: RED on 0d091a88c (the panic propagated
/// after ~50 ms with the gate unfinished).
#[test]
fn proof_panic_still_waits_for_an_inflight_gate() {
    let gate_done = Arc::new(AtomicBool::new(false));
    let started = Instant::now();
    let publish = |_: &BetaCrownResult, _: &ConcurrentSatReport| {
        let done = Arc::clone(&gate_done);
        adjudicate_with(move || {
            std::thread::sleep(Duration::from_millis(400));
            done.store(true, SeqCst);
            ConcurrentSatPublication::Downgraded
        })
    };
    let caught = catch_unwind(AssertUnwindSafe(|| {
        run_race(
            plan(&publish, 60_000),
            pool(),
            |_token: PgdCancelToken, _deadline: Instant| Ok(hit()),
            || -> Result<BetaCrownResult> {
                std::thread::sleep(Duration::from_millis(50));
                panic!("proof panicked")
            },
        )
    }));
    let propagated_after = started.elapsed();
    assert!(caught.is_err(), "the proof's panic propagates");
    assert!(
        gate_done.load(SeqCst),
        "the panic propagated after {propagated_after:?} with the candidate's adjudication \
         still in flight: the only verdict left was abandoned to process death"
    );
    assert!(
        propagated_after >= Duration::from_millis(390),
        "the unwinding join did not wait for the gate ({propagated_after:?})"
    );
}

/// (a) The bounded join's arithmetic: `GATE_RESERVE` after the decision,
/// capped at `overall_deadline - JOIN_RESERVE`, never before the decision,
/// and uncapped without an overall deadline.
#[test]
fn decided_join_deadline_is_the_gate_reserve_capped_by_the_overall_deadline() {
    let t = Instant::now();
    assert_eq!(decided_join_deadline(t, None), t + GATE_RESERVE);
    assert_eq!(
        decided_join_deadline(t, Some(t + Duration::from_secs(100))),
        t + GATE_RESERVE,
        "far from the deadline: the designed gate budget"
    );
    assert_eq!(
        decided_join_deadline(t, Some(t + Duration::from_secs(3))),
        (t + Duration::from_secs(3))
            .checked_sub(JOIN_RESERVE)
            .unwrap(),
        "near the deadline: the cap"
    );
    assert_eq!(
        decided_join_deadline(t, Some(t + Duration::from_millis(500))),
        t,
        "past the cap: no wait at all"
    );
}

/// (c) The undecided shrink's arithmetic, for every width pair and proof
/// wall up to `W`: the attack keeps at least the serial slice's CPU-seconds
/// under the CFS model the stretch is sized by
/// (`p * k_a / (N + k_a) + (D - p) >= slice`), the lanes after the race never
/// start later than the serial order would start them (`D <= slice + p`),
/// and `D` stays inside `[slice, W]`.
#[test]
fn undecided_shrink_conserves_attack_cpu_and_never_delays_later_lanes() {
    let start = Instant::now();
    for slice_ms in [10_u64, 1_000, 42_000, 57_000] {
        for host in 1..=24usize {
            for width in [1usize, 2, 4, 8, 20, 24] {
                let slice = Duration::from_millis(slice_ms);
                let w = stretched_window(slice, host, width);
                let window = RaceWindow {
                    slice,
                    window: w,
                    host_threads: host,
                    width_pool: width,
                };
                let stretched_end = start + w;
                for step in 0..=20u32 {
                    let p = w * step / 20;
                    let deadline =
                        undecided_attack_deadline(start, &window, start + p, stretched_end);
                    let d = deadline - start;
                    assert!(
                        d >= slice && d <= w,
                        "slice={slice_ms} N={host} k={width} p={p:?}: D={d:?}"
                    );
                    assert!(
                        d <= slice + p,
                        "later lanes start after the serial order: D={d:?}"
                    );
                    if d < w {
                        // CPU-seconds in serial-attack units, in nanoseconds.
                        let shared = p.as_nanos() * width as u128 / (host + width) as u128;
                        let alone = d.saturating_sub(p).as_nanos();
                        assert!(
                            shared + alone + 2 >= slice.as_nanos(),
                            "slice={slice_ms} N={host} k={width} p={p:?}: the attack lost CPU"
                        );
                    }
                }
            }
        }
    }
}

/// (c) EARLY UNDECIDED PROOF. W = 2 s (slice 1 s, N = k_a = 2); the proof
/// gives up at once. The attack keeps its serial-equivalent window
/// (`slice + p * N / (N + k_a)` ~ 1 s here) and is then cut, so the lanes
/// after the race start ~1 s earlier than the stretched window would allow
/// — and never later than the serial order (`slice + p`).
#[test]
fn early_undecided_proof_shrinks_the_attack_to_its_serial_equivalent_window() {
    let saw_cancel = AtomicBool::new(false);
    let ran_to_deadline = AtomicBool::new(false);
    let started = Instant::now();
    let (out, observed) = run_race_observed(
        plan(&never_publish, 2_000),
        pool(),
        polling_attack(&saw_cancel, &ran_to_deadline),
        || {
            Ok(result(BabVerificationStatus::Unknown {
                reason: "gave up early".to_string(),
            }))
        },
    );
    let out = out.expect("race");
    let elapsed = started.elapsed();
    let observed = observed.expect("the race ran");
    assert_eq!(observed.cancel, CancelReason::WindowShrunk);
    assert_eq!(observed.outcome, AttackOutcome::WindowShrunk);
    assert!(matches!(out.result, BabVerificationStatus::Unknown { .. }));
    assert!(
        elapsed >= Duration::from_millis(990),
        "the attack must keep its serial slice ({elapsed:?})"
    );
    assert!(
        elapsed < Duration::from_millis(1_600),
        "later lanes waited out the stretched 2 s window ({elapsed:?})"
    );
    assert!(
        saw_cancel.load(SeqCst) && !ran_to_deadline.load(SeqCst),
        "the attack is cut at the shrunk deadline, not at W"
    );
}
