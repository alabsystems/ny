// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #attack-concurrent at the real sites: the cancellation token stops the
//! REAL attack loops the race runs (graph exact-gradient PGD, the sequential
//! sampling + classified parallel attacker), and the two race sites are gated
//! by the key — dark runs never touch the publish hook, armed runs hand the
//! attack's candidate to it.

use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use ndarray::{arr1, arr2};
use ny_onnx::vnnlib::{parse_vnnlib, VnnLibSpec};
use ny_propagate::{
    layers::LinearLayer, BabVerificationStatus, BetaCrownConfig, BetaCrownResult,
    BetaCrownVerifier, GraphNetwork, Layer, Network, PgdCancelToken, PgdConfig,
};
use ny_tensor::BoundedTensor;

use super::super::attack_arming::AttackEngineSource;
use super::super::attack_concurrent::{
    ConcurrentAttackCtx, ConcurrentAttackWrapperFacts, ConcurrentSatPublication,
    ConcurrentSatReport, PreparedSat, RaceSite,
};
use super::phase_budget::PhaseBudgetLedger;

/// `Y_0 = x` on one input.
fn identity_network() -> Network {
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0_f32]]), Some(arr1(&[0.0_f32]))).expect("1x1 linear"),
    ));
    network
}

/// `Y_0 = x`, `Y_1 = -x` on one input.
fn mirror_network() -> Network {
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0_f32], [-1.0_f32]]), Some(arr1(&[0.0_f32, 0.0])))
            .expect("2x1 linear"),
    ));
    network
}

fn input(lower: f32, upper: f32) -> BoundedTensor {
    BoundedTensor::new(arr1(&[lower]).into_dyn(), arr1(&[upper]).into_dyn()).expect("box")
}

fn spec(lower: f32, upper: f32, assertion: &str, outputs: usize) -> VnnLibSpec {
    let mut text = String::from("(declare-const X_0 Real)\n");
    for j in 0..outputs {
        text.push_str(&format!("(declare-const Y_{j} Real)\n"));
    }
    text.push_str(&format!(
        "(assert (>= X_0 {lower}))\n(assert (<= X_0 {upper}))\n(assert {assertion})\n"
    ));
    parse_vnnlib(&text).expect("test property parses")
}

/// A long attack whose only way to stop early is its token.
fn long_pgd(deadline: Instant, cancel: Option<PgdCancelToken>) -> PgdConfig {
    PgdConfig {
        num_restarts: 1_000_000,
        num_steps: 100_000,
        deadline: Some(deadline),
        cancel,
        ..PgdConfig::default()
    }
}

/// The graph exact-gradient PGD (the graph race site's attack) stops at its
/// next restart/step poll once its token is cancelled: pre-cancelled it does
/// no work; cancelled mid-flight it returns within one step; without a token
/// the same loop runs to its deadline (the positive control).
#[test]
fn cancel_token_stops_graph_pgd_within_one_step() {
    let graph = GraphNetwork::from_sequential(&identity_network()).expect("graph");
    let input = input(0.0, 1.0);
    let unreachable = spec(0.0, 1.0, "(>= Y_0 5.0)", 1);
    let run = |pgd: PgdConfig| {
        super::graph_pgd::try_graph_pgd_upfront_with_config(
            &graph,
            &input,
            &unreachable,
            pgd,
            None,
            true,
        )
        .expect("attack runs")
    };

    // Positive control: no token, a 300 ms deadline, the loop runs it out.
    let started = Instant::now();
    assert!(run(long_pgd(Instant::now() + Duration::from_millis(300), None)).is_none());
    assert!(
        started.elapsed() >= Duration::from_millis(290),
        "control: the loop is long without a token ({:?})",
        started.elapsed()
    );

    // Pre-cancelled: immediate.
    let token = PgdCancelToken::new();
    token.cancel();
    let started = Instant::now();
    assert!(run(long_pgd(
        Instant::now() + Duration::from_secs(60),
        Some(token)
    ))
    .is_none());
    assert!(
        started.elapsed() < Duration::from_secs(1),
        "{:?}",
        started.elapsed()
    );

    // Cancelled mid-flight from another thread.
    let token = PgdCancelToken::new();
    let started = Instant::now();
    std::thread::scope(|scope| {
        let remote = token.clone();
        scope.spawn(move || {
            std::thread::sleep(Duration::from_millis(100));
            remote.cancel();
        });
        assert!(run(long_pgd(
            Instant::now() + Duration::from_secs(60),
            Some(token)
        ))
        .is_none());
    });
    assert!(
        started.elapsed() < Duration::from_secs(5),
        "a cancelled attack must stop within one step, took {:?}",
        started.elapsed()
    );
}

/// The sequential upfront attack (the sequential race site's attack: the
/// sampling+SPSA phase, then the classified attacker's PARALLEL restarts)
/// stops on its token, pre-cancelled and mid-flight.
#[test]
fn cancel_token_stops_parallel_restarts() {
    let network = mirror_network();
    // x in [-1, -0.5]: Y_0 = x < Y_1 = -x everywhere, so `Y_0 >= Y_1` is
    // unreachable and only the deadline or the token can end the attack.
    let input = input(-1.0, -0.5);
    let unreachable = spec(-1.0, -0.5, "(>= Y_0 Y_1)", 2);
    let run = |pgd: PgdConfig| {
        super::pgd::try_conjunctive_pgd_attack_upfront_with_config(
            &network,
            &input,
            &unreachable,
            PgdConfig {
                parallel: true,
                ..pgd
            },
            None,
            true,
        )
        .expect("attack runs")
    };

    let token = PgdCancelToken::new();
    token.cancel();
    let started = Instant::now();
    assert!(run(long_pgd(
        Instant::now() + Duration::from_secs(60),
        Some(token)
    ))
    .is_none());
    assert!(
        started.elapsed() < Duration::from_secs(2),
        "{:?}",
        started.elapsed()
    );

    let token = PgdCancelToken::new();
    let started = Instant::now();
    std::thread::scope(|scope| {
        let remote = token.clone();
        scope.spawn(move || {
            std::thread::sleep(Duration::from_millis(150));
            remote.cancel();
        });
        assert!(run(long_pgd(
            Instant::now() + Duration::from_secs(60),
            Some(token)
        ))
        .is_none());
    });
    assert!(
        started.elapsed() < Duration::from_secs(10),
        "cancelled parallel restarts must stop promptly, took {:?}",
        started.elapsed()
    );
}

fn armed_ctx(publish: super::super::attack_concurrent::PublishFn<'_>) -> ConcurrentAttackCtx<'_> {
    ConcurrentAttackCtx {
        wrapper: Some(ConcurrentAttackWrapperFacts {
            margin_row_lane_live: false,
        }),
        publisher_armed: true,
        publish,
    }
}

fn graph_site(
    config: &BetaCrownConfig,
    ctx: &ConcurrentAttackCtx<'_>,
    budget: Duration,
) -> BetaCrownResult {
    let graph = GraphNetwork::from_sequential(&identity_network()).expect("graph");
    let input = input(0.0, 1.0);
    // `Y_0 >= 0.5` holds on half the box: the upfront attack finds it at once.
    let reachable = spec(0.0, 1.0, "(>= Y_0 0.5)", 1);
    let verifier = BetaCrownVerifier::new(config.clone());
    let ledger = PhaseBudgetLedger::from_deadline(
        Some(Instant::now() + budget),
        config.phase_budget.clone(),
    );
    super::graph::verify_graph_relational(
        &graph,
        &input,
        &reachable,
        config,
        &verifier,
        false,
        false,
        true,
        budget.as_secs(),
        None,
        AttackEngineSource::disarmed(),
        true,
        &ledger,
        None,
        Some(ctx),
    )
    .expect("graph lane runs")
}

fn seq_site(
    config: &BetaCrownConfig,
    ctx: &ConcurrentAttackCtx<'_>,
    budget: Duration,
) -> BetaCrownResult {
    let network = mirror_network();
    let input = input(0.0, 1.0);
    // `Y_0 >= Y_1` holds on the whole box.
    let reachable = spec(0.0, 1.0, "(>= Y_0 Y_1)", 2);
    let verifier = BetaCrownVerifier::new(config.clone());
    let ledger = PhaseBudgetLedger::from_deadline(
        Some(Instant::now() + budget),
        config.phase_budget.clone(),
    );
    super::sequential::verify_sequential_relational(
        &network,
        &input,
        &reachable,
        config,
        &verifier,
        true,
        config.pgd_restarts,
        config.pgd_steps,
        budget.as_secs(),
        None,
        true,
        &ledger,
        Some(ctx),
    )
    .expect("sequential lane runs")
}

/// DEFAULT BIT-IDENTITY AT THE SITES: with the key off, both sites run the
/// serial code — the upfront attack's counterexample is returned directly as
/// the lane's `Violated` — and the publish hook is never touched even though a
/// fully armed context is handed in.
///
/// The hook COUNTS rather than panics (review/attack-concurrency): a panic
/// raised inside the publisher unwinds the ATTACK thread, which the race's
/// join swallows as `outcome=panic` and then returns the proof's result — so
/// a panicking double could not fail this test on a site that wrongly raced.
#[test]
fn key_off_runs_no_race() {
    let config = BetaCrownConfig::default();
    assert!(!config.attack_concurrent);
    let calls = AtomicUsize::new(0);
    let counting_publish = |_: &BetaCrownResult, _: &ConcurrentSatReport| {
        calls.fetch_add(1, SeqCst);
        PreparedSat::Declined(ConcurrentSatPublication::Downgraded)
    };
    let ctx = armed_ctx(&counting_publish);
    for site in [RaceSite::GraphUpfront, RaceSite::SeqUpfront] {
        let result = match site {
            RaceSite::GraphUpfront => graph_site(&config, &ctx, Duration::from_secs(12)),
            RaceSite::SeqUpfront => seq_site(&config, &ctx, Duration::from_secs(12)),
            RaceSite::RelationalShared => {
                panic!("this ordinary-site fixture has no dual-source adapter")
            }
        };
        assert_eq!(
            calls.load(SeqCst),
            0,
            "{site:?}: a dark site must never reach the concurrent publisher"
        );
        assert!(
            matches!(result.result, BabVerificationStatus::Violated { .. }),
            "{site:?}: the serial upfront attack's counterexample, got {:?}",
            result.result
        );
        assert_eq!(result.domains_explored, 0, "{site:?}: no proof work ran");
    }
}

/// ARMED: each site races, and the attack's counterexample reaches the
/// scored publisher (here a recording double that downgrades it) as the
/// lane's `Violated` result — the candidate a dark run would have returned.
#[test]
fn armed_sites_hand_the_attack_candidate_to_the_publisher() {
    let config = BetaCrownConfig {
        attack_concurrent: true,
        ..BetaCrownConfig::default()
    };
    for site in [RaceSite::GraphUpfront, RaceSite::SeqUpfront] {
        let calls = AtomicUsize::new(0);
        let seen = Mutex::new(Vec::new());
        let publish = |candidate: &BetaCrownResult, report: &ConcurrentSatReport| {
            calls.fetch_add(1, SeqCst);
            if let BabVerificationStatus::Violated { counterexample, .. } = &candidate.result {
                seen.lock()
                    .unwrap()
                    .push((report.site, counterexample.clone()));
            }
            // The detached adjudication path, as on the scored route.
            PreparedSat::Adjudicate(Box::new(|| ConcurrentSatPublication::Downgraded))
        };
        let ctx = armed_ctx(&publish);
        let _ = match site {
            RaceSite::GraphUpfront => graph_site(&config, &ctx, Duration::from_secs(12)),
            RaceSite::SeqUpfront => seq_site(&config, &ctx, Duration::from_secs(12)),
            RaceSite::RelationalShared => {
                panic!("this ordinary-site fixture has no dual-source adapter")
            }
        };
        assert_eq!(
            calls.load(SeqCst),
            1,
            "{site:?}: one candidate, one publication"
        );
        let seen = seen.into_inner().unwrap();
        assert_eq!(
            seen.len(),
            1,
            "{site:?}: the candidate is the lane's Violated"
        );
        assert_eq!(seen[0].0, site);
        let x = seen[0].1[0];
        match site {
            RaceSite::GraphUpfront => assert!((0.5..=1.0).contains(&x), "x={x}"),
            RaceSite::SeqUpfront => assert!((0.0..=1.0).contains(&x), "x={x}"),
            RaceSite::RelationalShared => {
                panic!("this ordinary-site fixture has no dual-source adapter")
            }
        }
    }
}
