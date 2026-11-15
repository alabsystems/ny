// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #kfsb-production: the single-shot kFSB scorer's fan-out at production
//! grade (`gpu_bab/kfsb_fanout.rs`).
//!
//! * SHARED BUDGET (arithmetic): over a sweep of live budgets, candidate
//!   counts, thread counts and R4 lease decisions, no simulation's share
//!   exceeds `total / width`, the shares in flight sum to at most the ONE
//!   budget, the total never exceeds the live budget nor a lease's cover, a
//!   denied lease runs ONE simulation at a time, and more than one in flight
//!   never get less than the minimum share.
//! * SHARED BUDGET (enforced): every worker of the fan-out's pool — including
//!   one that runs a NESTED Rayon job a simulation spawned — sees at most its
//!   share from `cpu_crown_dense_budget_bytes()`, never more than `width`
//!   simulations are live at once, and nothing leaks to the caller; the real
//!   simulations see their share and return bit-identical bounds at width 1
//!   and width k.
//! * THE LEASE: under an armed R4 lease authority the lane asks ONCE (granted
//!   or narrowed: the fan-out's total is the lease's cover; denied: serial),
//!   returns it when the outermost lane ends, and an unarmed authority is
//!   never asked.
//! * GPU ROUTE: with a sound GPU CROWN capability present (the hermetic
//!   adapter), the simulations take the GPU per-domain route (counted, and
//!   the adapter's own drop check fails a test that never crossed it); every
//!   device bound encloses the enumerated child range; the armed pick is
//!   still an unstable, unsplit neuron of the domain; it is the CPU pick
//!   wherever the two routes return bit-equal bounds and on at least 90% of
//!   the fixture decisions (the device fold is a different sound relaxation,
//!   so near-ties can rank differently).
//! * OBSERVABLE CAP: the first `KFSB_SINGLE_SHOT_DECISION_LINES` decisions
//!   print, the rest are counted, and the lane summary carries the totals.
//! * DARK: the key off builds no lane scope, makes no decision, asks no lease
//!   and installs no share.

use std::sync::atomic::{AtomicUsize, Ordering};

use ny_core::memory_lease::LeaseAuthority;
use ny_core::router::facts::{DeviceQual, HostClass, HostEnvelope};
use ny_core::router::lease::{
    lease_plan, LeaseDecision, LeaseHolder, LeaseNeed, DENSE_PAIR_PEAK_FACTOR,
    KFSB_FANOUT_LEASE_NEED, KFSB_FANOUT_MIN_SHARE_BYTES,
};
use rayon::prelude::*;

use super::kfsb_single_shot::{conv_residual, dags, kfsb_config, split, Dag};
use crate::beta_crown::engine::branching::kfsb_shared::GraphKfsbCandidate;
use crate::beta_crown::engine::graph::gpu_bab::kfsb_fanout::{
    decision_line, plan_kfsb_fanout, run_shared_budget_fanout, summary_line, FanoutBudgetSource,
    KfsbDecisionRecord, KfsbSingleShotCounts, KfsbSingleShotLaneScope, KfsbSingleShotTelemetry,
    KFSB_MIN_SHARE_BYTES, KFSB_SINGLE_SHOT_DECISION_LINES,
};
use crate::beta_crown::engine::graph::gpu_bab::kfsb_single_shot_calls_for_tests;
use crate::beta_crown::engine::graph::propagation::batched::interm_refine::HermeticSoundGpuCrownEngine;
use crate::beta_crown::engine::BetaCrownVerifier;
use crate::beta_crown::{BabVerificationStatus, GraphBabDomain};
use crate::network::crown_memory::{
    apply_dense_budget_thread_cap, cpu_crown_dense_budget_bytes, dense_budget_thread_cap,
    DenseBudgetThreadCap,
};

const MIB: usize = 1 << 20;
const GIB: u64 = 1 << 30;

fn candidates_of(unstable: &[(String, usize)]) -> Vec<GraphKfsbCandidate> {
    unstable
        .iter()
        .map(|(node, idx)| GraphKfsbCandidate {
            node_name: node.clone(),
            neuron_idx: *idx,
            main_score: 0.0,
            backup_score: 0.0,
        })
        .collect()
}

fn root(dag: &Dag) -> (crate::GraphNetwork, GraphBabDomain) {
    let graph = dag.graph();
    let input = dag.input_box();
    let node_bounds = graph.collect_node_bounds(&input).expect("ibp bounds");
    let domain =
        GraphBabDomain::root(node_bounds, -100.0, 100.0, &input, false).expect("root domain");
    (graph, domain)
}

type Sims = Vec<(Option<(f32, f32)>, Option<(f32, f32)>)>;

fn bits(s: &Sims) -> Vec<Option<(u32, u32)>> {
    s.iter()
        .flat_map(|(a, b)| [*a, *b])
        .map(|side| side.map(|(l, u)| (l.to_bits(), u.to_bits())))
        .collect()
}

// ---------------------------------------------------------------------------
// 1. Shared-budget arithmetic.
// ---------------------------------------------------------------------------

#[test]
fn kfsb_production_fanout_plan_shares_one_budget() {
    let live_values: [usize; 9] = [
        0,
        1,
        KFSB_MIN_SHARE_BYTES - 1,
        KFSB_MIN_SHARE_BYTES,
        3 * KFSB_MIN_SHARE_BYTES + 7,
        1 << 30,
        12 << 30,
        (2 << 30) + 12345,
        usize::MAX / 2,
    ];
    let leases: Vec<Option<LeaseDecision>> = vec![
        None,
        Some(LeaseDecision::Granted { bytes: 16 * GIB }),
        Some(LeaseDecision::Granted { bytes: 1 }),
        Some(LeaseDecision::Narrowed {
            bytes: 3 * GIB + 5,
            want: 16 * GIB,
        }),
        Some(LeaseDecision::Narrowed {
            bytes: KFSB_FANOUT_LEASE_NEED.min_bytes,
            want: 16 * GIB,
        }),
        Some(LeaseDecision::Denied {
            available: 0,
            want: 16 * GIB,
        }),
    ];
    let mut checked = 0usize;
    let mut violations = Vec::new();
    let mut widths_seen = std::collections::BTreeSet::new();
    for &live in &live_values {
        for lease in &leases {
            for candidates in 0..=41usize {
                for threads in [0usize, 1, 2, 7, 20, 64] {
                    let plan =
                        plan_kfsb_fanout(live, *lease, candidates, threads, KFSB_MIN_SHARE_BYTES);
                    checked += 1;
                    widths_seen.insert(plan.width);
                    let mut bad = |why: &str| {
                        violations.push(format!(
                            "live={live} lease={lease:?} k={candidates} t={threads}: {why} \
                             -> {plan:?}"
                        ));
                    };
                    if plan.width == 0 {
                        bad("width 0");
                    }
                    // No simulation exceeds its share; the shares in flight
                    // sum to at most the ONE budget.
                    if plan.share_bytes != plan.total_bytes / plan.width.max(1) {
                        bad("share is not total / width");
                    }
                    if plan.width.saturating_mul(plan.share_bytes) > plan.total_bytes {
                        bad("sum of shares exceeds the total");
                    }
                    if plan.total_bytes > live {
                        bad("total exceeds the live budget (one dark pass)");
                    }
                    if plan.width > candidates.max(1) || plan.width > threads.max(1) {
                        bad("more in flight than candidates or threads");
                    }
                    if plan.width > 1 && plan.share_bytes < KFSB_MIN_SHARE_BYTES {
                        bad("several in flight below the minimum share");
                    }
                    if plan.waves != candidates.div_ceil(plan.width) {
                        bad("waves");
                    }
                    match lease {
                        None => {
                            if plan.source != FanoutBudgetSource::Live || plan.total_bytes != live {
                                bad("unleased total is the live budget");
                            }
                        }
                        Some(LeaseDecision::Denied { .. }) => {
                            if plan.width != 1 || plan.share_bytes != live {
                                bad("denied lease runs one dark-equivalent pass at a time");
                            }
                        }
                        Some(decision) => {
                            let cover = decision.bytes() / DENSE_PAIR_PEAK_FACTOR;
                            if (plan.total_bytes as u64) > cover {
                                bad("total exceeds the lease's cover (leased / peak factor)");
                            }
                            if plan.total_bytes != live.min(cover as usize) {
                                bad("leased total is min(live, cover)");
                            }
                        }
                    }
                }
            }
        }
    }
    println!(
        "[kfsb-production] PLAN checked={checked} widths_seen={} violations={}",
        widths_seen.len(),
        violations.len()
    );
    assert!(checked > 10_000, "positive control: {checked}");
    assert!(
        widths_seen.len() > 10,
        "positive control: the sweep exercises many widths"
    );
    assert!(
        violations.is_empty(),
        "shared-budget arithmetic: {:#?}",
        &violations[..violations.len().min(20)]
    );
    // Concrete anchors: today's GB10 (12 GiB live, 20 threads, 20 candidates)
    // runs all 20 at 614 MiB each; a 2400 MiB live budget runs 9 at 266 MiB;
    // a narrowed 3 GiB lease covers 768 MiB = 3 simulations.
    let gb10 = plan_kfsb_fanout(12 << 30, None, 20, 20, KFSB_MIN_SHARE_BYTES);
    assert_eq!((gb10.width, gb10.share_bytes), (20, (12 << 30) / 20));
    let small = plan_kfsb_fanout(2_400 * MIB, None, 20, 20, KFSB_MIN_SHARE_BYTES);
    assert_eq!(small.width, 9);
    let narrowed = plan_kfsb_fanout(
        12 << 30,
        Some(LeaseDecision::Narrowed {
            bytes: 3 * GIB,
            want: 16 * GIB,
        }),
        20,
        20,
        KFSB_MIN_SHARE_BYTES,
    );
    assert_eq!((narrowed.total_bytes, narrowed.width), (768 * MIB, 3));
    assert_eq!(KFSB_MIN_SHARE_BYTES as u64, KFSB_FANOUT_MIN_SHARE_BYTES);
    println!("[kfsb-production] TEST_COMPLETE name=kfsb_production_fanout_plan_shares_one_budget");
}

// ---------------------------------------------------------------------------
// 2. The share is enforced on every worker, nested jobs included.
// ---------------------------------------------------------------------------

#[test]
fn kfsb_production_every_worker_sees_its_share_even_in_nested_jobs() {
    let live = cpu_crown_dense_budget_bytes();
    assert!(
        live > 4 * KFSB_MIN_SHARE_BYTES,
        "positive control: the host's live dense budget ({live}) must exceed the shares tested"
    );
    assert_eq!(
        dense_budget_thread_cap(),
        None,
        "no share outside a fan-out"
    );
    // A total below the live budget, so a worker that missed its share would
    // read MORE than the share and the check fails (mutation target: the
    // pool's thread wrapper not installing the share).
    let total = 4 * KFSB_MIN_SHARE_BYTES + 3;
    let items: Vec<usize> = (0..37).collect();
    for threads in [1usize, 3, 8] {
        let plan = plan_kfsb_fanout(total, None, items.len(), threads, KFSB_MIN_SHARE_BYTES);
        let live_now = AtomicUsize::new(0);
        let live_max = AtomicUsize::new(0);
        let nested_reads = AtomicUsize::new(0);
        let out = run_shared_budget_fanout(&items, &plan, threads, |item| {
            let now = live_now.fetch_add(1, Ordering::SeqCst) + 1;
            live_max.fetch_max(now, Ordering::SeqCst);
            let top = cpu_crown_dense_budget_bytes();
            // A nested Rayon fan-out inside the simulation: whichever worker
            // of the pool runs these jobs must see the share too.
            let nested_max = (0..64usize)
                .into_par_iter()
                .map(|_| {
                    nested_reads.fetch_add(1, Ordering::Relaxed);
                    std::thread::yield_now();
                    cpu_crown_dense_budget_bytes()
                })
                .max()
                .unwrap_or(0);
            live_now.fetch_sub(1, Ordering::SeqCst);
            (*item, top, nested_max)
        });
        assert_eq!(out.len(), items.len());
        for (i, (item, top, nested)) in out.iter().enumerate() {
            assert_eq!(*item, i, "results come back in item order");
            assert!(
                *top <= plan.share_bytes && *nested <= plan.share_bytes,
                "threads={threads}: a simulation read {top} / nested {nested} > share {}",
                plan.share_bytes
            );
        }
        assert!(
            live_max.load(Ordering::SeqCst) <= plan.width,
            "threads={threads}: {} simulations live at once, width {}",
            live_max.load(Ordering::SeqCst),
            plan.width
        );
        assert_eq!(nested_reads.load(Ordering::Relaxed), 64 * items.len());
        println!(
            "[kfsb-production] SHARE threads={threads} width={} share={} live_max={}",
            plan.width,
            plan.share_bytes,
            live_max.load(Ordering::SeqCst)
        );
    }
    // Nothing leaks to the caller.
    assert_eq!(dense_budget_thread_cap(), None);
    // The cap narrows and never widens; nested caps keep the tighter one.
    assert_eq!(apply_dense_budget_thread_cap(100, None), 100);
    assert_eq!(apply_dense_budget_thread_cap(100, Some(40)), 40);
    assert_eq!(apply_dense_budget_thread_cap(100, Some(400)), 100);
    {
        let _outer = DenseBudgetThreadCap::install(1000);
        let _inner = DenseBudgetThreadCap::install(5000);
        assert_eq!(dense_budget_thread_cap(), Some(1000));
        assert!(cpu_crown_dense_budget_bytes() <= 1000);
    }
    assert_eq!(dense_budget_thread_cap(), None);
    println!(
        "[kfsb-production] TEST_COMPLETE name=kfsb_production_every_worker_sees_its_share_even_in_nested_jobs"
    );
}

#[ntest::timeout(600000)]
#[test]
fn kfsb_production_simulations_run_under_the_shared_budget() {
    let _env = crate::tests::lock_env_shared();
    let mut sims_checked = 0usize;
    let mut compared = 0usize;
    for dag in dags() {
        let (graph, domain) = root(&dag);
        let verifier = BetaCrownVerifier::new(kfsb_config(true));
        let unstable = verifier.find_unstable_graph_neurons(&graph, &domain, &dag.relu_nodes());
        let candidates = candidates_of(&unstable);
        if candidates.is_empty() {
            continue;
        }
        for row in [[1.0f32], [-1.0]] {
            // The production path: the live budget read once, shared.
            let sims = verifier
                .kfsb_single_shot_child_bounds_for_tests(&graph, &domain, &row, &candidates)
                .expect("simulations");
            // The production plan, explicitly (no lease is armed in a unit
            // test, so the production path plans exactly this), under a share
            // SMALLER than the live budget so a worker that missed it reads
            // more than the share and the check fails.
            let k = candidates.len();
            let threads = rayon::current_num_threads();
            let live = plan_kfsb_fanout(
                cpu_crown_dense_budget_bytes(),
                None,
                k,
                threads,
                KFSB_MIN_SHARE_BYTES,
            );
            let tight = plan_kfsb_fanout(
                live.width.max(1) * KFSB_MIN_SHARE_BYTES,
                None,
                k,
                threads,
                KFSB_MIN_SHARE_BYTES,
            );
            for plan in [live, tight] {
                let (bounds, rec) = verifier
                    .kfsb_single_shot_child_bounds_planned_for_tests(
                        &graph,
                        &domain,
                        &row,
                        None,
                        &candidates,
                        &plan,
                        threads,
                    )
                    .expect("planned simulations");
                assert!(
                    rec.max_budget_seen <= plan.share_bytes,
                    "{}: a simulation's worker saw {} > its share {}",
                    dag.shape,
                    rec.max_budget_seen,
                    plan.share_bytes
                );
                assert!(rec.max_budget_seen > 0, "positive control: the probe read");
                assert!(plan.width * plan.share_bytes <= plan.total_bytes);
                assert_eq!(
                    bits(&bounds),
                    bits(&sims),
                    "{}: plan changed a bound",
                    dag.shape
                );
                sims_checked += rec.sims;
            }
            // An explicit width-k plan and a width-1 plan give bit-identical
            // bounds (the share is enough for these passes; the width is
            // scheduling only).
            let wide = plan_kfsb_fanout(k * KFSB_MIN_SHARE_BYTES, None, k, k, KFSB_MIN_SHARE_BYTES);
            let narrow = plan_kfsb_fanout(KFSB_MIN_SHARE_BYTES, None, k, k, KFSB_MIN_SHARE_BYTES);
            assert_eq!((wide.width, narrow.width), (k, 1));
            let (w, w_rec) = verifier
                .kfsb_single_shot_child_bounds_planned_for_tests(
                    &graph,
                    &domain,
                    &row,
                    None,
                    &candidates,
                    &wide,
                    k,
                )
                .expect("wide");
            let (n, n_rec) = verifier
                .kfsb_single_shot_child_bounds_planned_for_tests(
                    &graph,
                    &domain,
                    &row,
                    None,
                    &candidates,
                    &narrow,
                    1,
                )
                .expect("narrow");
            assert_eq!(bits(&w), bits(&n), "{}: width changed a bound", dag.shape);
            assert_eq!(bits(&w), bits(&sims), "{}: live plan differs", dag.shape);
            assert_eq!(w_rec.plan, Some(wide));
            assert_eq!(n_rec.plan, Some(narrow));
            assert_eq!(w_rec.sims, n_rec.sims);
            assert_eq!(
                (w_rec.gpu_sims, w_rec.gpu_refused),
                (0, 0),
                "no engine: CPU"
            );
            assert_eq!(w_rec.cpu_sims, w_rec.sims);
            compared += k;
        }
    }
    println!("[kfsb-production] SIMS probed={sims_checked} compared={compared}");
    assert!(sims_checked > 40 && compared > 40, "positive control");
    println!(
        "[kfsb-production] TEST_COMPLETE name=kfsb_production_simulations_run_under_the_shared_budget"
    );
}

// ---------------------------------------------------------------------------
// 3. The R4 lease.
// ---------------------------------------------------------------------------

fn armed_authority(mem_gib: u64, reserve_gib: u64) -> LeaseAuthority {
    let plan = lease_plan(&HostEnvelope {
        class: HostClass::Other,
        threads: 8,
        mem_bytes: mem_gib * GIB,
        reserve_bytes: reserve_gib * GIB,
        device: DeviceQual::None,
        declared: true,
    })
    .expect("a known envelope");
    let authority = LeaseAuthority::new();
    authority.arm(plan).expect("arm");
    authority
}

#[test]
fn kfsb_production_lease_is_asked_once_per_lane_and_returned() {
    // Unarmed: nothing asked, nothing held, the live budget.
    let telemetry = KfsbSingleShotTelemetry::default();
    let unarmed = LeaseAuthority::new();
    let lease = telemetry.fanout_lease(&unarmed);
    assert_eq!(lease.decision, None);
    assert_eq!((unarmed.snapshot().events).len(), 0);

    // 64 GiB host, 16 GiB reserve: E = 48, terminal 24, pool 24 -> the
    // fan-out's 16 GiB want is GRANTED inside a lane, held once, returned at
    // the outermost lane end.
    let authority = armed_authority(64, 16);
    telemetry.enter_lane_for_tests();
    telemetry.enter_lane_for_tests(); // a nested sub-verifier lane
    let first = telemetry.fanout_lease(&authority);
    let second = telemetry.fanout_lease(&authority);
    assert_eq!(
        first.decision,
        Some(LeaseDecision::Granted { bytes: 16 * GIB })
    );
    assert_eq!(second.decision, first.decision, "asked once per lane");
    assert_eq!(authority.held(LeaseHolder::KfsbFanout), 16 * GIB);
    let plan = plan_kfsb_fanout(12 << 30, first.decision, 20, 20, KFSB_MIN_SHARE_BYTES);
    assert_eq!(
        plan.total_bytes,
        4 << 30,
        "16 GiB leased covers 4 GiB of dense pairs"
    );
    telemetry.finish_lease(&authority, &first);
    assert_eq!(
        authority.held(LeaseHolder::KfsbFanout),
        16 * GIB,
        "a lane-held lease survives the decision"
    );
    let _ = telemetry.leave_lane_for_tests(&authority);
    assert_eq!(
        authority.held(LeaseHolder::KfsbFanout),
        16 * GIB,
        "the inner lane does not return it"
    );
    let decision = telemetry.leave_lane_for_tests(&authority);
    assert!(decision.is_some_and(LeaseDecision::admits));
    assert_eq!(authority.held(LeaseHolder::KfsbFanout), 0, "returned");
    let snap = authority.snapshot();
    assert_eq!(
        (snap.granted, snap.released),
        (2, 1),
        "terminal + fan-out, one release"
    );

    // A pool the concurrent margin-row worker already holds: NARROWED to what
    // is left (>= one share at its peak), so fewer run at once.
    let authority = armed_authority(52, 16); // E = 36, terminal 18, pool 18
    assert!(authority
        .acquire(LeaseHolder::MarginRowConcurrent)
        .is_some_and(LeaseDecision::admits));
    telemetry.enter_lane_for_tests();
    let narrowed = telemetry.fanout_lease(&authority);
    assert_eq!(
        narrowed.decision,
        Some(LeaseDecision::Narrowed {
            bytes: 2 * GIB,
            want: 16 * GIB
        })
    );
    let plan = plan_kfsb_fanout(12 << 30, narrowed.decision, 20, 20, KFSB_MIN_SHARE_BYTES);
    assert_eq!((plan.total_bytes, plan.width), (512 * MIB, 2));
    let _ = telemetry.leave_lane_for_tests(&authority);
    assert_eq!(authority.held(LeaseHolder::KfsbFanout), 0);

    // Nothing left: DENIED, asked once, the fan-out runs one dark-equivalent
    // pass at a time, and nothing is held or returned.
    let authority = armed_authority(48, 16); // E = 32, terminal 16, pool 16
    assert!(authority
        .acquire(LeaseHolder::MarginRowConcurrent)
        .is_some_and(LeaseDecision::admits));
    telemetry.enter_lane_for_tests();
    let denied = telemetry.fanout_lease(&authority);
    let again = telemetry.fanout_lease(&authority);
    assert!(matches!(
        denied.decision,
        Some(LeaseDecision::Denied { .. })
    ));
    assert_eq!(again.decision, denied.decision);
    assert_eq!(authority.snapshot().denied, 1, "a denial is recorded once");
    let plan = plan_kfsb_fanout(12 << 30, denied.decision, 20, 20, KFSB_MIN_SHARE_BYTES);
    assert_eq!((plan.width, plan.share_bytes), (1, 12 << 30));
    let _ = telemetry.leave_lane_for_tests(&authority);
    assert_eq!(authority.snapshot().released, 0);

    // Outside any lane scope: acquired for the decision and returned after it.
    let authority = armed_authority(64, 16);
    let per_decision = telemetry.fanout_lease(&authority);
    assert!(per_decision.decision.is_some_and(LeaseDecision::admits));
    assert_eq!(authority.held(LeaseHolder::KfsbFanout), 16 * GIB);
    telemetry.finish_lease(&authority, &per_decision);
    assert_eq!(authority.held(LeaseHolder::KfsbFanout), 0);
    // The plan's need is the narrowable fan-out model.
    assert_eq!(
        authority
            .plan()
            .expect("armed")
            .need(LeaseHolder::KfsbFanout),
        LeaseNeed {
            want_bytes: 16 * GIB,
            min_bytes: DENSE_PAIR_PEAK_FACTOR * KFSB_FANOUT_MIN_SHARE_BYTES,
        }
    );
    println!(
        "[kfsb-production] TEST_COMPLETE name=kfsb_production_lease_is_asked_once_per_lane_and_returned"
    );
}

// ---------------------------------------------------------------------------
// 4. The GPU per-domain route (untrusted search).
// ---------------------------------------------------------------------------

/// The device route's numbers are a DIFFERENT sound relaxation (the
/// per-domain resnet fold with default slopes over the constrained bounds,
/// f64 in the hermetic adapter) from the CPU constrained pass (inherited α),
/// so near-tied candidates can rank differently: the pick is checked against
/// the CPU pick (bit-equal wherever the two routes return bit-equal bounds,
/// and on at least 90% of the fixture decisions — measured 30 of 32 at
/// a42b64cb5, both misses ties under the device numbers), and every device
/// bound is checked to ENCLOSE the enumerated child range (it only ranks,
/// but it must still be a bound of the region it names).
#[ntest::timeout(600000)]
#[test]
fn kfsb_production_gpu_route_picks_an_unstable_unsplit_neuron_from_enclosing_bounds() {
    let _env = crate::tests::lock_env_shared();
    let device = HermeticSoundGpuCrownEngine::default();
    let engine: &dyn ny_core::GemmEngine = &device;
    let tol = 1e-4;
    let mut gpu_total = 0usize;
    let mut both_ran = 0usize;
    let mut agree = 0usize;
    let mut bounds_equal = 0usize;
    let mut contained = 0usize;
    let mut picks_checked = 0usize;
    let mut differing = Vec::new();
    let mut violations = Vec::new();
    let mut bad = Vec::new();
    let mut fixtures = Vec::new();
    for seed in [3u64, 11, 29, 47, 5, 71, 97, 131] {
        fixtures.push(conv_residual(seed ^ 0x7f4a));
    }
    for dag in &fixtures {
        let (graph, domain) = root(dag);
        // A depth-1 domain too, so "unsplit" is non-trivial.
        let verifier = BetaCrownVerifier::new(kfsb_config(true));
        let unstable_root =
            verifier.find_unstable_graph_neurons(&graph, &domain, &dag.relu_nodes());
        let mut domains = vec![domain.clone()];
        if let Some((node, idx)) = unstable_root.first() {
            if let Some(child) = domain
                .with_constraint(&graph, split(node, *idx, true), false)
                .expect("child")
            {
                domains.push(child);
            }
        }
        for domain in &domains {
            let unstable = verifier.find_unstable_graph_neurons(&graph, domain, &dag.relu_nodes());
            if unstable.is_empty() {
                continue;
            }
            let candidates = candidates_of(&unstable);
            for row in [[1.0f32], [-1.0]] {
                let k = candidates.len();
                let plan =
                    plan_kfsb_fanout(k * KFSB_MIN_SHARE_BYTES, None, k, k, KFSB_MIN_SHARE_BYTES);
                let (gpu_bounds, rec) = verifier
                    .kfsb_single_shot_child_bounds_planned_for_tests(
                        &graph,
                        domain,
                        &row,
                        Some(engine),
                        &candidates,
                        &plan,
                        k,
                    )
                    .expect("gpu-route simulations");
                let (cpu_bounds, cpu_rec) = verifier
                    .kfsb_single_shot_child_bounds_planned_for_tests(
                        &graph,
                        domain,
                        &row,
                        None,
                        &candidates,
                        &plan,
                        k,
                    )
                    .expect("cpu simulations");
                gpu_total += rec.gpu_sims;
                assert_eq!(rec.gpu_sims + rec.cpu_sims, rec.sims);
                assert_eq!(
                    rec.gpu_refused, rec.cpu_sims,
                    "a sound GPU CROWN engine: every CPU pass is a refusal"
                );
                assert_eq!((cpu_rec.gpu_sims, cpu_rec.gpu_refused), (0, 0));
                // Every device bound encloses the enumerated child range.
                for (candidate, (active, inactive)) in candidates.iter().zip(&gpu_bounds) {
                    for (is_active, side) in [(true, active), (false, inactive)] {
                        let Some((lower, upper)) = side else {
                            continue;
                        };
                        let mut splits = domain.history.constraints.clone();
                        splits.push(split(&candidate.node_name, candidate.neuron_idx, is_active));
                        let Some((lo, hi)) = dag.range(&splits, f64::from(row[0])) else {
                            continue;
                        };
                        contained += 1;
                        if f64::from(*lower) > lo + tol || f64::from(*upper) < hi - tol {
                            violations.push(format!(
                                "{} depth={} {}[{}] active={is_active} row={row:?}: device \
                                 [{lower}, {upper}] misses grid [{lo}, {hi}]",
                                dag.shape, domain.depth, candidate.node_name, candidate.neuron_idx
                            ));
                        }
                    }
                }
                // The selection, armed, with and without the engine.
                let pick_gpu = verifier
                    .select_graph_branch_kfsb_in_gpu_batched(
                        &graph,
                        domain,
                        &unstable,
                        &row,
                        Some(engine),
                    )
                    .expect("gpu pick");
                let pick_cpu = verifier
                    .select_graph_branch_kfsb_in_gpu_batched(&graph, domain, &unstable, &row, None)
                    .expect("cpu pick");
                picks_checked += 1;
                let in_unstable = unstable
                    .iter()
                    .any(|(n, i)| *n == pick_gpu.0 && *i == pick_gpu.1);
                let already_split = domain
                    .history
                    .constraints
                    .iter()
                    .any(|c| c.node_name == pick_gpu.0 && c.neuron_idx == pick_gpu.1);
                if !in_unstable || already_split {
                    bad.push(format!(
                        "{} depth={} row={row:?}: gpu pick {}:{} unstable={in_unstable} \
                         split={already_split}",
                        dag.shape, domain.depth, pick_gpu.0, pick_gpu.1
                    ));
                }
                if rec.gpu_sims == rec.sims && rec.sims > 0 {
                    both_ran += 1;
                    let same_pick =
                        (pick_gpu.0.as_str(), pick_gpu.1) == (pick_cpu.0.as_str(), pick_cpu.1);
                    if bits(&gpu_bounds) == bits(&cpu_bounds) {
                        bounds_equal += 1;
                        assert!(
                            same_pick,
                            "{}: bit-equal bounds must give the bit-equal pick",
                            dag.shape
                        );
                    }
                    if same_pick {
                        agree += 1;
                    } else {
                        differing.push(format!(
                            "{} depth={} row={row:?}: gpu {}:{} cpu {}:{}",
                            dag.shape, domain.depth, pick_gpu.0, pick_gpu.1, pick_cpu.0, pick_cpu.1
                        ));
                    }
                }
            }
        }
    }
    println!(
        "[kfsb-production] GPU gpu_sims={gpu_total} picks={picks_checked} both_ran={both_ran} \
         agree={agree} bounds_equal={bounds_equal} contained={contained} differing={differing:?}"
    );
    assert!(
        gpu_total > 0 && both_ran > 0,
        "VOID: the GPU route never engaged (gpu_sims={gpu_total}, both_ran={both_ran})"
    );
    assert!(
        contained > 40,
        "positive control: {contained} device bounds checked"
    );
    assert!(
        violations.is_empty(),
        "a device-route simulation is not a bound of its region: {violations:#?}"
    );
    assert!(
        bad.is_empty(),
        "the GPU-route pick is not an unstable unsplit neuron: {bad:#?}"
    );
    assert!(
        agree * 10 >= both_ran * 9,
        "the GPU-route pick agrees with the CPU pick on only {agree} of {both_ran}: {differing:#?}"
    );
    println!(
        "[kfsb-production] TEST_COMPLETE name=kfsb_production_gpu_route_picks_an_unstable_unsplit_neuron_from_enclosing_bounds"
    );
}

// ---------------------------------------------------------------------------
// 5. The observable cap.
// ---------------------------------------------------------------------------

#[test]
fn kfsb_production_observable_is_capped_with_a_summary() {
    let telemetry = KfsbSingleShotTelemetry::default();
    let plan = plan_kfsb_fanout(12 << 30, None, 20, 20, KFSB_MIN_SHARE_BYTES);
    let record = KfsbDecisionRecord {
        sims: 38,
        infeasible: 2,
        cpu_sims: 30,
        gpu_sims: 8,
        gpu_refused: 30,
        prescore_secs: 0.011,
        sim_secs: 0.14,
        pool_secs: 0.0004,
        plan: Some(plan),
        max_budget_seen: 0,
    };
    let total = 300u64;
    let printed: u64 = (0..total)
        .map(|_| u64::from(telemetry.record_decision(&record)))
        .sum();
    assert_eq!(printed, KFSB_SINGLE_SHOT_DECISION_LINES);
    let counts = telemetry.snapshot();
    assert_eq!(counts.decisions, total);
    assert_eq!(counts.printed, KFSB_SINGLE_SHOT_DECISION_LINES);
    assert_eq!(counts.suppressed, total - KFSB_SINGLE_SHOT_DECISION_LINES);
    assert_eq!(counts.sims, 38 * total);
    assert_eq!(counts.gpu_sims, 8 * total);
    let line = decision_line(3, 40, 40, 20, &record, Some(("Relu_26", 7)));
    for needle in [
        "[kfsb-single-shot] armed=1 depth=3 ",
        " sims=38 ",
        " prescore_ms=11.0 ",
        " sim_ms=140.0 ",
        " pick=Relu_26:7 ",
        " gpu=8 cpu=30 gpu_refused=30 ",
        " width=20 ",
        " budget=live ",
    ] {
        assert!(line.contains(needle), "{needle:?} in {line}");
    }
    let summary = summary_line(
        "multi_objective",
        &counts,
        (plan.share_bytes as u64, plan.share_bytes as u64),
        20,
        Some(LeaseDecision::Narrowed {
            bytes: 2 * GIB,
            want: 16 * GIB,
        }),
        12.5,
    );
    for needle in [
        "[kfsb-single-shot] summary lane=multi_objective armed=1 decisions=300 printed=32 \
         suppressed=268 ",
        " gpu=2400 cpu=9000 gpu_refused=9000 ",
        " width_max=20 ",
        " lease=narrowed:2.00GiB ",
    ] {
        assert!(summary.contains(needle), "{needle:?} in {summary}");
    }
    assert!(
        !summary.contains("[kfsb-single-shot] armed=1"),
        "the summary is not a decision line"
    );
    let empty = summary_line(
        "domain_list",
        &KfsbSingleShotCounts::default(),
        (0, 0),
        0,
        None,
        0.0,
    );
    assert!(empty.contains(" decisions=0 ") && empty.contains(" lease=none "));
    println!(
        "[kfsb-production] TEST_COMPLETE name=kfsb_production_observable_is_capped_with_a_summary"
    );
}

/// review(kfsb-production): the summary's width / share extremes are the
/// LANE's, like its deltas. A lineage runs one lane per clause or disjunct
/// (the CLI's `with_config_from` sub-verifiers share the telemetry), and the
/// running extremes were never reset, so the second lane's summary printed
/// the first lane's widest fan-out and largest share as its own.
#[test]
fn review_kfsb_production_summary_extremes_are_per_lane() {
    let telemetry = KfsbSingleShotTelemetry::default();
    let unarmed = LeaseAuthority::new();
    let record = |plan| KfsbDecisionRecord {
        sims: 2,
        cpu_sims: 2,
        plan: Some(plan),
        ..KfsbDecisionRecord::default()
    };
    // Lane 1: 20 simulations at 614 MiB each.
    let wide = plan_kfsb_fanout(12 << 30, None, 20, 20, KFSB_MIN_SHARE_BYTES);
    telemetry.enter_lane_for_tests();
    let _ = telemetry.record_decision(&record(wide));
    let wide_share = wide.share_bytes as u64;
    assert_eq!(
        telemetry.lane_extremes_for_tests(),
        ((wide_share, wide_share), 20),
        "positive control: lane 1 records its own extremes"
    );
    let _ = telemetry.leave_lane_for_tests(&unarmed);
    // Lane 2 on the same lineage: 2 simulations at the minimum share.
    let narrow = plan_kfsb_fanout(2 * KFSB_MIN_SHARE_BYTES, None, 2, 20, KFSB_MIN_SHARE_BYTES);
    assert_eq!(narrow.width, 2);
    telemetry.enter_lane_for_tests();
    let _ = telemetry.record_decision(&record(narrow));
    let narrow_share = narrow.share_bytes as u64;
    assert_eq!(
        telemetry.lane_extremes_for_tests(),
        ((narrow_share, narrow_share), 2),
        "lane 2's summary must not carry lane 1's widths and shares"
    );
    // A NESTED lane keeps the enclosing lane's extremes (no reset inside).
    telemetry.enter_lane_for_tests();
    let _ = telemetry.record_decision(&record(wide));
    assert_eq!(
        telemetry.lane_extremes_for_tests(),
        ((narrow_share, wide_share), 20)
    );
    let _ = telemetry.leave_lane_for_tests(&unarmed);
    let _ = telemetry.leave_lane_for_tests(&unarmed);
    println!(
        "[kfsb-production] TEST_COMPLETE name=review_kfsb_production_summary_extremes_are_per_lane"
    );
}

/// End to end on the conjunctive lane: the lineage's decision count passes
/// the line budget on a long run and only the budget prints.
#[ntest::timeout(600000)]
#[test]
fn kfsb_production_lane_counts_every_decision_and_prints_the_budget() {
    let mut most = 0u64;
    for dag in dags() {
        let graph = dag.graph();
        let input = dag.input_box();
        let (_, hi) = dag.range(&[], 1.0).expect("range");
        let verifier = BetaCrownVerifier::new(kfsb_config(true));
        let before = kfsb_single_shot_calls_for_tests();
        let _ = verifier.verify_graph_relu_split_multi_objective_conjunctive_with_engine(
            &graph,
            &input,
            &[vec![-1.0], vec![1.0]],
            &[-(hi as f32 - 0.05), 1.0e6],
            None,
            None,
        );
        let calls = kfsb_single_shot_calls_for_tests() - before;
        let counts = verifier.kfsb_single_shot_telemetry.snapshot();
        assert_eq!(
            counts.decisions, calls,
            "{}: every decision counted",
            dag.shape
        );
        assert_eq!(
            counts.printed,
            calls.min(KFSB_SINGLE_SHOT_DECISION_LINES),
            "{}",
            dag.shape
        );
        assert_eq!(counts.printed + counts.suppressed, counts.decisions);
        most = most.max(calls);
    }
    println!("[kfsb-production] LANE most_decisions={most}");
    assert!(most > 0, "VOID: the armed lane never decided");
    println!(
        "[kfsb-production] TEST_COMPLETE name=kfsb_production_lane_counts_every_decision_and_prints_the_budget"
    );
}

// ---------------------------------------------------------------------------
// 6. Dark is untouched.
// ---------------------------------------------------------------------------

#[ntest::timeout(600000)]
#[test]
fn kfsb_production_dark_builds_no_scope_and_installs_no_share() {
    let dark = BetaCrownVerifier::new(kfsb_config(false));
    assert!(KfsbSingleShotLaneScope::enter(&dark, "multi_objective").is_none());
    let armed = BetaCrownVerifier::new(kfsb_config(true));
    assert!(
        KfsbSingleShotLaneScope::enter(&armed, "multi_objective").is_some(),
        "positive control: the armed key builds the scope"
    );
    let mut verified = 0usize;
    for dag in dags() {
        let graph = dag.graph();
        let input = dag.input_box();
        let (_, hi) = dag.range(&[], 1.0).expect("range");
        let verifier = BetaCrownVerifier::new(kfsb_config(false));
        let before = kfsb_single_shot_calls_for_tests();
        let result = verifier.verify_graph_relu_split_multi_objective_conjunctive_with_engine(
            &graph,
            &input,
            &[vec![-1.0], vec![1.0]],
            &[-(hi as f32 + 1.0), 1.0e6],
            None,
            None,
        );
        verified +=
            usize::from(result.is_ok_and(|r| matches!(r.result, BabVerificationStatus::Verified)));
        assert_eq!(
            kfsb_single_shot_calls_for_tests(),
            before,
            "dark never decides"
        );
        assert_eq!(
            verifier.kfsb_single_shot_telemetry.snapshot(),
            KfsbSingleShotCounts::default(),
            "dark records nothing"
        );
        assert_eq!(dense_budget_thread_cap(), None, "dark installs no share");
    }
    assert!(
        verified > 0,
        "positive control: the dark lane verifies TRUE rows"
    );
    println!(
        "[kfsb-production] TEST_COMPLETE name=kfsb_production_dark_builds_no_scope_and_installs_no_share"
    );
}
