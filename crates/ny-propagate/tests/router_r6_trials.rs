// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #router-r6 — the O-TRIAL rules END TO END on the real lanes, through the
//! PROCESS-GLOBAL trial authority the lanes consult.
//!
//! * the lowdim lane's `rule.trial_split_restart`: a 16-input box on which
//!   only ONE input moves the objective and the fifteen others are wider, so
//!   the widest-coordinate rule spends its first splits on coordinates that
//!   cannot lift a bound — the first two 16-wave windows read FAR (certified
//!   volume flat, queue growing) with every bounded child at zero lift, and
//!   the lane RESTARTS once with the other heuristic (`[router] trial
//!   knob=... dir=switch decision=started`), the tree discarded;
//! * the lowdim lane's `rule.trial_width`: four inputs move the objective, so
//!   the first windows close nothing and a later one closes domains — a rise
//!   from zero closes per second, which starts a width trial (`dir=up
//!   decision=started`) deterministically (no timing in the decision);
//! * the input-split loop's DORMANCY: with the site armed and `reorder_bab`
//!   off, the real loop records `blocked why=dormant:reorder_bab_off` and
//!   builds no machine;
//! * and DEFAULT BIT-IDENTITY: the same lowdim runs unarmed before and after
//!   (disarmed) produce identical counters.
//!
//! The objective is `relu(h) - relu(h) == 0` everywhere, the threshold
//! `-1e-12`: the property is TRUE, and no lane run may report a candidate.
//! Its own test process on purpose (it arms the process-global authority);
//! its one test runs serially.

use std::time::{Duration, Instant};

use ndarray::{Array2, ArrayD, IxDyn};
use ny_core::router::trial::{RestartRule, TrialDecision, TrialDir, WidthRule};
use ny_core::trial_authority::{self, TrialDirectives, TrialSite};
use ny_propagate::layers::{LinearLayer, ReLULayer};
use ny_propagate::lowdim::{prove_objective_lower_bound, LowDimConfig, LowDimOutcome, LowDimStop};
use ny_propagate::{GraphNetwork, GraphNode, Layer, NETWORK_INPUT};
use ny_tensor::BoundedTensor;

const DIMS: usize = 16;

/// `y = relu(h) - relu(h)` with `h` the sum of the LAST `relevant` inputs.
fn flat_graph(relevant: usize) -> GraphNetwork {
    let mut w1 = vec![0.0f32; 2 * DIMS];
    for i in DIMS - relevant..DIMS {
        w1[i] = 1.0;
        w1[DIMS + i] = 1.0;
    }
    let l1 = Array2::from_shape_vec((2, DIMS), w1).expect("shape");
    let l2 = Array2::from_shape_vec((1, 2), vec![1.0f32, -1.0]).expect("shape");
    let mut g = GraphNetwork::new();
    g.add_node(GraphNode::new(
        "l1",
        Layer::Linear(LinearLayer::new(l1, None).expect("l1")),
        vec![NETWORK_INPUT.into()],
    ));
    g.add_node(GraphNode::new(
        "r1",
        Layer::ReLU(ReLULayer::new()),
        vec!["l1".into()],
    ));
    g.add_node(GraphNode::new(
        "l2",
        Layer::Linear(LinearLayer::new(l2, None).expect("l2")),
        vec!["r1".into()],
    ));
    g.set_output("l2");
    g
}

/// Review fix (router-r6): `y = -2000 relu(x_15 - 1.999)`, a FALSE property
/// for `y > -1` on `input_box(1)`: `y <= -1` on the slab `x_15 >= 1.9995`.
/// The fifteen other inputs are dead (zero weight) and wider, so the widest
/// rule's first splits lift nothing (FAR + zero lift: the restart fires) and
/// every box's centre (`x_15 = 0.5` until `x_15` is split) is no candidate.
fn slab_graph() -> GraphNetwork {
    let mut w1 = vec![0.0f32; DIMS];
    w1[DIMS - 1] = 1.0;
    let l1 = Array2::from_shape_vec((1, DIMS), w1).expect("shape");
    let b1 = ndarray::Array1::from_vec(vec![-1.999f32]);
    let l2 = Array2::from_shape_vec((1, 1), vec![-2000.0f32]).expect("shape");
    let mut g = GraphNetwork::new();
    g.add_node(GraphNode::new(
        "l1",
        Layer::Linear(LinearLayer::new(l1, Some(b1)).expect("l1")),
        vec![NETWORK_INPUT.into()],
    ));
    g.add_node(GraphNode::new(
        "r1",
        Layer::ReLU(ReLULayer::new()),
        vec!["l1".into()],
    ));
    g.add_node(GraphNode::new(
        "l2",
        Layer::Linear(LinearLayer::new(l2, None).expect("l2")),
        vec!["r1".into()],
    ));
    g.set_output("l2");
    g
}

/// The irrelevant inputs span `[-4, 8]` (width 12), the relevant ones
/// `[-1, 2]` (width 3; no bisection midpoint is ever 0): the widest-coordinate
/// rule halves every irrelevant input twice before it touches a relevant one,
/// so the first 36 splits cannot lift any bound or close any box.
fn input_box(relevant: usize) -> BoundedTensor {
    let lo: Vec<f32> = (0..DIMS)
        .map(|i| if i >= DIMS - relevant { -1.0 } else { -4.0 })
        .collect();
    let hi: Vec<f32> = (0..DIMS)
        .map(|i| if i >= DIMS - relevant { 2.0 } else { 8.0 })
        .collect();
    let lo = ArrayD::from_shape_vec(IxDyn(&[1, DIMS]), lo).expect("lo");
    let hi = ArrayD::from_shape_vec(IxDyn(&[1, DIMS]), hi).expect("hi");
    BoundedTensor::new(lo, hi).expect("box")
}

fn cfg() -> LowDimConfig {
    LowDimConfig {
        batch: 8,
        ..LowDimConfig::default()
    }
}

fn run(relevant: usize) -> LowDimOutcome {
    prove_objective_lower_bound(
        &flat_graph(relevant),
        &input_box(relevant),
        &[1.0],
        -1e-12,
        Instant::now() + Duration::from_secs(3),
        &cfg(),
    )
}

fn stats_of(outcome: &LowDimOutcome) -> ny_propagate::lowdim::LowDimStats {
    match outcome {
        LowDimOutcome::Verified(s) | LowDimOutcome::Unproved(s, _) => {
            let mut s = *s;
            s.secs = 0.0;
            s
        }
        LowDimOutcome::Declined(why) => panic!("the lane declined: {why}"),
    }
}

fn assert_no_candidate(outcome: &LowDimOutcome) {
    assert!(
        !matches!(outcome, LowDimOutcome::Unproved(_, LowDimStop::Candidate)),
        "the property is true: no run may report a candidate: {outcome:?}"
    );
}

#[ntest::timeout(180000)]
#[test]
fn router_r6_trials_execute_end_to_end_on_the_real_lanes() {
    let authority = trial_authority::global();
    assert!(
        !authority.is_armed(),
        "nothing arms the authority in this process"
    );

    // Controls, unarmed: today's lane, bit for bit (counters; the wall clock
    // aside), and a deterministic lane (two runs agree).
    let control_one = run(1);
    let control_four = run(4);
    assert_no_candidate(&control_one);
    assert_no_candidate(&control_four);
    assert_eq!(
        stats_of(&control_one),
        stats_of(&run(1)),
        "the lane is deterministic"
    );
    assert_eq!(
        stats_of(&control_four),
        stats_of(&run(4)),
        "the lane is deterministic"
    );
    assert!(
        stats_of(&control_one).waves > 48,
        "the control runs long enough for three windows: {control_one:?}"
    );

    // ---- rule.trial_split_restart: FAR and zero lift => ONE restart ------
    authority
        .arm(TrialDirectives {
            split_restart: Some(RestartRule::SHIPPED),
            sites: TrialSite::LowdimSplit.bit(),
            ..TrialDirectives::NONE
        })
        .expect("arm the restart site");
    // The write-only progress bus shows the lane's own queue at its safe
    // points: the restart must DISCARD the tree (the queue drops back to the
    // root's descendants), not keep it.
    ny_core::router::bus::reset();
    ny_core::router::bus::enable(Instant::now());
    let restarted = run(1);
    let samples = ny_core::router::bus::drain();
    ny_core::router::bus::reset();
    assert_no_candidate(&restarted);
    let open_at = |wave: u64| {
        samples
            .iter()
            .find(|s| s.cap.as_str() == "lowdim" && s.unit_index == wave)
            .map(|s| s.open)
            .unwrap_or_else(|| panic!("no lowdim sample at wave {wave}: {samples:?}"))
    };
    assert!(
        open_at(36) < open_at(32),
        "the restart discarded the tree: open {} at wave 32 (before) vs {} at wave 36 (after)",
        open_at(32),
        open_at(36)
    );
    let snap = authority.snapshot();
    let restarts: Vec<_> = snap
        .events
        .iter()
        .filter(|e| e.dir == TrialDir::Switch && e.decision == TrialDecision::Started)
        .collect();
    assert_eq!(restarts.len(), 1, "exactly one restart per row: {snap:?}");
    let event = restarts[0];
    assert_eq!(event.knob, "bab.branching.input_split.lowdim_split");
    assert_eq!(event.site, "lowdim");
    assert!(event.why.contains("from=widest to=sbwr"), "{event:?}");
    assert!(
        event.metric > 0.9,
        "the zero-lift fraction it fired on: {event:?}"
    );
    assert_eq!(
        event.unit, 32,
        "the end of the second 16-wave window: {event:?}"
    );
    assert!(
        event
            .line()
            .starts_with("[router] trial knob=bab.branching.input_split.lowdim_split dir=switch decision=started"),
        "{}",
        event.line()
    );
    assert_ne!(
        stats_of(&restarted),
        stats_of(&control_one),
        "the restart changed the search (the tree was discarded and regrown)"
    );
    authority.disarm();

    // ---- review fix: a restart on a FALSE property never reads as a proof --
    // `y > -1` is false on the slab (see `slab_graph`). The widest rule's
    // first windows are FAR with every child at zero lift, so the lane
    // restarts; whatever the new tree then does, the lane must NEVER report
    // Verified. A restart that left the queue EMPTY (it reads as a proof at
    // the next wave top) or bounded a stale box in the root's place would.
    let slab = |deadline: Instant| {
        prove_objective_lower_bound(&slab_graph(), &input_box(1), &[1.0], -1.0, deadline, &cfg())
    };
    let slab_control = slab(Instant::now() + Duration::from_secs(3));
    assert!(
        !matches!(slab_control, LowDimOutcome::Verified(_)),
        "control: the slab property is false: {slab_control:?}"
    );
    authority
        .arm(TrialDirectives {
            split_restart: Some(RestartRule::SHIPPED),
            sites: TrialSite::LowdimSplit.bit(),
            ..TrialDirectives::NONE
        })
        .expect("arm the restart site (a new row)");
    let slab_restarted = slab(Instant::now() + Duration::from_secs(3));
    let snap = authority.snapshot();
    assert!(
        snap.events
            .iter()
            .any(|e| e.dir == TrialDir::Switch && e.decision == TrialDecision::Started),
        "positive control: the restart FIRED on the false property: {snap:?}"
    );
    assert!(
        !matches!(slab_restarted, LowDimOutcome::Verified(_)),
        "SOUNDNESS: a restart turned a false property into a proof: {slab_restarted:?}"
    );
    authority.disarm();

    // ---- rule.trial_width: a rise from zero closes => a raise -----------
    authority
        .arm(TrialDirectives {
            width: Some(WidthRule::SHIPPED),
            sites: TrialSite::LowdimWidth.bit(),
            ..TrialDirectives::NONE
        })
        .expect("arm the width site");
    let widened = run(4);
    assert_no_candidate(&widened);
    let snap = authority.snapshot();
    let raise = snap
        .events
        .iter()
        .find(|e| e.decision == TrialDecision::Started && e.dir == TrialDir::Up)
        .unwrap_or_else(|| panic!("a width trial started: {snap:?}"));
    assert_eq!(raise.knob, "lowdim.batch");
    assert_eq!((raise.from, raise.to), (8.0, 16.0), "x2 from today's width");
    assert_eq!(
        raise.baseline, raise.metric,
        "judged against its own window"
    );
    for e in &snap.events {
        assert!(
            (2.0..=32.0).contains(&e.to) && (2.0..=32.0).contains(&e.from),
            "every trialled width stays inside [8/4, 8x4]: {e:?}"
        );
    }
    assert!(
        snap.counts.started <= 2,
        "<= 2 trials per knob per row: {snap:?}"
    );
    authority.disarm();

    // ---- the input-split loop's dormancy, through the real loop ----------
    authority
        .arm(TrialDirectives {
            width: Some(WidthRule::SHIPPED),
            sites: TrialSite::InputSplitWidth.bit(),
            ..TrialDirectives::NONE
        })
        .expect("arm the input-split site");
    let verifier = ny_propagate::BetaCrownVerifier::new(ny_propagate::BetaCrownConfig {
        branching_heuristic: ny_propagate::BranchingHeuristic::InputSplit,
        verify_upper_bound: false,
        use_alpha_crown: false,
        use_crown_ibp: false,
        enable_cuts: false,
        enable_relaxed_clip: false,
        input_split_ibp_enhancement: false,
        max_domains: 64,
        max_depth: 4,
        batch_size: 64,
        timeout: Duration::from_secs(5),
        reorder_bab: false,
        ..Default::default()
    });
    let _ = verifier.verify_graph_input_split_with_engine(
        &flat_graph(1),
        &input_box(1),
        &[1.0_f32],
        -1e-12,
        None,
        None,
    );
    let snap = authority.snapshot();
    assert!(
        snap.events.iter().any(|e| e.site == "input_split"
            && e.decision == TrialDecision::Blocked
            && e.why == "dormant:reorder_bab_off"),
        "the real loop consulted the dormancy rule: {snap:?}"
    );
    assert_eq!(snap.counts.started, 0, "a dormant batch is never trialled");
    authority.disarm();

    // ---- disarmed: today's lane again, bit for bit ------------------------
    assert_eq!(stats_of(&run(1)), stats_of(&control_one));
    assert_eq!(stats_of(&run(4)), stats_of(&control_four));
}
