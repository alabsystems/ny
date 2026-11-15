// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #iso-edge-cap tests: the dark-by-default property, the allowance law, and
//! the soundness of the governed edge path (sampled containment, never
//! verifying a violated row, reuse equals recomputation, no authority past
//! the stop).

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use ndarray::{arr1, arr2, Array2};
use ny_tensor::BoundedTensor;

use super::super::super::shared::{
    extract_obj_bounds, graph_spec_ibp_fallback, MultiObjInputDomain,
};
use super::super::process_batch::{
    edge_domain_rows, escalate_edge_domain, try_edge_alpha_pass, try_edge_milp_escalation,
    try_edge_milp_escalation_governed,
};
use super::super::BabWall;
use super::*;
use crate::beta_crown::config::{BetaCrownConfig, EdgeWaveCapConfig};
use crate::beta_crown::engine::BetaCrownVerifier;
use crate::beta_crown::graph_mip_leaf::{
    GraphMipLeafOracle, GraphMipLeafRequest, GraphMipLeafVerdict,
};
use crate::layers::{LinearLayer, ReLULayer, SubLayer};
use crate::{GraphNetwork, GraphNode, Layer};

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const HORIZON: Duration = Duration::from_secs(100);

fn cap_config(max_secs: Option<f64>, rebound_share: Option<f64>) -> BetaCrownConfig {
    BetaCrownConfig {
        input_split_edge_wave_cap: Some(EdgeWaveCapConfig {
            max_secs,
            rebound_share,
        }),
        ..BetaCrownConfig::default()
    }
}

/// A wall with an absolute stop `stop_in` from `now` (or none), carrying an
/// armed edge wave of `allowance`.
fn armed_wall(now: Instant, stop_in: Option<Duration>, allowance: Duration) -> BabWall {
    BabWall::clamped(HORIZON, stop_in.map(|d| now + d), now).with_edge_wave(Some(EdgeWave {
        index: 7,
        allowance,
    }))
}

/// The sub-towers difference miniature (near-identical towers joined by Sub,
/// the isomorphic lane's shape in two dimensions).
fn sub_towers_graph() -> GraphNetwork {
    let wa1 = arr2(&[[1.0_f32, -0.7], [0.6, 1.1], [-0.9, 0.8]]);
    let wb1 = wa1.mapv(|v| v * 1.01);
    let wa2 = arr2(&[[0.8_f32, -1.2, 0.5]]);
    let wb2 = wa2.mapv(|v| v * 0.99);
    let ba1 = arr1(&[0.05_f32, -0.1, 0.2]);
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "a_l1",
        Layer::Linear(LinearLayer::new(wa1, Some(ba1.clone())).unwrap()),
    ));
    graph.add_node(GraphNode::new(
        "a_r1",
        Layer::ReLU(ReLULayer),
        vec!["a_l1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "a_l2",
        Layer::Linear(LinearLayer::new(wa2, None).unwrap()),
        vec!["a_r1".to_string()],
    ));
    graph.add_node(GraphNode::from_input(
        "b_l1",
        Layer::Linear(LinearLayer::new(wb1, Some(ba1)).unwrap()),
    ));
    graph.add_node(GraphNode::new(
        "b_r1",
        Layer::ReLU(ReLULayer),
        vec!["b_l1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "b_l2",
        Layer::Linear(LinearLayer::new(wb2, None).unwrap()),
        vec!["b_r1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "diff",
        Layer::Sub(SubLayer),
        vec!["a_l2".to_string(), "b_l2".to_string()],
    ));
    graph.set_output("diff");
    graph
}

fn unit_box() -> BoundedTensor {
    BoundedTensor::new(
        arr1(&[-0.5_f32, -0.5]).into_dyn(),
        arr1(&[0.5_f32, 0.5]).into_dyn(),
    )
    .unwrap()
}

/// `+y` and `-y` over the single difference output, one row per clause.
fn band_spec() -> Array2<f32> {
    arr2(&[[1.0_f32], [-1.0_f32]])
}

fn forward(graph: &GraphNetwork, x: [f32; 2]) -> f32 {
    let point = BoundedTensor::new(arr1(&x).into_dyn(), arr1(&x).into_dyn()).unwrap();
    let out = graph
        .propagate_concrete_point(&point, None, None)
        .expect("concrete forward");
    *out.lower().iter().next().expect("one output")
}

/// A deterministic 41 x 41 grid over the unit box (corners included).
fn grid_outputs(graph: &GraphNetwork) -> Vec<f32> {
    let n = 41;
    let mut out = Vec::with_capacity(n * n);
    for i in 0..n {
        for j in 0..n {
            let a = -0.5 + (i as f32) / ((n - 1) as f32);
            let b = -0.5 + (j as f32) / ((n - 1) as f32);
            out.push(forward(graph, [a, b]));
        }
    }
    out
}

fn sampled_min_max(graph: &GraphNetwork) -> (f32, f32) {
    grid_outputs(graph)
        .into_iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), y| {
            (lo.min(y), hi.max(y))
        })
}

/// A popped domain over the unit box with SOUND (IBP) starting bounds, deep
/// enough for any gate.
fn domain(graph: &GraphNetwork) -> MultiObjInputDomain {
    let input = unit_box();
    let (bounds, _) =
        graph_spec_ibp_fallback(graph, &input, &band_spec(), None, None).expect("ibp bounds");
    MultiObjInputDomain {
        input_bounds: Arc::new(input),
        obj_bounds: extract_obj_bounds(&bounds, 2).expect("two rows"),
        linear_bounds: None,
        depth: 30,
        priority: 0.0,
        needs_bounding: false,
        node_bounds_override: None,
        inherited_alpha_state: None,
    }
}

fn edge_verifier(alpha: bool, milp: bool) -> BetaCrownVerifier {
    BetaCrownVerifier::new(BetaCrownConfig {
        input_split_edge_alpha: alpha,
        input_split_edge_alpha_top: 64,
        input_split_edge_alpha_iters: 40,
        input_split_edge_milp: milp,
        input_split_edge_milp_gap: 100.0,
        input_split_edge_milp_depth: 0,
        ..BetaCrownConfig::default()
    })
}

/// Thresholds that make row 0 (`y > t0`) FALSE on the box — a grid point sits
/// below `t0` — and row 1 (`-y > t1`) true. The grouped domain therefore must
/// NEVER be verified.
fn violated_thresholds(graph: &GraphNetwork) -> [f32; 2] {
    let (lo, hi) = sampled_min_max(graph);
    [lo + 0.25 * (hi - lo) + 1e-3, -hi - 1.0]
}

/// A TRUE but tight property: both rows hold on every grid point with a 5 %
/// margin of the sampled range.
fn true_thresholds(graph: &GraphNetwork) -> [f32; 2] {
    let (lo, hi) = sampled_min_max(graph);
    [lo - 0.05 * (hi - lo), -hi - 0.05 * (hi - lo)]
}

/// Every lower bound must sit at or below every sampled value of its row.
fn assert_contained(graph: &GraphNetwork, obj_bounds: &[(f32, f32)], what: &str) {
    let (lo, hi) = sampled_min_max(graph);
    let tol = 1e-5_f32;
    assert!(
        obj_bounds[0].0 <= lo + tol,
        "{what}: row +y lower {} exceeds the sampled minimum {lo} — an UNSOUND bound",
        obj_bounds[0].0
    );
    assert!(
        obj_bounds[1].0 <= -hi + tol,
        "{what}: row -y lower {} exceeds the sampled minimum {} — an UNSOUND bound",
        obj_bounds[1].0,
        -hi
    );
}

// ---------------------------------------------------------------------------
// Dark by default
// ---------------------------------------------------------------------------

/// The shipped state: no block, no policy, no wave on any wall.
#[test]
fn edge_wave_cap_ships_dark_by_default() {
    let config = BetaCrownConfig::default();
    assert_eq!(config.input_split_edge_wave_cap, None);
    assert_eq!(resolve_edge_wave_cap(&config), None);
    // The acas_xu base the relational lane builds on is dark too.
    assert_eq!(BetaCrownConfig::acas_xu().input_split_edge_wave_cap, None);
}

/// Every wall `clamped` builds carries no wave, so a dark run's governor is
/// unarmed, never refuses, and the processors take the historical path.
#[test]
fn a_clamped_wall_carries_no_edge_wave_and_its_governor_is_unarmed() {
    let now = Instant::now();
    for wall in [
        BabWall::clamped(HORIZON, None, now),
        BabWall::clamped(HORIZON, Some(now + Duration::from_secs(10)), now),
    ] {
        assert_eq!(wall.edge_wave(), None);
        assert_eq!(
            wall.with_edge_wave(None),
            wall,
            "attaching nothing must be the identity"
        );
        let governor = EdgeWaveGovernor::for_wave(wall, now);
        assert!(!governor.is_armed());
        assert!(governor.may_start(now));
        assert!(governor.may_start(now + Duration::from_secs(1_000)));
    }
}

// ---------------------------------------------------------------------------
// The policy
// ---------------------------------------------------------------------------

/// An empty block arms with the WAVE6_PLAN §7.C6 defaults: min(2 s, 0.25 x).
#[test]
fn edge_wave_cap_empty_block_takes_the_plan_defaults() {
    let cap = resolve_edge_wave_cap(&cap_config(None, None)).expect("presence arms");
    assert_eq!(cap.max_secs(), EDGE_WAVE_CAP_DEFAULT_MAX_SECS);
    assert_eq!(cap.rebound_share(), EDGE_WAVE_CAP_DEFAULT_REBOUND_SHARE);
    assert_eq!(EDGE_WAVE_CAP_DEFAULT_MAX_SECS, 2.0);
    assert_eq!(EDGE_WAVE_CAP_DEFAULT_REBOUND_SHARE, 0.25);
}

#[test]
fn edge_wave_cap_named_fields_override_their_defaults() {
    let cap = resolve_edge_wave_cap(&cap_config(Some(5.0), Some(0.5))).unwrap();
    assert_eq!(cap.max_secs(), 5.0);
    assert_eq!(cap.rebound_share(), 0.5);
    // Zero is honoured: the typed way to measure "no edge work at all".
    let zero = resolve_edge_wave_cap(&cap_config(Some(0.0), None)).unwrap();
    assert_eq!(zero.allowance(10.0), Duration::ZERO);
}

/// A malformed number fails OPEN into today's shipped behaviour (dark), and
/// never panics.
#[test]
fn edge_wave_cap_malformed_fields_fail_open_to_dark() {
    for (max_secs, share) in [
        (Some(f64::NAN), None),
        (Some(f64::INFINITY), None),
        (Some(-1.0), None),
        (None, Some(f64::NAN)),
        (None, Some(f64::NEG_INFINITY)),
        (None, Some(-0.25)),
    ] {
        assert_eq!(
            resolve_edge_wave_cap(&cap_config(max_secs, share)),
            None,
            "max_secs={max_secs:?} rebound_share={share:?} must resolve dark"
        );
    }
    // Absurdly large but finite values are clamped, not rejected.
    let big = resolve_edge_wave_cap(&cap_config(Some(1e300), Some(1e300))).unwrap();
    assert!(big.max_secs() <= 3_600.0);
    assert!(big.allowance(1e300) <= Duration::from_secs(3_600));
}

/// `allowance = min(max, rebound_share x rebound)`.
#[test]
fn edge_wave_allowance_is_the_min_of_the_ceiling_and_the_rebound_share() {
    let cap = resolve_edge_wave_cap(&cap_config(None, None)).unwrap();
    // The measured iso wave: a ~1.6 s rebound buys 0.4 s of edge work.
    let a = cap.allowance(1.6);
    assert!((a.as_secs_f64() - 0.4).abs() < 1e-9, "{a:?}");
    // A long rebound is capped at max_secs.
    assert_eq!(cap.allowance(40.0), Duration::from_secs(2));
    // The wave value carries index and allowance through.
    let wave = cap.wave(3, 1.6);
    assert_eq!(wave.index, 3);
    assert_eq!(wave.allowance, a);
}

/// No rebound measurement, no edge budget — and never a panic.
#[test]
fn edge_wave_allowance_is_zero_without_a_rebound_measurement() {
    let cap = resolve_edge_wave_cap(&cap_config(None, None)).unwrap();
    for rebound in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(cap.allowance(rebound), Duration::ZERO, "rebound={rebound}");
    }
}

// ---------------------------------------------------------------------------
// The governor
// ---------------------------------------------------------------------------

#[test]
fn governor_refuses_new_steps_once_the_allowance_is_spent() {
    let now = Instant::now();
    let wall = armed_wall(now, None, Duration::from_millis(400));
    let mut governor = EdgeWaveGovernor::for_wave(wall, now);
    assert!(governor.is_armed());
    assert!(governor.may_start(now));
    governor.charge(Duration::from_millis(399));
    assert!(governor.may_start(now), "1 ms of allowance is left");
    governor.charge(Duration::from_millis(1));
    assert!(!governor.may_start(now), "the allowance is spent");
    assert_eq!(governor.spent(), Duration::from_millis(400));
    governor.charge(Duration::from_secs(5));
    assert!(
        !governor.may_start(now),
        "an overshoot never re-opens the wave"
    );
}

#[test]
fn governor_refuses_new_steps_at_or_after_the_stop() {
    let now = Instant::now();
    // A stop 10 s out, a generous allowance.
    let wall = armed_wall(now, Some(Duration::from_secs(10)), Duration::from_secs(60));
    let governor = EdgeWaveGovernor::for_wave(wall, now);
    assert!(governor.may_start(now + Duration::from_secs(9)));
    assert!(!governor.may_start(now + Duration::from_secs(10)));
    assert!(!governor.may_start(now + Duration::from_secs(11)));
    assert_eq!(governor.stop_deadline(), now + Duration::from_secs(10));

    // A stop already passed is a zero stop: nothing may start.
    let passed = BabWall::clamped(
        HORIZON,
        Some(now.checked_sub(Duration::from_secs(5)).unwrap()),
        now,
    )
    .with_edge_wave(Some(EdgeWave {
        index: 0,
        allowance: Duration::from_secs(60),
    }));
    let governor = EdgeWaveGovernor::for_wave(passed, now);
    assert!(!governor.may_start(now));
}

/// The work deadline is never later than the stop, nor than the remaining
/// allowance from `now`.
#[test]
fn work_deadline_never_exceeds_the_stop_or_the_remaining_allowance() {
    let now = Instant::now();
    for (stop_in, allowance, spent) in [
        (Some(10u64), 60_000u64, 0u64),
        (Some(10), 400, 0),
        (Some(10), 400, 300),
        (Some(10), 400, 900),
        (None, 400, 100),
        (None, 0, 0),
    ] {
        let wall = armed_wall(
            now,
            stop_in.map(Duration::from_secs),
            Duration::from_millis(allowance),
        );
        let mut governor = EdgeWaveGovernor::for_wave(wall, now);
        governor.charge(Duration::from_millis(spent));
        let at = now + Duration::from_secs(1);
        let deadline = governor.work_deadline(at);
        assert!(deadline <= governor.stop_deadline());
        assert!(deadline <= now + HORIZON, "never past the horizon either");
        let remaining = Duration::from_millis(allowance.saturating_sub(spent));
        assert!(
            deadline <= at + remaining,
            "stop_in={stop_in:?} allowance={allowance} spent={spent}"
        );
    }
}

#[test]
fn a_zero_allowance_refuses_every_step() {
    let now = Instant::now();
    let governor = EdgeWaveGovernor::for_wave(armed_wall(now, None, Duration::ZERO), now);
    assert!(governor.is_armed());
    assert!(!governor.may_start(now));
    assert!(governor.work_deadline(now) <= now);
}

// ---------------------------------------------------------------------------
// The governed edge path: soundness
// ---------------------------------------------------------------------------

/// NEVER VERIFIES A VIOLATED ROW, and every bound it leaves behind contains
/// every sampled output — at a generous allowance (the α pass runs to its
/// iteration count), at a 1 ms allowance (cut short wherever the clock lands)
/// and at zero (refused outright).
#[test]
fn governed_edge_alpha_never_verifies_a_violated_row_and_stays_contained() {
    let _serial = crate::random::restart_offset_test_lock();
    let graph = sub_towers_graph();
    let verifier = edge_verifier(true, false);
    let spec = band_spec();
    let thresholds = violated_thresholds(&graph);
    let clause_sizes = [1usize, 1usize];
    for allowance in [
        Duration::from_secs(60),
        Duration::from_millis(1),
        Duration::ZERO,
    ] {
        let now = Instant::now();
        let mut governor = EdgeWaveGovernor::for_wave(armed_wall(now, None, allowance), now);
        let mut d = domain(&graph);
        let before = d.obj_bounds.clone();
        let mut budget = 64usize;
        let verified = escalate_edge_domain(
            &verifier,
            &graph,
            &mut d,
            &spec,
            &thresholds,
            &clause_sizes,
            None,
            now + HORIZON,
            &mut budget,
            &mut governor,
        );
        assert!(
            !verified,
            "allowance={allowance:?}: a domain holding a counterexample was VERIFIED"
        );
        assert_contained(&graph, &d.obj_bounds, &format!("allowance={allowance:?}"));
        for (after, before) in d.obj_bounds.iter().zip(&before) {
            assert!(after.0 >= before.0, "the merge must never loosen a bound");
        }
        if allowance.is_zero() {
            assert_eq!(governor.alpha_passes(), 0);
            assert_eq!(governor.skipped(), 1, "the refusal is the cap's engagement");
            assert_eq!(d.obj_bounds, before, "a refused step changes nothing");
            assert_eq!(budget, 64, "a refused step spends no α-pass budget");
        } else {
            assert_eq!(governor.alpha_passes(), 1, "the pass must have engaged");
            assert_eq!(budget, 63);
        }
    }
}

/// With a generous allowance and a far stop the governed pass reaches the
/// same bounds and the same decision as the unarmed (historical) pass on the
/// same domain: the cap removes WORK, not bound quality.
#[test]
fn governed_edge_alpha_matches_the_unarmed_pass_when_the_allowance_is_generous() {
    let _serial = crate::random::restart_offset_test_lock();
    let graph = sub_towers_graph();
    let verifier = edge_verifier(true, false);
    let spec = band_spec();
    // A true property, tight enough that the α pass has work to do.
    let thresholds = true_thresholds(&graph);
    let clause_sizes = [1usize, 1usize];

    let now = Instant::now();
    let mut dark = domain(&graph);
    let mut dark_governor = EdgeWaveGovernor::for_wave(BabWall::clamped(HORIZON, None, now), now);
    let mut dark_budget = 64usize;
    let dark_verified = escalate_edge_domain(
        &verifier,
        &graph,
        &mut dark,
        &spec,
        &thresholds,
        &clause_sizes,
        None,
        now + HORIZON,
        &mut dark_budget,
        &mut dark_governor,
    );

    let mut armed = domain(&graph);
    let mut governor =
        EdgeWaveGovernor::for_wave(armed_wall(now, None, Duration::from_secs(60)), now);
    let mut budget = 64usize;
    let armed_verified = escalate_edge_domain(
        &verifier,
        &graph,
        &mut armed,
        &spec,
        &thresholds,
        &clause_sizes,
        None,
        now + HORIZON,
        &mut budget,
        &mut governor,
    );

    assert_eq!(armed_verified, dark_verified);
    assert_eq!(budget, dark_budget);
    // ENGAGEMENT, or the parity below is vacuous: if the edge gate refused
    // this domain, neither pass would run and the bounds would agree
    // trivially. Both the unarmed and the armed α pass must have started.
    assert_eq!(dark_budget, 63, "the unarmed α pass never engaged: VOID");
    assert_eq!(
        governor.alpha_passes(),
        1,
        "the armed α pass never engaged: VOID"
    );
    for (a, d) in armed.obj_bounds.iter().zip(&dark.obj_bounds) {
        assert!(
            (a.0 - d.0).abs() <= 1e-6,
            "armed lower {} vs unarmed {}",
            a.0,
            d.0
        );
    }
    assert_contained(&graph, &armed.obj_bounds, "armed");
    assert_contained(&graph, &dark.obj_bounds, "dark");
}

/// DARK PARITY: the shared seam, unarmed, IS the historical inline sequence —
/// α pass (budget-gated), monotone merge, then the MILP escalation, both on
/// the same deadline.
#[test]
fn dark_escalation_is_the_historical_sequence() {
    let _serial = crate::random::restart_offset_test_lock();
    let graph = sub_towers_graph();
    let verifier = edge_verifier(true, true);
    let spec = band_spec();
    let thresholds = violated_thresholds(&graph);
    let clause_sizes = [1usize, 1usize];
    let now = Instant::now();
    let deadline = now + HORIZON;

    let mut seam = domain(&graph);
    let mut governor = EdgeWaveGovernor::for_wave(BabWall::clamped(HORIZON, None, now), now);
    let mut seam_budget = 64usize;
    let seam_verified = escalate_edge_domain(
        &verifier,
        &graph,
        &mut seam,
        &spec,
        &thresholds,
        &clause_sizes,
        None,
        deadline,
        &mut seam_budget,
        &mut governor,
    );

    // The historical inline blocks, verbatim.
    let mut inline = domain(&graph);
    let mut inline_budget = 64usize;
    let mut inline_verified = false;
    if let Some(rows) = edge_domain_rows(
        &inline,
        &thresholds,
        &clause_sizes,
        verifier.config.input_split_edge_milp_gap,
        verifier.config.input_split_edge_milp_depth,
        spec.nrows(),
    ) {
        inline_budget -= 1;
        if let Some(fresh) =
            try_edge_alpha_pass(&verifier, &graph, &inline, &spec, &rows, None, deadline)
        {
            inline.obj_bounds =
                super::super::super::batching::tighten_obj_lower_bounds(&inline.obj_bounds, fresh);
            inline_verified = super::super::super::grouped_semantics::disjunctive_domain_verified(
                &inline.obj_bounds,
                &thresholds,
                &clause_sizes,
            );
        }
    }
    if !inline_verified {
        inline_verified = try_edge_milp_escalation(
            &verifier,
            &graph,
            &inline,
            &spec,
            &thresholds,
            &clause_sizes,
            None,
            deadline,
        );
    }

    assert_eq!(seam_verified, inline_verified);
    assert_eq!(seam_budget, inline_budget);
    // ENGAGEMENT, or the parity is vacuous: the historical α block must have
    // run (a refused edge gate would make both sides trivially equal).
    assert_eq!(
        inline_budget, 63,
        "the historical α block never engaged: VOID"
    );
    assert_eq!(seam.obj_bounds, inline.obj_bounds);
    // A dark governor records nothing.
    assert_eq!(governor.alpha_passes(), 0);
    assert_eq!(governor.skipped(), 0);
    assert_eq!(governor.spent(), Duration::ZERO);
}

// ---------------------------------------------------------------------------
// The governed edge-MILP escalation
// ---------------------------------------------------------------------------

/// Records every consult; answers with a fixed verdict.
struct RecordingOracle {
    verdict: fn() -> GraphMipLeafVerdict,
    consults: AtomicUsize,
    node_bounds: Mutex<Option<HashMap<String, BoundedTensor>>>,
    /// The deadline handed over on each consult, in order.
    deadlines: Mutex<Vec<Option<Instant>>>,
}

impl RecordingOracle {
    fn new(verdict: fn() -> GraphMipLeafVerdict) -> Arc<Self> {
        Arc::new(Self {
            verdict,
            consults: AtomicUsize::new(0),
            node_bounds: Mutex::new(None),
            deadlines: Mutex::new(Vec::new()),
        })
    }
}

impl GraphMipLeafOracle for RecordingOracle {
    fn solve_leaf(&self, req: &GraphMipLeafRequest<'_>) -> GraphMipLeafVerdict {
        self.consults.fetch_add(1, Ordering::SeqCst);
        *self.node_bounds.lock().unwrap() = Some(
            req.node_bounds
                .iter()
                .map(|(name, bounds)| (name.clone(), (**bounds).clone()))
                .collect(),
        );
        self.deadlines.lock().unwrap().push(req.deadline);
        (self.verdict)()
    }
}

fn verified_all_rows() -> GraphMipLeafVerdict {
    GraphMipLeafVerdict::VerifiedAllRows
}

fn undecided() -> GraphMipLeafVerdict {
    GraphMipLeafVerdict::Undecided
}

fn violated() -> GraphMipLeafVerdict {
    GraphMipLeafVerdict::Violated {
        witness: vec![0.0, 0.0],
        output: vec![0.0],
    }
}

fn milp_verifier(alpha: bool, oracle: Arc<RecordingOracle>) -> BetaCrownVerifier {
    edge_verifier(alpha, true).with_graph_mip_leaf_oracle(oracle)
}

/// A domain that ran edge-α pays ONE node collection: the MILP request gets
/// α's own collection, and it equals a fresh collection over the same exact
/// box at the same effort deadline.
#[test]
fn governed_milp_reuses_the_alpha_collection_instead_of_recollecting() {
    let _serial = crate::random::restart_offset_test_lock();
    let graph = sub_towers_graph();
    let oracle = RecordingOracle::new(undecided);
    let verifier = milp_verifier(true, oracle.clone());
    let spec = band_spec();
    let thresholds = violated_thresholds(&graph);
    let clause_sizes = [1usize, 1usize];
    let now = Instant::now();
    let mut governor =
        EdgeWaveGovernor::for_wave(armed_wall(now, None, Duration::from_secs(60)), now);
    let mut d = domain(&graph);
    let mut budget = 64usize;
    let verified = escalate_edge_domain(
        &verifier,
        &graph,
        &mut d,
        &spec,
        &thresholds,
        &clause_sizes,
        None,
        now + HORIZON,
        &mut budget,
        &mut governor,
    );
    assert!(!verified);
    assert_eq!(governor.alpha_passes(), 1);
    assert_eq!(governor.milp_reuses(), 1, "the α collection must be reused");
    assert_eq!(governor.milp_collects(), 0, "no second collection");
    assert_eq!(governor.milp_consults(), 1);
    assert_eq!(oracle.consults.load(Ordering::SeqCst), 1);

    let received = oracle
        .node_bounds
        .lock()
        .unwrap()
        .take()
        .expect("consulted");
    let fresh = crate::network::collect_intermediate_bounds(
        &graph,
        d.input_bounds.as_ref(),
        Some(now + HORIZON),
        None,
    )
    .expect("fresh collection");
    assert_eq!(
        received.keys().collect::<std::collections::BTreeSet<_>>(),
        fresh.keys().collect::<std::collections::BTreeSet<_>>()
    );
    for (name, bounds) in &fresh {
        let got = &received[name];
        for (a, b) in got.lower().iter().zip(bounds.lower().iter()) {
            assert!((a - b).abs() <= 1e-6, "{name}: lower {a} vs fresh {b}");
        }
        for (a, b) in got.upper().iter().zip(bounds.upper().iter()) {
            assert!((a - b).abs() <= 1e-6, "{name}: upper {a} vs fresh {b}");
        }
    }
}

/// Without an α pass the governed MILP collects exactly once, as the unarmed
/// path does.
#[test]
fn governed_milp_collects_when_edge_alpha_did_not_run() {
    let graph = sub_towers_graph();
    let oracle = RecordingOracle::new(undecided);
    let verifier = milp_verifier(false, oracle.clone());
    let now = Instant::now();
    let mut governor =
        EdgeWaveGovernor::for_wave(armed_wall(now, None, Duration::from_secs(60)), now);
    let d = domain(&graph);
    let accepted = try_edge_milp_escalation_governed(
        &verifier,
        &graph,
        &d,
        &band_spec(),
        &violated_thresholds(&graph),
        &[1, 1],
        None,
        now + HORIZON,
        None,
        &mut governor,
    );
    assert!(!accepted);
    assert_eq!(governor.milp_collects(), 1);
    assert_eq!(governor.milp_reuses(), 0);
    assert_eq!(oracle.consults.load(Ordering::SeqCst), 1);
}

/// Only the certified `VerifiedAllRows` counts; an advisory violation and
/// `Undecided` never do. (A plumbing test: the property is true, and the mock
/// stands in for the certified oracle's contract.)
#[test]
fn governed_milp_counts_only_verified_all_rows() {
    let graph = sub_towers_graph();
    let thresholds = true_thresholds(&graph);
    for (verdict, expected) in [
        (verified_all_rows as fn() -> GraphMipLeafVerdict, true),
        (undecided as fn() -> GraphMipLeafVerdict, false),
        (violated as fn() -> GraphMipLeafVerdict, false),
    ] {
        let oracle = RecordingOracle::new(verdict);
        let verifier = milp_verifier(false, oracle.clone());
        let now = Instant::now();
        let mut governor =
            EdgeWaveGovernor::for_wave(armed_wall(now, None, Duration::from_secs(60)), now);
        let accepted = try_edge_milp_escalation_governed(
            &verifier,
            &graph,
            &domain(&graph),
            &band_spec(),
            &thresholds,
            &[1, 1],
            None,
            now + HORIZON,
            None,
            &mut governor,
        );
        assert_eq!(accepted, expected);
        assert_eq!(oracle.consults.load(Ordering::SeqCst), 1);
    }
}

/// NO AUTHORITY PAST THE STOP: with the scored stop already passed, even an
/// oracle that would certify every row is never consulted and nothing counts.
#[test]
fn governed_milp_declines_everything_after_the_stop() {
    let graph = sub_towers_graph();
    let oracle = RecordingOracle::new(verified_all_rows);
    let verifier = milp_verifier(false, oracle.clone());
    let now = Instant::now();
    let passed = BabWall::clamped(
        HORIZON,
        Some(now.checked_sub(Duration::from_secs(1)).unwrap()),
        now,
    )
    .with_edge_wave(Some(EdgeWave {
        index: 0,
        allowance: Duration::from_secs(60),
    }));
    let mut governor = EdgeWaveGovernor::for_wave(passed, now);
    let accepted = try_edge_milp_escalation_governed(
        &verifier,
        &graph,
        &domain(&graph),
        &band_spec(),
        &violated_thresholds(&graph),
        &[1, 1],
        None,
        now + HORIZON,
        None,
        &mut governor,
    );
    assert!(!accepted);
    assert_eq!(oracle.consults.load(Ordering::SeqCst), 0);
}

/// A spent allowance refuses the consult (and counts the refusal), whatever
/// the oracle would have said.
#[test]
fn governed_milp_is_refused_once_the_allowance_is_spent() {
    let graph = sub_towers_graph();
    let oracle = RecordingOracle::new(verified_all_rows);
    let verifier = milp_verifier(false, oracle.clone());
    let now = Instant::now();
    let mut governor = EdgeWaveGovernor::for_wave(armed_wall(now, None, Duration::ZERO), now);
    let accepted = try_edge_milp_escalation_governed(
        &verifier,
        &graph,
        &domain(&graph),
        &band_spec(),
        &violated_thresholds(&graph),
        &[1, 1],
        None,
        now + HORIZON,
        None,
        &mut governor,
    );
    assert!(!accepted);
    assert_eq!(governor.skipped(), 1);
    assert_eq!(
        governor.milp_collects(),
        0,
        "no collection is paid for a refused step"
    );
    assert_eq!(oracle.consults.load(Ordering::SeqCst), 0);
}

/// The deadline the oracle is handed is never later than the scored stop,
/// even under an allowance that reaches past it.
#[test]
fn governed_milp_hands_the_oracle_a_deadline_inside_the_stop() {
    let graph = sub_towers_graph();
    let oracle = RecordingOracle::new(undecided);
    let verifier = milp_verifier(false, oracle.clone());
    let now = Instant::now();
    let stop = Duration::from_secs(5);
    let mut governor =
        EdgeWaveGovernor::for_wave(armed_wall(now, Some(stop), Duration::from_secs(60)), now);
    let _ = try_edge_milp_escalation_governed(
        &verifier,
        &graph,
        &domain(&graph),
        &band_spec(),
        &violated_thresholds(&graph),
        &[1, 1],
        None,
        now + HORIZON,
        None,
        &mut governor,
    );
    let deadline = oracle
        .deadlines
        .lock()
        .unwrap()
        .last()
        .copied()
        .expect("consulted")
        .expect("a deadline is always handed over");
    assert!(
        deadline <= now + stop,
        "the oracle may not run past the stop"
    );
}
