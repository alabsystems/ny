// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #router-r4 review: the lowdim lane's CR-F safe point END TO END, through
//! the PROCESS-GLOBAL FAR and deadline authorities the lane itself consults.
//!
//! The builder's suite exercises the FAR authority on synthetic rings and the
//! lane under an UNARMED authority (every `at_safe_point` answers `Continue`),
//! so the one line that turns a hand-on into the lane's outcome —
//! `return finish(.., Some(LowDimStop::HandOn))` in `prove_conjunction_core`
//! — was reached by no test: a mutant that returned `None` there (which
//! `finish` maps to `LowDimOutcome::Verified`, i.e. a PROOF) survived every
//! test. This file drives the real lane into a live hand-on and pins that the
//! stopped lane reports UNDECIDED, never a proof.
//!
//! It is its OWN test process on purpose: it arms the process-global FAR and
//! deadline authorities, which a shared unit-test binary must never do
//! (another test's lane would read them). Its one test runs serially.

use std::time::{Duration, Instant};

use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_core::deadline_authority::{self, Frame, LaneSlot, SLOTS};
use ny_core::far_authority::{self, FarAction};
use ny_core::router::checkpoint::CheckpointId;
use ny_core::router::decl::{ids, AmendmentSet, LaneRole, ProgressKind, SafePoints, WorkUnit};
use ny_core::router::facts::{DeviceQual, HostClass, HostEnvelope, InstanceFacts};
use ny_core::router::plan::{
    self, AdmissionState, BudgetView, CapClass, CapInput, LedgerState, PolicyInput,
};
use ny_core::router::r4::{FarCommit, FarRoute};
use ny_core::router::safepoint::At;
use ny_core::router::{PRIORS_VERSION, ROUTER_SCHEMA};
use ny_propagate::layers::{LinearLayer, ReLULayer};
use ny_propagate::lowdim::{prove_objective_lower_bound, LowDimConfig, LowDimOutcome, LowDimStop};
use ny_propagate::{GraphNetwork, GraphNode, Layer, NETWORK_INPUT};
use ny_tensor::BoundedTensor;

/// A 2-input ReLU net whose first output is bounded far above -1000 on the
/// unit box: the lane proves `y0 > -1000` at its root, in one wave.
fn easy_graph() -> GraphNetwork {
    let l1 =
        Array2::from_shape_vec((3, 2), vec![1.0f32, -0.5, 0.25, 1.0, -1.0, 0.5]).expect("shape");
    let l2 =
        Array2::from_shape_vec((2, 3), vec![0.5f32, -1.0, 0.75, 1.0, 0.5, -0.25]).expect("shape");
    let mut g = GraphNetwork::new();
    g.add_node(GraphNode::new(
        "l1",
        Layer::Linear(
            LinearLayer::new(l1, Some(Array1::from(vec![0.1f32, -0.2, 0.3]))).expect("l1"),
        ),
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

fn unit_box() -> BoundedTensor {
    let lo = ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![0.0f32, 0.0]).expect("lo");
    let hi = ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![1.0f32, 1.0]).expect("hi");
    BoundedTensor::new(lo, hi).expect("box")
}

fn run_lane() -> LowDimOutcome {
    prove_objective_lower_bound(
        &easy_graph(),
        &unit_box(),
        &[1.0, 0.0],
        -1000.0,
        Instant::now() + Duration::from_secs(30),
        &LowDimConfig::default(),
    )
}

fn cap(id: &str, ledger: LedgerState, progress: ProgressKind) -> CapInput {
    CapInput {
        id: id.to_owned(),
        class: CapClass::Lane(LaneRole::Prover { terminal_ok: true }),
        ledger,
        progress,
        unit: WorkUnit::Level,
        first_rung_ms: None,
        safe_points: SafePoints::BetweenUnits,
        accepts: AmendmentSet {
            deadline: true,
            hand_on_source: true,
            hand_on_sink: true,
            defer: false,
        },
        admission: AdmissionState::Admit,
        // #router-r5: a lane carries no routing data.
        route: None,
    }
}

/// A 100 s row (not CR-S short) whose lowdim source has a STRICTLY sealed
/// recipient after it; c2 at 5 s of the row clock, c3 never reached.
fn plan_with_sealed_sink() -> plan::Plan {
    let input = PolicyInput {
        schema: ROUTER_SCHEMA.to_owned(),
        facts: InstanceFacts::known_empty(),
        host: HostEnvelope {
            class: HostClass::Other,
            threads: 8,
            mem_bytes: 64 << 30,
            reserve_bytes: 16 << 30,
            device: DeviceQual::None,
            declared: true,
        },
        budget: BudgetView {
            official_ms: 100_000,
            internal_ms: 95_000,
            join_margin_ms: 2_000,
        },
        capabilities: vec![
            cap(
                ids::LOWDIM.as_str(),
                LedgerState::SealedAsShipped,
                ProgressKind::Volume,
            ),
            cap("sink", LedgerState::Sealed, ProgressKind::Opaque),
        ],
        pins: Vec::new(),
        priors_version: PRIORS_VERSION.to_owned(),
        registry_version: "review00".to_owned(),
        priors: None,
    };
    let mut p = plan::plan(&input);
    assert!(!p.short_row, "a 100 s row is not CR-S short");
    for c in &mut p.checkpoints {
        match c.id {
            CheckpointId::C1 => c.at_ms = 0,
            CheckpointId::C2 => c.at_ms = 5_000,
            CheckpointId::C3 => c.at_ms = u64::MAX / 4,
        }
    }
    p
}

#[ntest::timeout(120000)]
#[test]
fn router_r4_review_lowdim_hand_on_reports_undecided_never_a_proof() {
    // Positive control, unarmed: the lane PROVES this row (so a hand-on that
    // leaked into a `None` stop would read as this same proof).
    let control = run_lane();
    assert!(
        matches!(control, LowDimOutcome::Verified(_)),
        "control: the unarmed lane proves the easy row: {control:?}"
    );

    // The row started 100 s ago; the source's (verifier's) cell runs to
    // 900 s, the recipient's slot has not started.
    let anchor = Instant::now()
        .checked_sub(Duration::from_secs(100))
        .expect("the monotonic clock is past 100 s");
    let cells = deadline_authority::global();
    cells
        .arm(
            Frame::new(
                anchor,
                anchor + Duration::from_secs(1_000),
                [Duration::from_secs(5); SLOTS],
                Duration::from_secs(2),
            ),
            [true; SLOTS],
            Some(LaneSlot::Verifier),
            None,
        )
        .expect("arm the deadline cells");
    cells
        .claim_deadline(
            LaneSlot::Verifier,
            anchor + Duration::from_mins(15),
            Instant::now(),
        )
        .expect("the source's cell is claimed");
    let far = far_authority::global();
    far.arm(
        plan_with_sealed_sink(),
        FarCommit {
            routes: vec![FarRoute {
                from: ids::LOWDIM.as_str().to_owned(),
                to: "sink".to_owned(),
                from_slot: LaneSlot::Verifier,
                to_slot: LaneSlot::MarginRowInline,
            }],
            observe_only: false,
        },
        anchor,
    );
    // Three earlier lowdim wave boundaries BEFORE c2 (row clock 1-3 s): the
    // certified volume is flat while the queue doubles — FAR in work units.
    for (i, open) in [4u64, 8, 16].into_iter().enumerate() {
        let unit = i as u64;
        let t = 1.0 + i as f64;
        let action = far.decide_at(
            far_authority::volume_sample(ids::LOWDIM, unit, 0.4, open),
            At::Unit(unit),
            t,
            cells,
            anchor + Duration::from_secs_f64(t),
        );
        assert_eq!(action, FarAction::Continue, "nothing is due before c2");
    }

    // The real lane, past c2: its first safe point evaluates c2 on the FAR
    // window, the cells hand its unspent contract on, and the lane STOPS.
    let stopped = run_lane();
    match &stopped {
        LowDimOutcome::Unproved(stats, LowDimStop::HandOn) => {
            assert_eq!(stats.waves, 0, "stopped at its first wave boundary");
            assert_eq!(stats.closed, 0, "no leaf was closed");
        }
        other => {
            panic!("a handed-on lane must report UNDECIDED (HandOn), never a proof: {other:?}")
        }
    }
    let snap = far.snapshot();
    assert_eq!(snap.handed, 1, "{snap:?}");
    assert!(
        snap.events.iter().any(|e| e.handed && e.decision.act),
        "{snap:?}"
    );
    let ledger = cells.snapshot();
    let hand = ledger
        .hand_ons
        .iter()
        .find(|h| h.far)
        .expect("the hand-on is ledgered");
    assert_eq!(hand.from, LaneSlot::Verifier);
    assert_eq!(hand.to, Some(LaneSlot::MarginRowInline));
    assert!(
        hand.credited_ns <= hand.unspent_ns,
        "time is conserved: {hand:?}"
    );

    // The source stays stopped for the row: a later lowdim call stops at its
    // first boundary too, again undecided, and nothing is handed twice.
    let again = run_lane();
    assert!(
        matches!(again, LowDimOutcome::Unproved(_, LowDimStop::HandOn)),
        "{again:?}"
    );
    assert_eq!(
        cells.snapshot().hand_ons.iter().filter(|h| h.far).count(),
        1,
        "handed once"
    );

    // Disarmed (a fault, or the session's end): today's lane, the proof again.
    far.disarm();
    cells.disarm();
    let after = run_lane();
    assert!(
        matches!(after, LowDimOutcome::Verified(_)),
        "disarmed, the lane is today's lane: {after:?}"
    );
}
