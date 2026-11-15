// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #router-r3 (ROUTER_PROGRAM §10 R3) END TO END on a small margin-row
//! instance: the acting optimizer's `rule.governor` stops the root alpha
//! ascent through the process's stop authority, and the time it frees reaches
//! the downstream phase (the tree) inside the lane's own deadline — the one
//! the R2 deadline cell recorded, which no O-STOP rule ever writes.
//!
//! This file is its OWN test process on purpose: it arms the process-global
//! stop and deadline authorities, which a shared unit-test binary must never
//! do (another test's lane would read them). Its one test runs serially.

use std::time::{Duration, Instant};

use ny_core::deadline_authority::{self, Frame, LaneSlot, SLOTS};
use ny_core::stop_authority::{self, RatioRule, StopDirectives, StopSite};
use ny_propagate::margin_row::{
    run_margin_row_lane_with_policy, MarginRowOutcome, MarginRowRuntimePolicy, TwinOpSpec, TwinSpec,
};

/// SplitMix64 weights in `[-scale, scale]` (dependency-free, deterministic).
struct Rng(u64);

impl Rng {
    fn unit(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }

    fn weights(&mut self, n: usize, scale: f64) -> Vec<f64> {
        (0..n).map(|_| (2.0 * self.unit() - 1.0) * scale).collect()
    }
}

/// The margin-row unit tests' tiny twin-wall net: 2x6x6 input, a residual
/// block, two trunk relus, a 5-wide head, 4 classes.
fn tiny_spec(seed: u64, scale: f64) -> TwinSpec {
    let mut rng = Rng(seed);
    let conv1 = TwinOpSpec::Conv {
        input: 0,
        weight: rng.weights(4 * 2 * 3 * 3, scale),
        bias: rng.weights(4, scale),
        bias_err: vec![0.0; 4],
        weight_rel_err: 1e-15,
        kernel: (4, 2, 3, 3),
        stride: (2, 2),
        pads: (1, 1, 1, 1),
        ishape: (2, 6, 6),
        oshape: (4, 3, 3),
    };
    let conv2 = TwinOpSpec::Conv {
        input: 2,
        weight: rng.weights(4 * 4 * 3 * 3, scale),
        bias: rng.weights(4, scale),
        bias_err: vec![1e-18; 4],
        weight_rel_err: 1e-15,
        kernel: (4, 4, 3, 3),
        stride: (1, 1),
        pads: (1, 1, 1, 1),
        ishape: (4, 3, 3),
        oshape: (4, 3, 3),
    };
    TwinSpec {
        n_in: 72,
        ops: vec![
            conv1,
            TwinOpSpec::Relu { input: 1 },
            conv2,
            TwinOpSpec::Add { lhs: 3, rhs: 2 },
            TwinOpSpec::Relu { input: 4 },
            TwinOpSpec::Flatten { input: 5 },
            TwinOpSpec::Gemm {
                input: 6,
                weight: rng.weights(5 * 36, scale),
                bias: rng.weights(5, scale),
                bias_err: Vec::new(),
                weight_rel_err: 0.0,
                shape: (5, 36),
            },
            TwinOpSpec::Relu { input: 7 },
            TwinOpSpec::Gemm {
                input: 8,
                weight: rng.weights(4 * 5, scale),
                bias: rng.weights(4, scale),
                bias_err: Vec::new(),
                weight_rel_err: 0.0,
                shape: (4, 5),
            },
        ],
    }
}

fn verdict_token(outcome: &MarginRowOutcome) -> &'static str {
    match outcome {
        MarginRowOutcome::Unsat(..) => "unsat",
        MarginRowOutcome::Unknown { .. } => "unknown",
    }
}

fn frame_for(anchor: Instant, hard: Instant) -> Frame {
    Frame::new(
        anchor,
        hard,
        [
            Duration::from_secs(5),
            Duration::from_secs(5),
            Duration::from_secs(5),
            Duration::from_secs(5),
            Duration::from_secs(3),
            Duration::from_secs(3),
        ],
        Duration::from_secs(2),
    )
}

#[test]
fn router_r3_governor_yields_and_the_freed_time_reaches_the_tree() {
    let levers = ny_levers::LeverSet::freeze(ny_levers::all());
    let mut policy =
        MarginRowRuntimePolicy::from_levers(&levers).expect("the complete workspace policy");
    policy.root_alpha.alpha_enabled = true;
    policy.root_alpha.alpha_iters = 30;
    policy.root_alpha.alpha_secs = 60.0;
    let stop = stop_authority::global();
    let cells = deadline_authority::global();
    let governor = StopDirectives {
        governor: Some(RatioRule {
            ratio: 0.15,
            min_samples: 5,
        }),
        ..StopDirectives::NONE
    };
    let mut engaged = 0usize;
    for seed in 0..8u64 {
        let spec = tiny_spec(0x5eed_3300 + seed, 0.4 + (seed % 4) as f64 * 0.2);
        let half = 0.35 + (seed % 3) as f64 * 0.1;
        let lo = vec![-half; 72];
        let hi = vec![half; 72];
        let adv = [1usize, 2, 3];

        // Unarmed: nothing is recorded and nothing is printed by the stop
        // authority — today's lane.
        stop.disarm();
        cells.disarm();
        let recorded_before = stop.snapshot().yields.len();
        let deadline = Instant::now() + Duration::from_secs(60);
        let off = run_margin_row_lane_with_policy(
            &spec,
            &lo,
            &hi,
            0,
            &adv,
            Some(deadline),
            2_000,
            policy,
        );
        assert_eq!(
            stop.snapshot().yields.len(),
            recorded_before,
            "seed {seed}: an unarmed stop authority records nothing"
        );

        // Armed: the R2 cell records the lane's deadline (no safe points:
        // never moved), and the stop authority arms `rule.governor` only.
        let anchor = Instant::now();
        let hard = anchor + Duration::from_secs(63);
        let deadline = anchor + Duration::from_secs(60);
        cells
            .arm(
                frame_for(anchor, hard),
                [true; SLOTS],
                Some(LaneSlot::Verifier),
                None,
            )
            .expect("arm the R2 cells");
        cells
            .claim_deadline(LaneSlot::MarginRowInline, deadline, anchor)
            .expect("claim the lane's deadline");
        let cell_before = cells.deadline(LaneSlot::MarginRowInline);
        stop.arm(governor).expect("arm rule.governor");
        let t_start = Instant::now();
        let on = run_margin_row_lane_with_policy(
            &spec,
            &lo,
            &hi,
            0,
            &adv,
            Some(deadline),
            2_000,
            policy,
        );
        let wall = t_start.elapsed();
        let snapshot = stop.snapshot();
        // One direction: the R2 cell is exactly what the lane claimed.
        assert_eq!(
            cells.deadline(LaneSlot::MarginRowInline),
            cell_before,
            "seed {seed}: an O-STOP stop never writes a lane's deadline cell"
        );
        // Sound either way: the lane only ever publishes unsat or unknown.
        let _ = (verdict_token(&off), verdict_token(&on));
        for y in &snapshot.yields {
            assert_eq!(y.site, StopSite::AlphaAscent, "seed {seed}: {y:?}");
            assert!(
                y.freed_ms <= y.budget_ms,
                "seed {seed}: a yield frees at most its own budget: {y:?}"
            );
        }
        if !snapshot.yields.is_empty() {
            engaged += 1;
            // Every freed slice was handed to the tree, at its start, inside
            // the lane's own deadline.
            assert_eq!(
                snapshot.hand_ons.len(),
                snapshot.yields.len(),
                "seed {seed}: every yield reaches a recipient: {snapshot:?}"
            );
            for h in &snapshot.hand_ons {
                assert_eq!(h.to, "margin-row.tree", "seed {seed}");
                assert_eq!(h.rule, "rule.governor", "seed {seed}");
                let window = h.window_ms.expect("the tree has the lane's deadline");
                assert!(
                    window <= 60_000,
                    "seed {seed}: the window is the lane's own"
                );
                assert!(
                    h.realized && h.credited_ms == h.freed_ms,
                    "seed {seed}: the freed time is inside the tree's window: {h:?}"
                );
                assert!(h.freed_ms > 0, "seed {seed}: the yield freed time: {h:?}");
            }
            // The lane finished well inside the deadline it never left.
            assert!(wall < Duration::from_secs(60), "seed {seed}: {wall:?}");
        }
        stop.disarm();
        cells.disarm();
    }
    assert!(
        engaged > 0,
        "rule.governor never yielded on any instance: VOID, not a pass"
    );
}
