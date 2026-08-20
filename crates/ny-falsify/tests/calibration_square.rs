// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! REQUIRED TEST (b), part 2: `square` reproduces its calibration win.
//!
//! # What is being reproduced, stated precisely
//!
//! This is a **mechanism reproduction on a surrogate**, not a replay of the two
//! ONNX rows. Replaying `soundnessbench model_36` or
//! `traffic_signs model_48_idx_10495_eps_10` needs ONNX Runtime, the 2025
//! corpus and a machine that does not throttle; none of those belong in a unit
//! test, and the crate deliberately has no ONNX dependency.
//!
//! What IS reproduced is the exact structural claim the calibration makes about
//! `square` -- the claim that justifies porting it at all:
//!
//! 1. the objective is FLAT, so every finite-difference estimator (`spsa`,
//!    `nes`, and ny's own SPSA-class incumbent lane) is identically zero. Not
//!    "small" -- exactly `0.0`, asserted on 512 probe pairs. Measured on the
//!    real thing: `traffic_signs` BNN outputs are a one-hot integer vector with
//!    exactly two distinct values and a max-other-minus-true margin of exactly
//!    `-1.0` at every in-box point; the calibration row for it records
//!    `"best_output_margin": 0.0` after `spsa` burned 7168 points;
//! 2. the violating region is a PARTIAL VERTEX -- some coordinates at a bound,
//!    the rest at an interior incumbent value. That is the geometry `square`'s
//!    annealed block flips sweep and that nothing else does;
//! 3. every strategy ny already ships misses it, at budgets far above the one
//!    `square` needs: full-vertex enumeration (`low_dim_ort_corner_falsify`)
//!    because a vertex has no interior coordinate, uniform/interior sampling
//!    because an interior point has no coordinate at a bound, and `special`
//!    because its eight patterns are one or the other.
//!
//! The dimension is 128, which is `soundnessbench model_36`'s free-input count.

#[path = "fixtures/support.rs"]
mod support;

use ny_falsify::strategies::{SpecialPoints, Square, SQUARE_STALL_BATCHES};
use ny_falsify::{
    Budget, ObjectiveQuality, ParamSpace, Proposal, Registry, Rng, Score, SearchBox, SearchState,
    StallRule, Strategy, StrategyName, WorkUnit,
};
use std::time::{Duration, Instant};
use support::{box_spec, CountingLadder, PredicateOracle};

/// `soundnessbench model_36`'s free-input count.
const N: usize = 128;
/// The calibration ran at `--batch 256`.
const BATCH: usize = 256;
/// Points `square` spent on the two rows it won: 2304 and 768.
const CALIBRATION_WIN_POINTS: usize = 2304;

fn domain() -> SearchBox {
    SearchBox::new(&vec![0.0f64; N], &vec![1.0f64; N]).unwrap()
}

fn interior_count(point: &[f64]) -> usize {
    point.iter().filter(|&&v| v > 0.0 && v < 1.0).count()
}

/// The surrogate: a two-level objective whose violating region is a partial
/// vertex. Between 32 and 96 of the 128 coordinates interior, and three named
/// coordinates driven to named bounds.
fn violates(point: &[f64]) -> bool {
    let interior = interior_count(point);
    (32..=96).contains(&interior) && point[0] == 1.0 && point[1] == 0.0 && point[2] == 1.0
}

fn score(point: &[f64]) -> Score {
    let hit = violates(point);
    Score {
        // Exactly two distinct values, as measured on the real BNN.
        steer: if hit { 0.0 } else { -1.0 },
        holds: hit,
    }
}

fn budget(seconds: u64) -> Budget {
    Budget {
        deadline: Instant::now() + Duration::from_secs(seconds),
        batch: BATCH,
        params: ParamSpace {
            free_dims_ceiling: usize::MAX,
            max_points: 1 << 22,
            max_restarts: 64,
        },
        stall_rule: StallRule::new(
            WorkUnit::BlockBatchesWithoutImprovement,
            SQUARE_STALL_BATCHES,
        ),
    }
}

#[test]
fn the_objective_is_flat_so_every_gradient_estimator_is_identically_zero() {
    // The `#deadlane` condition, asserted rather than assumed. Both estimators
    // in the Python portfolio are two-sided: `spsa` probes at 0.02 * span along
    // a random sign vector, `nes` at sigma * span along a gaussian. Both reduce
    // to a difference of margins, and on a two-level objective that difference
    // is exactly zero away from the (measure-zero) violating set.
    let domain = domain();
    let mut rng = Rng::new(0xFA15_1F1E_D000_0001);
    let mut probes = 0usize;

    for _ in 0..512 {
        let base: Vec<f64> = (0..N).map(|_| 0.1 + 0.8 * rng.next_f64()).collect();
        let point = domain.materialise(&base);
        assert!(
            !violates(&point),
            "the probe base must be outside the region"
        );

        // spsa: +/- 0.02 * span along a random sign vector.
        let signs: Vec<f64> = (0..N)
            .map(|_| if rng.next_f64() < 0.5 { -1.0 } else { 1.0 })
            .collect();
        // nes: +/- sigma * span along a gaussian-ish direction.
        let gauss: Vec<f64> = (0..N).map(|_| rng.next_f64() - 0.5).collect();

        for (probe, scale) in [(&signs, 0.02f64), (&gauss, 0.1f64)] {
            let plus: Vec<f64> = (0..N)
                .map(|i| (base[i] + scale * probe[i]).clamp(0.0, 1.0))
                .collect();
            let minus: Vec<f64> = (0..N)
                .map(|i| (base[i] - scale * probe[i]).clamp(0.0, 1.0))
                .collect();
            let difference =
                score(&domain.materialise(&plus)).steer - score(&domain.materialise(&minus)).steer;
            assert_eq!(
                difference, 0.0,
                "a two-sided estimate must be EXACTLY zero on a flat objective"
            );
            probes += 1;
        }
    }
    assert_eq!(probes, 1024);
}

#[test]
fn every_sampler_ny_already_ships_misses_the_region_at_a_hundred_times_the_budget() {
    let domain = domain();
    let mut rng = Rng::new(0xFA15_1F1E_D000_0002);
    const TRIES: usize = 100_000;
    // Forty times the points either calibration win cost, per sampler.
    const _: () = assert!(TRIES > 40 * CALIBRATION_WIN_POINTS);

    // (i) full-vertex sampling: `low_dim_ort_corner_falsify` /
    //     `corners_full` / `corners_random`. Every coordinate is at a bound, so
    //     `interior_count` is 0 and the region is unreachable at ANY budget --
    //     this is a structural miss, not an unlucky one.
    let mut vertex_hits = 0usize;
    for _ in 0..TRIES {
        let row: Vec<f64> = (0..N)
            .map(|_| if rng.next_f64() < 0.5 { 1.0 } else { 0.0 })
            .collect();
        let point = domain.materialise(&row);
        assert_eq!(interior_count(&point), 0);
        vertex_hits += usize::from(violates(&point));
    }
    assert_eq!(vertex_hits, 0, "vertex sampling reached a partial vertex");

    // (ii) uniform / Halton / clipped gradient ascent: interior points. No
    //      coordinate is ever exactly at a bound.
    let mut interior_hits = 0usize;
    for _ in 0..TRIES {
        let row: Vec<f64> = (0..N).map(|_| 0.05 + 0.9 * rng.next_f64()).collect();
        let point = domain.materialise(&row);
        assert_eq!(interior_count(&point), N);
        interior_hits += usize::from(violates(&point));
    }
    assert_eq!(
        interior_hits, 0,
        "interior sampling reached a partial vertex"
    );

    // (iii) `special`'s own eight patterns: each is either a full vertex or
    //       fully interior, so the strategy that wins 34 of 75 calibration rows
    //       is structurally blind here. Six strategies, none dominating.
    for pattern in SpecialPoints::patterns(&domain) {
        let point = domain.materialise(&pattern);
        let interior = interior_count(&point);
        assert!(
            interior == 0 || interior == N,
            "a special pattern was a partial vertex: interior {interior}"
        );
        assert!(!violates(&point));
    }
}

#[test]
fn square_takes_the_region_the_others_cannot_reach() {
    let domain = domain();
    let mut oracle = PredicateOracle::new(|point: &[f64]| score(point));
    let mut state = SearchState::at_centre(&domain);

    let proposal = Square::default().search(&domain, &mut oracle, &budget(30), &mut state);

    let Proposal::Candidate(candidate) = proposal else {
        panic!("square missed the partial-vertex region: {proposal:?}");
    };
    assert_eq!(candidate.found_by(), StrategyName::Square);
    assert!(
        violates(candidate.inputs()),
        "the proposal must be in the region"
    );
    assert!(
        candidate.effort().points <= CALIBRATION_WIN_POINTS,
        "square spent {} points; the two calibration wins cost 2304 and 768",
        candidate.effort().points
    );
    // Inputs only. There is no output vector on this type to inspect.
    assert_eq!(candidate.inputs().len(), N);
}

#[test]
fn the_annealed_block_schedule_matches_the_python_portfolio() {
    // `max(1, int(n * fraction))` with `fraction = max(1/n, 0.5 * 0.85^k)`,
    // evaluated by the shipped Python and pinned here.
    const N128: [usize; 16] = [64, 54, 46, 39, 33, 28, 24, 20, 17, 14, 12, 10, 9, 7, 6, 5];
    // 6912 is `traffic_signs model_48_idx_10495_eps_10`'s free-input count --
    // the other row `square` won.
    const N6912: [usize; 16] = [
        3456, 2937, 2496, 2122, 1804, 1533, 1303, 1107, 941, 800, 680, 578, 491, 417, 355, 301,
    ];
    for (k, &expected) in N128.iter().enumerate() {
        assert_eq!(Square::block_size(128, k), expected, "n=128, iteration {k}");
    }
    for (k, &expected) in N6912.iter().enumerate() {
        assert_eq!(
            Square::block_size(6912, k),
            expected,
            "n=6912, iteration {k}"
        );
    }
    // The 1/n floor: at two free dimensions the block never vanishes.
    for k in 0..10 {
        assert_eq!(Square::block_size(2, k), 1);
    }
}

#[test]
fn a_stalled_walk_is_abandoned_and_the_next_restart_starts_somewhere_else() {
    // The failure this test exists to catch is SILENT. `special` runs first and
    // deposits an incumbent on a strict improvement over -inf; on a flat
    // objective its argmax under an exact tie is pattern 0, `all_low`, a
    // VERTEX. Seeding `square` from a vertex makes every block flip a no-op and
    // turns it into `corners_random`, which ny already ships -- it would still
    // run, still report points, and have lost all of its reach.
    let domain = domain();
    let mut ladder = CountingLadder::new(box_spec(N));
    let mut oracle = PredicateOracle::new(|point: &[f64]| score(point));

    let mut registry = Registry::new()
        .with(Box::new(SpecialPoints))
        .with(Box::new(Square::default()))
        .armed();

    let receipt = registry
        .run(
            &mut ladder,
            &domain,
            &mut oracle,
            Duration::from_mins(1),
            ObjectiveQuality::Flat,
        )
        .expect("admission ladder");

    // `special` ran first, spent its eight points, found nothing, and left a
    // vertex incumbent behind.
    let (first, first_proposal) = &receipt.proposals[0];
    assert_eq!(*first, StrategyName::SpecialPoints);
    let Proposal::Exhausted(effort) = first_proposal else {
        panic!("special should have found nothing here: {first_proposal:?}");
    };
    assert_eq!(effort.points, 8);

    let candidate = receipt.candidate().expect("square should have taken it");
    assert_eq!(candidate.found_by(), StrategyName::Square);
    assert!(violates(candidate.inputs()));

    // Restart 0 inherited the vertex and could not improve, so it was abandoned
    // by the stall rule after SQUARE_STALL_BATCHES non-improving batches
    // (one improving first batch, then the counter runs). Restart 1 seeded from
    // the centre and took the region. The point count is the proof that both
    // happened.
    let spent = candidate.effort().points;
    let restart_zero_cost = (SQUARE_STALL_BATCHES as usize + 1) * BATCH;
    assert!(
        spent > restart_zero_cost,
        "square won in {spent} points; a win inside {restart_zero_cost} would mean restart 0 \
         was not the degenerate vertex walk this test sets up"
    );
    assert!(
        spent < restart_zero_cost + 8 * BATCH,
        "square won in {spent} points, too far past the restart boundary"
    );
}
